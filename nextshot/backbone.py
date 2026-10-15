"""Independent region encoder, native Qwen fusion, and joint numeric decoder."""
import math

import torch
from torch import nn
from torch.nn import functional as F

from .constants import TOKENS


def pair_features(geometry, same_semantic):
    """Source-only directed geometry; no guessed support, gaze or depth labels."""
    d = geometry[:, None, :, :3] - geometry[:, :, None, :3]
    sizes = geometry[..., 5:7].clamp_min(1e-6)
    lo = geometry[..., 3:5] - sizes/2
    hi = geometry[..., 3:5] + sizes/2
    overlap = (torch.minimum(hi[:, :, None], hi[:, None, :]) -
               torch.maximum(lo[:, :, None], lo[:, None, :])).clamp_min(0)
    inter = overlap.prod(-1)
    area = sizes.prod(-1)
    iou = inter/(area[:, :, None] + area[:, None, :] - inter).clamp_min(1e-8)
    ratio = overlap/torch.minimum(sizes[:, :, None], sizes[:, None, :]).clamp_min(1e-6)
    b, n, _ = geometry.shape
    role = geometry[..., 13]
    return torch.cat([d, torch.linalg.vector_norm(d[..., :2], dim=-1, keepdim=True),
                      iou[..., None], ratio, same_semantic[..., None],
                      role[:, :, None, None].expand(b, n, n, 1),
                      role[:, None, :, None].expand(b, n, n, 1)], dim=-1)


class RelationAttention(nn.Module):
    def __init__(self, dim, heads, dropout=0., relations=True):
        super().__init__()
        if dim % heads:
            raise ValueError('Hidden size must be divisible by attention heads')
        self.heads = heads
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim*3)
        self.out = nn.Linear(dim, dim)
        self.bias = nn.Sequential(nn.Linear(10, 32), nn.GELU(), nn.Linear(32, heads)) if relations else None
        self.ff = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim*4), nn.GELU(),
                                nn.Dropout(dropout), nn.Linear(dim*4, dim))
        self.dropout = dropout

    def forward(self, x, mask, relations):
        if not mask.any(-1).all():
            raise ValueError('Empty attention context')
        b, n, d = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(b, n, 3, self.heads, d//self.heads).permute(2, 0, 3, 1, 4)
        logits = (q @ k.transpose(-1, -2))/math.sqrt(d//self.heads)
        if self.bias is not None:
            logits = logits + self.bias(relations).permute(0, 3, 1, 2)
        logits = logits.masked_fill(~mask[:, None, None, :], -torch.inf)
        weights = logits.float().softmax(-1).to(v.dtype)
        weights = F.dropout(weights, self.dropout, self.training)
        x = x + self.out((weights @ v).transpose(1, 2).reshape(b, n, d))
        return (x + self.ff(x))*mask[..., None]


class SourceEncoder(nn.Module):
    def __init__(self, c):
        super().__init__()
        d = c['hidden_dim']
        self.visual = nn.Sequential(nn.LayerNorm(c['feature_dim']), nn.Linear(c['feature_dim'], d), nn.GELU())
        self.geometry = nn.Sequential(nn.Linear(c['geometry_dim'] + c['orientation_dim'], d), nn.GELU())
        self.fuse = nn.Sequential(nn.Linear(2*d, d), nn.LayerNorm(d))
        self.frame = nn.Parameter(torch.randn(1, 1, d)*.02)
        self.blocks = nn.ModuleList(RelationAttention(d, c['heads'], c['dropout'], c['explicit_relations'])
                                    for _ in range(c['source_layers']))

    def forward(self, g):
        mask = g['node_mask']
        if not mask.any(-1).all():
            raise ValueError('No valid source regions')
        x = self.fuse(torch.cat([self.visual(g['visual']),
                                self.geometry(torch.cat([g['geometry'], g['orientation']], -1))], -1))
        b, n, _ = x.shape
        frame_geom = g['geometry'].new_zeros((b, 1, 14))
        frame_geom[..., [0, 1, 3, 4]] = .5
        frame_geom[..., [5, 6]] = 1.
        frame_geom[..., 10] = 1.  # Depth unknown; frame is a coordinate anchor.
        frame_geom[:, 0, 12] = g['geometry'][:, 0, 12]
        geometry = torch.cat([g['geometry'], frame_geom], 1)
        valid = torch.cat([mask, mask.new_ones((b, 1))], 1)
        semantic = g['semantic_match'].new_zeros((b, n+1, n+1))
        semantic[:, :n, :n] = g['semantic_match']
        rel = pair_features(geometry, semantic)
        frame = self.frame.expand(b, -1, -1) + self.geometry(torch.cat([
            frame_geom, frame_geom.new_zeros((b, 1, 12))], -1))
        x = torch.cat([x*mask[..., None], frame], 1)
        for block in self.blocks:
            x = block(x, valid, rel)
        return x[:, :n], x[:, n], rel[:, :n, :n]


class JointLayoutDecoder(nn.Module):
    def __init__(self, c, language_dim):
        super().__init__()
        d = c['hidden_dim']
        self.region_readout = nn.Sequential(nn.LayerNorm(language_dim), nn.Linear(language_dim, d))
        self.global_readout = nn.Sequential(nn.LayerNorm(language_dim), nn.Linear(language_dim, d))
        self.source_geometry = nn.Linear(14, d)
        self.fuse = nn.Sequential(nn.Linear(4*d, d), nn.GELU(), nn.LayerNorm(d))
        self.blocks = nn.ModuleList(RelationAttention(d, c['heads'], c['dropout'], c['explicit_relations'])
                                    for _ in range(c['decoder_layers']))
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 3))
        # Small nonzero initialization permits backbone gradients on the first step.
        nn.init.normal_(self.head[-1].weight, std=.001)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, h, global_h, source_h, g, relations):
        n = h.shape[1]
        x = self.fuse(torch.cat([self.region_readout(h.float()),
            self.global_readout(global_h.float())[:, None].expand(-1, n, -1),
            source_h, self.source_geometry(g['geometry'])], -1))
        for block in self.blocks:
            x = block(x, g['node_mask'], relations)
        # Natural units, no old target mean/std or hard output clipping.
        return self.head(x)*g['node_mask'][..., None]


class LoRALinear(nn.Module):
    """Standard additive low-rank update; keeps the frozen native projection."""
    def __init__(self, base, rank, alpha):
        super().__init__()
        self.base = base
        self.scale = alpha/rank
        self.a = nn.Linear(base.in_features, rank, bias=False, device=base.weight.device, dtype=torch.float32)
        self.b = nn.Linear(rank, base.out_features, bias=False, device=base.weight.device, dtype=torch.float32)
        nn.init.kaiming_uniform_(self.a.weight, a=math.sqrt(5))
        nn.init.zeros_(self.b.weight)
        for p in base.parameters():
            p.requires_grad_(False)

    def forward(self, x):
        return self.base(x) + (self.b(self.a(x.float()))*self.scale).to(x.dtype)


def attach_language_lora(backbone, rank, alpha):
    names = [name for name, layer in backbone.named_modules()
             if name.startswith('language_model.layers.') and name.endswith(('.q_proj', '.v_proj'))
             and isinstance(layer, nn.Linear)]
    if not names:
        raise ValueError('No native language q/v projections found')
    for name in names:
        parent_name, leaf = name.rsplit('.', 1)
        parent = backbone.get_submodule(parent_name)
        setattr(parent, leaf, LoRALinear(getattr(parent, leaf), rank, alpha))
    return names


class StructuredLayoutModel(nn.Module):
    def __init__(self, c, device='cuda:0'):
        super().__init__()
        if c.get('deterministic_runtime', False):
            from .runtime import configure_determinism
            configure_determinism()
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        self.config = dict(c)
        self.processor = AutoProcessor.from_pretrained(c['model_path'], local_files_only=True,
            min_pixels=c['min_pixels'], max_pixels=c['max_pixels'],
            size={'shortest_edge': c['min_pixels'], 'longest_edge': c['max_pixels']})
        self.processor.tokenizer.add_special_tokens({'additional_special_tokens': list(TOKENS)})
        self.token_ids = {t: self.processor.tokenizer.convert_tokens_to_ids(t) for t in TOKENS}
        container = Qwen3VLForConditionalGeneration.from_pretrained(c['model_path'], local_files_only=True,
                     dtype=torch.bfloat16, attn_implementation='sdpa', device_map={'': device})
        self.backbone = container.model
        del container  # No text generation head is needed for the numeric prototype.
        for p in self.backbone.parameters():
            p.requires_grad_(False)
        self.lora_targets = attach_language_lora(self.backbone, c['lora_rank'], c['lora_alpha'])
        if c['gradient_checkpointing']:
            self.backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        language_dim = self.backbone.config.text_config.hidden_size
        self.source_encoder = SourceEncoder(c).to(device)
        self.projector = nn.Sequential(nn.LayerNorm(c['hidden_dim']), nn.Linear(c['hidden_dim'], language_dim),
                                       nn.GELU(), nn.Linear(language_dim, language_dim)).to(device)
        self.query_region = nn.Parameter(torch.randn(language_dim, device=device)*.02)
        self.query_global = nn.Parameter(torch.randn(language_dim, device=device)*.02)
        self.decoder = JointLayoutDecoder(c, language_dim).to(device)
        self.language_dim = language_dim

    def forward(self, batch):
        from .constants import REGION, FRAME, QUERY, GLOBAL
        g = batch['graph']
        source_h, frame_h, relations = self.source_encoder(g)
        nodes = self.projector(source_h)
        frame = self.projector(frame_h)
        ids = batch['input_ids']
        soft = torch.zeros_like(ids, dtype=torch.bool)
        for tid in self.token_ids.values():
            soft |= ids == tid
        safe_ids = ids.masked_fill(soft, 0)
        embeds = self.backbone.get_input_embeddings()(safe_ids).clone()
        for i, positions in enumerate(batch['positions']):
            n = len(positions[REGION])
            embeds[i, positions[REGION]] = nodes[i, :n].to(embeds.dtype)
            embeds[i, positions[FRAME]] = frame[i].to(embeds.dtype)
            embeds[i, positions[QUERY]] = (nodes[i, :n] + self.query_region).to(embeds.dtype)
            embeds[i, positions[GLOBAL]] = (frame[i] + self.query_global).to(embeds.dtype)
        # Native M-RoPE is computed using the original IDs and processor type IDs.
        pos, _ = self.backbone.get_rope_index(ids, mm_token_type_ids=batch['mm_token_type_ids'],
                    image_grid_thw=batch['image_grid_thw'], attention_mask=batch['attention_mask'])
        native = self.backbone(inputs_embeds=embeds, pixel_values=batch['pixel_values'],
                    image_grid_thw=batch['image_grid_thw'], attention_mask=batch['attention_mask'],
                    position_ids=pos, use_cache=False, return_dict=True)
        hidden = native.last_hidden_state
        nmax = g['node_mask'].shape[1]
        regional, global_states = [], []
        for i, positions in enumerate(batch['positions']):
            h = hidden[i, positions[QUERY]]
            regional.append(F.pad(h, (0, 0, 0, nmax-len(h))))
            global_states.append(hidden[i, positions[GLOBAL][0]])
        h, global_h = torch.stack(regional), torch.stack(global_states)
        return {'delta': self.decoder(h, global_h, source_h, g, relations)}

    def trainable_state(self):
        names = {name for name, p in self.named_parameters() if p.requires_grad}
        return {name: value.detach().cpu() for name, value in self.state_dict().items() if name in names}

    def load_trainable_state(self, state):
        expected = {name for name, p in self.named_parameters() if p.requires_grad}
        if set(state) != expected:
            raise ValueError('Adapter checkpoint has missing/unexpected parameters')
        self.load_state_dict(state, strict=False)

    def parameter_report(self):
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total_parameters': sum(p.numel() for p in self.parameters()), 'trainable_parameters': trainable,
                'lora_targets': self.lora_targets, 'language_dim': self.language_dim,
                'visual_encoder_frozen': not any(p.requires_grad for p in self.backbone.visual.parameters()),
                'old_model_weights_used': False, 'text_or_score_heads_present': False}
