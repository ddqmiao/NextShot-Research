"""Final frame, body presentation and region geometry are decided inside the relational core."""
import torch
from torch import nn
from torch.nn import functional as F
from .backbone import StructuredLayoutModel, RelationAttention
from .constants import REGION, FRAME, QUERY, GLOBAL
from .constants import SEMANTIC_DIM, FRAMING, PARTS, BOUNDARY, VISIBILITY


def logit(x): return torch.logit(x.clamp(1e-4, 1-1e-4))


def decode_geometry(raw, geometry):
    """Source is initialization only. Outputs are VISIBLE geometry in the NEW target frame."""
    size = geometry[..., 5:7].clamp(1e-4, 1-1e-4)
    center_fraction = ((geometry[..., 3:5]-size/2)/(1-size)).clamp(0, 1)
    target_size = torch.sigmoid(logit(size)+raw[..., 2:4])
    center = target_size/2+(1-target_size)*torch.sigmoid(logit(center_fraction)+raw[..., :2])
    lo = center-target_size/2; hi = center+target_size/2
    src_lo = geometry[..., 3:5]-size/2
    centroid_fraction = ((geometry[..., :2]-src_lo)/size).clamp(0, 1)
    centroid = lo+target_size*torch.sigmoid(logit(centroid_fraction)+raw[..., 4:6])
    occupancy = (geometry[..., 2].exp()/size.prod(-1)).clamp(1e-4, 1-1e-4)
    area = target_size.prod(-1)*torch.sigmoid(logit(occupancy)+raw[..., 6])
    return torch.cat([lo, hi], -1), centroid, area


class FramingDecoder(nn.Module):
    def __init__(self, c, language_dim):
        super().__init__(); d = c['hidden_dim']
        self.region_readout = nn.Sequential(nn.LayerNorm(language_dim), nn.Linear(language_dim, d))
        self.global_readout = nn.Sequential(nn.LayerNorm(language_dim), nn.Linear(language_dim, d))
        self.semantic = nn.Linear(SEMANTIC_DIM, d)
        self.body_evidence = nn.Linear(68, d)
        self.geometry = nn.Linear(14, d)
        self.fuse = nn.Sequential(nn.Linear(6*d, d), nn.GELU(), nn.LayerNorm(d))
        self.blocks = nn.ModuleList(RelationAttention(d, c['heads'], 0., True) for _ in range(2))
        self.focus = nn.Linear(d, 1)
        self.global_fuse = nn.Sequential(nn.Linear(2*d, d), nn.GELU(), nn.LayerNorm(d))
        self.frame_head = nn.Linear(d, 1)
        self.body_head = nn.Linear(d, len(FRAMING)+len(PARTS)*(len(BOUNDARY)+len(VISIBILITY)))
        self.body_condition = nn.Linear(self.body_head.out_features, d)
        self.frame_condition = nn.Linear(1, d)
        self.final_block = RelationAttention(d, c['heads'], 0., True)
        self.geometry_head = nn.Linear(d, 7)
        for layer in [self.frame_head, self.geometry_head]:
            nn.init.normal_(layer.weight, std=.001); nn.init.zeros_(layer.bias)

    def forward(self, regional, global_h, source_h, g, relations):
        n = regional.shape[1]; gh = self.global_readout(global_h.float())
        x = self.fuse(torch.cat([self.region_readout(regional.float()), gh[:, None].expand(-1, n, -1),
            source_h, self.geometry(g['geometry']), self.semantic(g['semantic']), self.body_evidence(g['body'])], -1))
        for block in self.blocks: x = block(x, g['node_mask'], relations)
        # Trainable composition-core emphasis, not a fixed Qwen subject assignment.
        focus_logits = self.focus(x).squeeze(-1)+4*g['focus_hint']
        emphasis = focus_logits.masked_fill(~g['node_mask'], -torch.inf).softmax(-1)
        whole = self.global_fuse(torch.cat([gh, (x*emphasis[..., None]).sum(1)], -1))
        log_aspect = g['geometry'][:, 0, 12]+self.frame_head(whole).squeeze(-1)
        body = self.body_head(x+whole[:, None])
        f = len(FRAMING); b = len(PARTS)*len(BOUNDARY)
        framing_logits = body[..., :f]
        boundary_logits = body[..., f:f+b].reshape(*body.shape[:2], len(PARTS), len(BOUNDARY))
        visibility_logits = body[..., f+b:].reshape(*body.shape[:2], len(PARTS), len(VISIBILITY))
        body_probs = torch.cat([framing_logits.softmax(-1), boundary_logits.softmax(-1).flatten(-2), visibility_logits.softmax(-1).flatten(-2)], -1)
        # Frame and body goals enter the final relational pass, not merely output explanations.
        x = x+whole[:, None]+self.frame_condition(log_aspect[:, None])[:, None]
        x = x+self.body_condition(body_probs)*g['person'][..., None]
        x = self.final_block(x, g['node_mask'], relations)
        boxes, centers, areas = decode_geometry(self.geometry_head(x), g['geometry'])
        return {'log_aspect': log_aspect, 'boxes': boxes, 'centroids': centers, 'areas': areas,
                'framing_logits': framing_logits, 'boundary_logits': boundary_logits, 'visibility_logits': visibility_logits,
                'emphasis': emphasis, 'emphasis_semantics': 'internal_learned_allocation_not_calibrated_subject_truth'}


class FramingModel(StructuredLayoutModel):

    def forward(self, batch):
        g = batch['graph']; source_h, frame_h, relations = self.source_encoder(g)
        nodes, frame = self.projector(source_h), self.projector(frame_h)
        ids = batch['input_ids']; soft = torch.zeros_like(ids, dtype=torch.bool)
        for tid in self.token_ids.values(): soft |= ids == tid
        embeds = self.backbone.get_input_embeddings()(ids.masked_fill(soft, 0)).clone()
        for i, p in enumerate(batch['positions']):
            n = len(p[REGION]); embeds[i, p[REGION]] = nodes[i, :n].to(embeds.dtype)
            embeds[i, p[FRAME]] = frame[i].to(embeds.dtype)
            embeds[i, p[QUERY]] = (nodes[i, :n]+self.query_region).to(embeds.dtype)
            embeds[i, p[GLOBAL]] = (frame[i]+self.query_global).to(embeds.dtype)
        pos, _ = self.backbone.get_rope_index(ids, mm_token_type_ids=batch['mm_token_type_ids'], image_grid_thw=batch['image_grid_thw'], attention_mask=batch['attention_mask'])
        hidden = self.backbone(inputs_embeds=embeds, pixel_values=batch['pixel_values'], image_grid_thw=batch['image_grid_thw'],
            attention_mask=batch['attention_mask'], position_ids=pos, use_cache=False, return_dict=True).last_hidden_state
        nmax = g['node_mask'].shape[1]
        h = torch.stack([F.pad(hidden[i, p[QUERY]], (0, 0, 0, nmax-len(p[QUERY]))) for i, p in enumerate(batch['positions'])])
        gh = torch.stack([hidden[i, p[GLOBAL][0]] for i, p in enumerate(batch['positions'])])
        return self.decoder(h, gh, source_h, g, relations)
