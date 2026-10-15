"""Current frame/body/layout core, with training-only POR supervision."""
import hashlib
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .aes_core import Model as AESModel
from .constants import TOKENS, REGION, FRAME, QUERY, GLOBAL
from .constants import ATTRIBUTE_TOKENS
from .config import validate, aes_config


def hash_state(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        h.update(name.encode())
        h.update(value.contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def reason_losses(hidden, labels, vocabulary_weight, chunk_size):
    values, counts = [], []
    for i in range(len(hidden)):
        pos = (labels[i] != -100).nonzero().flatten()
        count = len(pos)
        counts.append(count)
        if not count:
            values.append(hidden[i].sum() * 0)
            continue
        if int(pos.min()) < 1:
            raise ValueError('Answer must follow source-only prefix')
        pieces = []
        for start in range(0, count, chunk_size):
            selected = pos[start:start + chunk_size]
            states, target = hidden[i, selected - 1], labels[i, selected]
            def compute(x, y):
                logits = F.linear(x.to(vocabulary_weight.dtype), vocabulary_weight)
                return F.cross_entropy(logits.float(), y, reduction='sum')
            value = checkpoint(compute, states, target, use_reentrant=False) if torch.is_grad_enabled() else compute(states, target)
            pieces.append(value)
        values.append(torch.stack(pieces).sum() / count)
    return torch.stack(values), counts


class Model(AESModel):
    def __init__(self, c, device='cuda:0', with_auxiliary=True):
        validate(c)
        super().__init__(aes_config(c), device)
        self.config = dict(c)
        self.initialized_layout_sha256 = hash_state(self.trainable_state())
        self.with_auxiliary = with_auxiliary
        if not with_auxiliary:
            return
        # Construct auxiliary parameters AFTER the unchanged AES core. Same seed
        # gives exactly the same shared/core initialization as the AES-only run.
        self.processor.tokenizer.add_special_tokens({'extra_special_tokens': list(ATTRIBUTE_TOKENS)}, replace_extra_special_tokens=False)
        self.attribute_ids = {t: self.processor.tokenizer.convert_tokens_to_ids(t) for t in ATTRIBUTE_TOKENS}
        self.attribute_queries = nn.Parameter(torch.randn(7, self.language_dim, device=device) * .02)
        self.attribute_head = nn.Sequential(nn.LayerNorm(self.language_dim), nn.Linear(self.language_dim, 1)).to(device)
        nn.init.normal_(self.attribute_head[-1].weight, std=.001)
        nn.init.constant_(self.attribute_head[-1].bias, .5)
        # Native 4B ties its vocabulary head to the input embedding. Keep ONE
        # frozen tensor, not a second model/head copy and not an 8B checkpoint.
        if not self.backbone.config.text_config.tie_word_embeddings:
            raise ValueError('Expected native tied 4B vocabulary head')
        if self.backbone.get_input_embeddings().weight.requires_grad:
            raise ValueError('Vocabulary embedding must stay frozen')

    def forward(self, batch):
        if batch.get('task', 'aes') == 'aes':
            return super().forward(batch)
        if not self.with_auxiliary or batch['task'] != 'por':
            raise ValueError('Unknown/disabled auxiliary branch')
        g = batch['graph']
        source_h, frame_h, _ = self.source_encoder(g)
        nodes, frame = self.projector(source_h), self.projector(frame_h)
        ids = batch['input_ids']
        soft = torch.zeros_like(ids, dtype=torch.bool)
        for token in (*self.token_ids.values(), *self.attribute_ids.values()):
            soft |= ids == token
        embeds = self.backbone.get_input_embeddings()(ids.masked_fill(soft, 0)).clone()
        for i, p in enumerate(batch['positions']):
            n = len(p[REGION])
            embeds[i, p[REGION]] = nodes[i, :n].to(embeds.dtype)
            embeds[i, p[FRAME]] = frame[i].to(embeds.dtype)
            embeds[i, p[QUERY]] = (nodes[i, :n] + self.query_region).to(embeds.dtype)
            embeds[i, p[GLOBAL]] = (frame[i] + self.query_global).to(embeds.dtype)
            embeds[i, batch['attribute_positions'][i]] = (frame[i] + self.attribute_queries).to(embeds.dtype)
        pos, _ = self.backbone.get_rope_index(ids, mm_token_type_ids=batch['mm_token_type_ids'],
            image_grid_thw=batch['image_grid_thw'], attention_mask=batch['attention_mask'])
        hidden = self.backbone(inputs_embeds=embeds, pixel_values=batch['pixel_values'], image_grid_thw=batch['image_grid_thw'],
            attention_mask=batch['attention_mask'], position_ids=pos, use_cache=False, return_dict=True).last_hidden_state
        selected = torch.stack([hidden[i, batch['attribute_positions'][i]] for i in range(len(hidden))])
        predicted = self.attribute_head(selected.float()).squeeze(-1)
        mask = batch['score_mask']
        if not mask.any(-1).all():
            raise ValueError('No valid POR attribute for a source')
        targets = torch.where(mask, batch['scores'] / 10., torch.zeros_like(batch['scores']))
        element = F.smooth_l1_loss(torch.where(mask, predicted, torch.zeros_like(predicted)), targets, beta=.1, reduction='none')
        attributes = (element * mask).sum(-1) / mask.sum(-1)
        reasons, counts = reason_losses(hidden, batch['labels'], self.backbone.get_input_embeddings().weight,
                                        self.config['reason_chunk_tokens'])
        return dict(scores=predicted * 10., attribute_losses=attributes, reason_losses=reasons, reason_tokens=counts)

    def layout_state(self):
        return {k: v for k, v in self.trainable_state().items() if not k.startswith(('attribute_queries', 'attribute_head.'))}

    def load_layout_state(self, state):
        expected = set(self.layout_state())
        if set(state) != expected:
            raise ValueError('Layout adapter contains missing or auxiliary parameters')
        self.load_state_dict(state, strict=False)

    def parameter_report(self):
        return dict(super().parameter_report(), auxiliary_heads_training_only=self.with_auxiliary,
            reason_vocabulary='native_frozen_tied_input_embedding', initialized_layout_sha256=self.initialized_layout_sha256,
            por_has_layout_target=False, text_or_score_heads_present=self.with_auxiliary)
