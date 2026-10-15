"""POR batches with source-only prefixes and answer-only causal supervision."""
import numpy as np
import torch
from .source import Collator
from .constants import ATTRIBUTES, ATTRIBUTE_NAMES, ATTRIBUTE_TOKENS, REASON_STATUS, TOKENS, QUERY, GLOBAL

def validate_por_target(source, target):
    keys = {'schema_version', 'source_image_id', 'source_sha256', 'attributes', 'score_scale', 'scores',
            'score_mask', 'reasons', 'reason_mask', 'text_ids', 'reason_status', 'region_grounding_available'}
    if (set(target) != keys or target['schema_version'] != 'structured_por_targets/v1' or
        target['source_image_id'] != source['record']['image_id'] or target['source_sha256'] != source['source_sha256'] or
        target['attributes'] != ATTRIBUTES or target['score_scale'] != [0, 10] or
        target['reason_status'] != REASON_STATUS or target['region_grounding_available'] is not False):
        raise ValueError('POR target schema/source binding mismatch')
    scores, mask = np.asarray(target['scores'], float), np.asarray(target['score_mask'])
    if scores.shape != (7,) or mask.shape != (7,) or mask.dtype != np.bool_ or not mask.any():
        raise ValueError('POR score shapes/masks invalid')
    if not np.isfinite(scores[mask]).all() or (scores[mask] < 0).any() or (scores[mask] > 10).any():
        raise ValueError('POR score range invalid')
    if any(len(target[key]) != 7 for key in ('reasons', 'reason_mask', 'text_ids')):
        raise ValueError('POR reason identity count mismatch')
    if any(type(x) is not bool for x in target['reason_mask']): raise ValueError('Reason masks must be boolean')
    for text, use in zip(target['reasons'], target['reason_mask']):
        if not isinstance(text, str) or use and (not text.strip() or '<|' in text or len(text) > 4000):
            raise ValueError('Invalid active reason')


def attribute_prompt():
    return '\nComposition state readouts:\n'+''.join(f'{name}: {token}\n' for name, token in zip(ATTRIBUTE_NAMES, ATTRIBUTE_TOKENS))


class JointCollator:
    def __init__(self, processor, max_answer_tokens=128):
        self.processor = processor
        self.source_collator = Collator(processor)
        self.tokens = {t: processor.tokenizer.convert_tokens_to_ids(t) for t in TOKENS+ATTRIBUTE_TOKENS}
        self.max_answer_tokens = max_answer_tokens

    def aes(self, sources):
        return dict(self.source_collator(sources), task='aes')

    def por(self, pairs, attributes, with_answers=True):
        if len(pairs) != len(attributes) or any(type(a) is not int or not 0 <= a < 7 for a in attributes):
            raise ValueError('One valid reason attribute per source is required')
        sources, targets = zip(*pairs)
        for s, t in pairs: validate_por_target(s, t)
        base = self.source_collator(sources); tokenizer = self.processor.tokenizer
        rows, types, labels, prefix_lengths, text_truncated = [], [], [], [], []
        attribute_ids = tokenizer.encode(attribute_prompt(), add_special_tokens=False)
        for i, (source, target, attr) in enumerate(zip(sources, targets, attributes)):
            valid = base['attention_mask'][i].bool()
            source_ids = base['input_ids'][i][valid]
            source_types = base['mm_token_type_ids'][i][valid]
            # All score queries precede the selected reason instruction and target answer.
            instruction = f'\nExplain only {ATTRIBUTE_NAMES[attr]} using evidence visible in this photograph. Do not output a numeric score.\nReason: '
            suffix = attribute_ids+tokenizer.encode(instruction, add_special_tokens=False)
            prefix = torch.cat([source_ids, torch.tensor(suffix, dtype=torch.long)])
            answer = []
            truncated = False
            if with_answers and target['reason_mask'][attr]:
                full = tokenizer.encode(target['reasons'][attr], add_special_tokens=False)
                truncated = len(full) > self.max_answer_tokens
                answer = full[:self.max_answer_tokens]
                if not truncated: answer += [tokenizer.eos_token_id]
            ids = torch.cat([prefix, torch.tensor(answer, dtype=torch.long)])
            y = torch.full_like(ids, -100)
            if answer: y[len(prefix):] = torch.tensor(answer)
            rows.append(ids); labels.append(y); prefix_lengths.append(len(prefix)); text_truncated.append(truncated)
            types.append(torch.cat([source_types, torch.zeros(len(ids)-len(source_types), dtype=source_types.dtype)]))
        length = max(len(x) for x in rows)
        input_ids = torch.full((len(rows), length), tokenizer.pad_token_id, dtype=torch.long)
        attention = torch.zeros_like(input_ids); mm = torch.zeros_like(input_ids); y = torch.full_like(input_ids, -100)
        positions, attr_positions = [], []
        for i, (ids, typ, lab) in enumerate(zip(rows, types, labels)):
            n = len(ids); input_ids[i, :n] = ids; attention[i, :n] = 1; mm[i, :n] = typ; y[i, :n] = lab
            p = {t: (ids == self.tokens[t]).nonzero().flatten().tolist() for t in TOKENS}
            a = [(ids == self.tokens[t]).nonzero().flatten().tolist() for t in ATTRIBUTE_TOKENS]
            if any(len(x) != 1 for x in a) or min(x[0] for x in a) <= p[GLOBAL][0]:
                raise ValueError('Attribute queries must follow layout readouts')
            if max(x[0] for x in a) >= prefix_lengths[i]: raise ValueError('Target leaked before attribute query')
            positions.append(p); attr_positions.append([x[0] for x in a])
        return {**base, 'task': 'por', 'input_ids': input_ids, 'attention_mask': attention,
            'mm_token_type_ids': mm, 'positions': positions, 'attribute_positions': torch.tensor(attr_positions),
            'labels': y, 'prefix_lengths': prefix_lengths, 'reason_attributes': attributes, 'text_truncated': text_truncated,
            'scores': torch.tensor([t['scores'] for t in targets]),
            'score_mask': torch.tensor([t['score_mask'] for t in targets], dtype=torch.bool)}
