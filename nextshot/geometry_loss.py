"""Whole-reference geometry supervision plus explicitly WEAK observed-body supervision."""
import torch
from torch.nn import functional as F
from .constants import FRAMING, BOUNDARY, VISIBILITY, PARTS


def balanced(values, valid, person):
    active = [values[valid & role].mean() for role in (person, ~person) if (valid & role).any()]
    return torch.stack(active).mean() if active else values.sum()*0


def loss(output, targets, graph):
    losses, details = [], []
    for i, target in enumerate(targets):
        refs = target['references']; n = len(target['region_ids']); device = output['boxes'].device
        masks = torch.tensor([r['valid'] for r in refs], device=device, dtype=torch.bool)
        common = masks.all(0); person = graph['person'][i, :n]
        scores, geometries, frames = [], [], []
        for j, ref in enumerate(refs):
            valid = masks[j]
            # Replace unknown values BEFORE arithmetic, including external NaN placeholders.
            box = torch.tensor(ref['boxes'], device=device, dtype=torch.float32)
            center = torch.tensor(ref['centroids'], device=device, dtype=torch.float32)
            area = torch.tensor(ref['areas'], device=device, dtype=torch.float32)
            box = torch.where(valid[:, None], box, torch.zeros_like(box))
            center = torch.where(valid[:, None], center, torch.zeros_like(center))
            area = torch.where(valid, area, torch.ones_like(area))
            predicted_box = torch.where(valid[:, None], output['boxes'][i, :n], torch.zeros_like(box))
            predicted_center = torch.where(valid[:, None], output['centroids'][i, :n], torch.zeros_like(center))
            predicted_area = torch.where(valid, output['areas'][i, :n], torch.ones_like(area))
            err = F.smooth_l1_loss(predicted_box/.1, box/.1, reduction='none').mean(-1)
            err = err+F.smooth_l1_loss(predicted_center/.1, center/.1, reduction='none').mean(-1)
            err = err+F.smooth_l1_loss(predicted_area.clamp_min(1e-8).log()/.7, area.clamp_min(1e-8).log()/.7, reduction='none')
            fl = F.smooth_l1_loss(output['log_aspect'][i]/.3, output['log_aspect'].new_tensor(ref['log_aspect'])/.3)
            scores.append(fl+balanced(err, common, person)); geometries.append(err); frames.append(fl)
        j = int(torch.stack(scores).detach().argmin()); ref = refs[j]; valid = masks[j]
        geometric = balanced(geometries[j], valid, person)
        weak = []; coverage = {'framing': 0, 'boundary': 0, 'visibility': 0}
        for k, labels in enumerate(ref['body']):
            if labels is None or not bool(person[k]) or not bool(valid[k]): continue
            for key, classes in [('framing', FRAMING), ('boundary', BOUNDARY), ('visibility', VISIBILITY)]:
                logits = output[key+'_logits'][i, k]
                y = torch.tensor(labels[key], device=device, dtype=torch.long)
                active = y != classes.index('unknown')
                if active.any():
                    ll = F.cross_entropy(logits.reshape(-1, len(classes))[active.reshape(-1)], y.reshape(-1)[active.reshape(-1)])
                    weak.append(ll); coverage[key] += int(active.sum())
        body_loss = torch.stack(weak).mean() if weak else output['framing_logits'][i].sum()*0
        total = geometric+frames[j]+.1*body_loss
        losses.append(total); details.append({'reference_id': ref['reference_id'], 'geometry': float(geometric.detach()),
            'frame': float(frames[j].detach()), 'weak_body': float(body_loss.detach()), 'coverage': coverage,
            'valid_regions': int(valid.sum()), 'selection_common_regions': int(common.sum())})
    return torch.stack(losses).mean(), details
