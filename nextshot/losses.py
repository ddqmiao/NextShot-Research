"""Whole-reference layout and coarse human framing supervision."""
import torch
from torch.nn import functional as F
from .geometry_loss import loss as layout_loss

def loss(output,targets,graph):
    base,details=layout_loss(output,targets,graph);extras=[]
    for i,(target,detail) in enumerate(zip(targets,details)):
        ref=next(r for r in target['references'] if r['reference_id']==detail['reference_id'])
        indices=[k for k,(label,valid) in enumerate(zip(ref['coarse_body'],ref['valid']))
                 if label is not None and valid and bool(graph['person'][i,k])]
        logits=output['coarse_framing_logits'][i]
        if indices:
            y=torch.tensor([ref['coarse_body'][k] for k in indices],device=logits.device,dtype=torch.long)
            value=F.cross_entropy(logits[indices],y)
        else:value=logits.sum()*0
        extras.append(value);detail.update(coarse_body_loss=float(value.detach()),coarse_supervised_regions=len(indices),
                                          coarse_supervised_labels=[ref['coarse_body'][k] for k in indices])
    return base+.1*torch.stack(extras).mean(),details
