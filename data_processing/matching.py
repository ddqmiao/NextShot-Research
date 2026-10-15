"""Content correspondence with explicit rejection, without geometric similarity."""
from dataclasses import asdict, dataclass
import numpy as np
from scipy.optimize import linear_sum_assignment
from .semantics import evidence, describe
from .ids import make_match_id


@dataclass(frozen=True)
class MatchConfig:
    region_weight: float = .75
    context_weight: float = .15
    semantic_weight: float = .10
    primary_bonus: float = .035
    candidate_visual: float = .40
    candidate_score: float = .48
    unmatched_score: float = .60
    match_score: float = .72
    object_visual: float = .70
    primary_visual: float = .65
    environment_visual: float = .72
    assignment_margin: float = .045
    candidate_top_k: int = 3


def partial_assignment(scores, eligible, unmatched_score):
    """A real edge pays the same reject opportunity cost regardless of n/m."""
    n,m=scores.shape
    utility=np.zeros((n+m,n+m),dtype=float)
    utility[:n,:m]=np.where(eligible,scores-unmatched_score,-1e6)
    rr,cc=linear_sum_assignment(utility,maximize=True)
    chosen={(int(i),int(j)) for i,j in zip(rr,cc) if i<n and j<m and eligible[i,j] and scores[i,j]>unmatched_score}
    return chosen,float(utility[rr,cc].sum())


def match_regions(pair, source, target, visual, context, method, cfg=MatchConfig()):
    n,m=len(source),len(target)
    visual=np.asarray(visual,float);context=np.asarray(context,float)
    if visual.shape!=(n,m) or context.shape!=(n,m): raise ValueError('Wrong matrix shape')
    if not np.isfinite(visual).all() or not np.isfinite(context).all(): raise ValueError('Nonfinite features')
    sem=[[evidence(a,b) for b in target] for a in source]
    semantic=np.array([[x['score'] for x in row] for row in sem],float).reshape(n,m)
    compatible=np.array([[x['compatible'] for x in row] for row in sem],bool).reshape(n,m)
    grain=np.array([[x['granularity_review'] for x in row] for row in sem],bool).reshape(n,m)
    primary=np.array([[bool(a.get('primary') and b.get('primary')) for b in target] for a in source],bool).reshape(n,m)
    score=cfg.region_weight*np.clip(visual,0,1)+cfg.context_weight*np.clip(context,0,1)+cfg.semantic_weight*semantic
    score=np.minimum(1.,score+cfg.primary_bonus*primary)
    candidate=compatible & ((visual>=cfg.candidate_visual)&(score>=cfg.candidate_score) | primary | (semantic>=.78))
    # Granularity conflicts remain visible as candidates but cannot be confirmed.
    eligible=candidate & ~grain
    chosen,optimum=partial_assignment(score,eligible,cfg.unmatched_score)
    matched=set();reasons={};margins={}
    for i,j in sorted(chosen):
        alternate=eligible.copy();alternate[i,j]=False
        _,alt=partial_assignment(score,alternate,cfg.unmatched_score)
        margin=optimum-alt;margins[f'{i}:{j}']=margin
        env=describe(source[i])['environment'] or describe(target[j])['environment']
        threshold=cfg.environment_visual if env else cfg.primary_visual if primary[i,j] else cfg.object_visual
        required_score=cfg.match_score-(.025 if primary[i,j] else 0)
        if visual[i,j]<threshold:reason='insufficient_region_appearance'
        elif score[i,j]<required_score:reason='insufficient_combined_evidence'
        elif margin<cfg.assignment_margin:reason='competing_assignment'
        else:reason='accepted_content_correspondence';matched.add((i,j))
        reasons[f'{i}:{j}']=reason
    edges=set(matched)
    for i in range(n):
        if not any(a==i for a,_ in matched):
            for j in sorted(np.flatnonzero(candidate[i]),key=lambda j:(-score[i,j],int(j)))[:cfg.candidate_top_k]:edges.add((i,int(j)))
    for j in range(m):
        if not any(b==j for _,b in matched):
            for i in sorted(np.flatnonzero(candidate[:,j]),key=lambda i:(-score[i,j],int(i)))[:cfg.candidate_top_k]:edges.add((int(i),j))
    # A competing retained edge involving an accepted region implies unresolved
    # identity/extent. Keep that component ambiguous instead of hiding the split.
    conflicts={edge for edge in matched if any((i==edge[0] or j==edge[1]) and (i,j)!=edge
                and score[i,j]>=score[edge]-cfg.assignment_margin for i,j in edges)}
    for edge in conflicts:matched.remove(edge);reasons[f'{edge[0]}:{edge[1]}']='possible_granularity_or_identity_conflict'
    records=[]
    def append(i,j,status):
        a=source[i]['region_id'] if i is not None else None
        b=target[j]['region_id'] if j is not None else None
        records.append(dict(match_id=make_match_id(pair['pair_id'],a,b,method),pair_id=pair['pair_id'],
            source_image_id=pair['worse_image_id'],source_region_id=a,target_image_id=pair['better_image_id'],
            target_region_id=b,match_status=status,match_score=float(score[i,j]) if i is not None and j is not None else None,method_version=method))
    for i,j in sorted(edges):
        status='matched' if (i,j) in matched else 'ambiguous'
        append(i,j,status)
        reasons.setdefault(f'{i}:{j}', 'person_granularity_requires_review' if grain[i,j] else 'candidate_not_confirmed')
    for i in range(n):
        if not any(a==i for a,_ in edges):append(i,None,'source_only')
    for j in range(m):
        if not any(b==j for _,b in edges):append(None,j,'target_only')
    return records,dict(config=asdict(cfg),source_region_ids=[r['region_id'] for r in source],
        target_region_ids=[r['region_id'] for r in target],region_similarity=visual.tolist(),
        context_similarity=context.tolist(),semantic_evidence=sem,score=score.tolist(),
        candidate=candidate.tolist(),assignment_margins=margins,edge_reasons=reasons,
        uses_geometry=False,uses_depth=False)
