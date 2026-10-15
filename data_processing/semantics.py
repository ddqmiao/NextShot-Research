"""Transparent semantic candidate evidence, not instance identity labels."""
import re

PATTERNS = {
    'tree': r'\b(tree\w*|woodland|forest\w*|grove|branches|canopy)\b',
    'grass': r'\b(grass\w*|lawn|meadow|turf)\b',
    'flower': r'\b(flower\w*|blossom\w*|bloom\w*|flowerbed\w*)\b',
    'hedge': r'\b(hedge\w*|bush\w*|shrub\w*)\b',
    'mountain': r'\b(mountain\w*|hill\w*|ridge\w*)\b',
    'ground': r'\b(ground|floor\w*|pavement|paved|walkway|path\w*|road\w*|tile\w*|gravel)\b',
    'rock': r'\b(rock\w*|boulder\w*|cliff\w*)\b',
    'building': r'\b(building\w*|temple\w*|pagoda\w*|tower\w*|facade|façade|house\w*|palace|pavilion)\b',
    'wall': r'\b(wall\w*|glass facade|window\w*)\b',
    'pillar': r'\b(pillar\w*|column\w*)\b',
    'railing': r'\b(railing\w*|fence\w*|balustrade\w*)\b',
    'steps': r'\b(stair\w*|steps|staircase)\b',
    'bench': r'\b(bench\w*|chair\w*|seat\w*)\b',
    'table': r'\b(table\w*|desk\w*)\b',
    'bag': r'\b(bag\w*|basket\w*|purse\w*)\b',
    'hat': r'\b(hat\w*|cap)\b',
    'poster': r'\b(poster\w*|sign\w*|billboard\w*|painting\w*)\b',
    'food': r'\b(food|fruit\w*|picnic|snack\w*|drink\w*|bottle\w*|cup\w*)\b',
    'vehicle': r'\b(car\w*|vehicle\w*|bicycle\w*|bike\w*|motorcycle\w*)\b',
    'lamp': r'\b(lamp\w*|lantern\w*)\b',
    'planter': r'\b(planter\w*|flowerpot\w*)\b',
    'water': r'\b(water|lake\w*|sea|ocean|river\w*|pond\w*|pool\w*)\b',
    'sky': r'\b(sky|cloud\w*)\b',
}
GROUP_TAGS = {
    'vegetation': {'tree','grass','flower','hedge'},
    'landscape': {'mountain','ground','rock','water'},
    'sky': {'sky'}, 'water': {'water'},
    'architecture': {'building','wall','pillar','railing','steps','ground','poster','planter'},
    'interior': {'wall','ground','building','bench','table','poster','lamp'},
    'space': {'ground','water','sky','wall'},
    'object': set(PATTERNS)-{'sky','water','mountain','grass','tree','flower','hedge'},
}
STOP = set('a an the of in on at with and or to from near beside behind front foreground background left right center middle distant close large small tall short standing sitting lying posing woman man person subject above below under along around by'.split())


def describe(region):
    text = (region.get('semantic') or '').lower()
    group = region.get('semantic_group') or 'other'
    tags = {k for k in GROUP_TAGS.get(group, PATTERNS) if re.search(PATTERNS[k], text)}
    grain = 'region'
    if group == 'person':
        tags = {'person'}
        grain = 'individual'
        if re.search(r'\b(people|crowd|tourists|visitors|group of|couple|two (women|men|persons))\b', text):
            grain = 'crowd'
        elif re.search(r"^(?:a |the )?(?:person.s )?(?:arm|hand|leg|shoulder|body part|partial body)", text):
            grain = 'body_part'
    environment = group in {'sky','water','space','landscape'} or (
        group == 'vegetation' and not re.search(r'\b(a tree|single tree|one tree|tree with|tree trunk)\b', text))
    tokens = set(re.findall(r'[a-z]+', text)) - STOP
    tokens = {x[:-1] if x.endswith('s') and len(x)>4 else x for x in tokens}
    return {'group':group,'tags':sorted(tags),'grain':grain,'environment':environment,'tokens':sorted(tokens)}


def evidence(a, b):
    x, y = describe(a), describe(b)
    gx, gy = x['group'], y['group']
    tx, ty = set(x['tags']), set(y['tags'])
    same = gx == gy
    compatible = True
    reason = 'compatible'
    if (gx == 'person') != (gy == 'person'):
        compatible, reason = False, 'person_nonperson'
    elif (gx == 'sky') != (gy == 'sky') and 'other' not in (gx,gy):
        compatible, reason = False, 'sky_non_sky'
    elif tx and ty and not tx & ty:
        compatible, reason = False, 'distinct_content_types'
    elif not same and 'other' not in (gx,gy) and not tx & ty:
        compatible = {gx,gy} <= {'architecture','interior','landscape','space','object'}
        reason = 'soft_group_boundary' if compatible else 'incompatible_groups'
    jaccard = len(set(x['tokens']) & set(y['tokens'])) / max(1,len(set(x['tokens']) | set(y['tokens'])))
    tag_similarity = len(tx & ty)/max(1,len(tx|ty)) if tx and ty else .3
    score = min(1., .35*same + .45*tag_similarity + .2*jaccard)
    if gx == gy == 'person': score = .75 + .25*jaccard
    granularity = gx == gy == 'person' and (x['grain'] != 'individual' or y['grain'] != 'individual')
    if granularity: reason = 'person_granularity_requires_review'
    return {'compatible':bool(compatible),'score':float(score),'reason':reason,
            'granularity_review':granularity,'source':x,'target':y}
