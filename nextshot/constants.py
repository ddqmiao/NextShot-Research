"""Explicit distinction between source observations, optional intent, and target state."""

POTENTIAL = ('low', 'medium', 'high', 'unknown')
ROLES = ('landmark', 'framing', 'context', 'texture', 'possible_distraction', 'unknown')
FRAMING = ('full_body', 'knee_up', 'waist_up', 'chest_up', 'head_closeup', 'unknown')
PARTS = ('head', 'left_hand', 'right_hand', 'left_foot', 'right_foot')
BOUNDARY = ('in_frame', 'cut_by_frame', 'out_of_frame', 'unknown')
VISIBILITY = ('observed', 'occluded', 'unknown')
SEMANTIC_DIM = len(POTENTIAL)+len(ROLES)+len(FRAMING)+len(PARTS)*(len(BOUNDARY)+len(VISIBILITY))



REGION = '<|layout_source_region|>'
FRAME = '<|layout_source_frame|>'
QUERY = '<|layout_region_query|>'
GLOBAL = '<|layout_global_query|>'
TOKENS = (REGION, FRAME, QUERY, GLOBAL)

ATTRIBUTES = [
    'application_of_classical_composition_principles',
    'depth_of_field_and_layering', 'visual_center_stability',
    'visual_flow_guidance', 'structural_support_stability',
    'appropriateness_of_negative_space', 'subject_integrity',
]
ATTRIBUTE_NAMES = [x.replace('_', ' ').title() for x in ATTRIBUTES]
ATTRIBUTE_TOKENS = tuple(f'<|layout_attribute_{i}|>' for i in range(7))
REASON_STATUS = 'original_dataset_assessment_not_region_grounded_or_independently_verified'
