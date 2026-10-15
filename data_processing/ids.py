"""Stable IDs for source-reference region correspondences."""
import hashlib
import json


def make_match_id(pair_id, source_region_id, target_region_id, method_version):
    canonical = json.dumps([pair_id, source_region_id, target_region_id, method_version],
                           ensure_ascii=False, separators=(',', ':'))
    return 'aesrecon:correspondence:' + hashlib.sha256(canonical.encode()).hexdigest()
