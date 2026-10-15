"""Portable model validation, independent of private paths and dataset manifests."""
from pathlib import Path


def aes_config(config):
    result = dict(config)
    result['por_enabled'] = False
    return result


def validate_aes(config):
    expected = dict(feature_dim=2048, geometry_dim=14, orientation_dim=12,
                    body_dim=533, decoder_layers=2, dropout=0.0,
                    explicit_relations=True, body_guidance_enabled=True,
                    body_guidance_granularity='full_half_only')
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f'The released main architecture requires {key}={value!r}')
    if not isinstance(config.get('model_path'), (str, Path)) or not str(config['model_path']):
        raise ValueError('Set model_path to a local Qwen3-VL-4B-Instruct directory')
    if config['hidden_dim'] % config['heads'] or config['lora_rank'] < 1:
        raise ValueError('Invalid attention dimension or LoRA rank')
    if config['min_pixels'] <= 0 or config['max_pixels'] < config['min_pixels']:
        raise ValueError('Invalid image resolution limits')
    if config.get('offline_semantic_observations', False):
        raise ValueError('The main architecture does not use an offline semantic observer')


def validate(config):
    validate_aes(config)
    if config.get('schema_version') != 'nextshot/v1' or not config.get('por_enabled'):
        raise ValueError('Use the NextShot main configuration with POR enabled')
    if config['reason_chunk_tokens'] < 1:
        raise ValueError('reason_chunk_tokens must be positive')
