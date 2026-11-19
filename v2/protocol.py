"""Protocol validation and immutable-input checks; standard library only."""
import hashlib
import json
from pathlib import Path

FRAMEWORKS = ('autogluon', 'tpot', 'pycaret', 'flaml', 'h2o')
SCENARIOS = ('baseline', 'random_over', 'smote', 'editednearestneighbors',
             'tomeklinks', 'smoteenn', 'smotetomek', 'class_weighted')
TERMINAL = {'completed', 'unsupported', 'failed', 'budget_exceeded', 'postprocessing_timeout',
            'container_timeout', 'skipped_preprocessing_failed', 'interrupted', 'qualified'}
REVIEW_FILES = ('adapters.py', 'common.py', 'prepare.py', 'run.py', 'orchestrate.py',
                'protocol.py', 'parallel.py', 'weighting.py', 'h2o_metric.py', 'compose.yaml', 'config.json')


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def capability(framework, scenario):
    if framework not in FRAMEWORKS or scenario not in SCENARIOS:
        raise ValueError(f'Unknown framework/scenario: {framework}/{scenario}')
    return True, None


def validate_config(config):
    for key, allowed in [('frameworks', FRAMEWORKS), ('scenarios', SCENARIOS)]:
        values = config[key]
        if not values or len(values) != len(set(values)) or not set(values) <= set(allowed):
            raise ValueError(f'Invalid or repeated {key}')
    from v2.parallel import validate_slots
    validate_slots(config['execution_slots'], config['cpus'])
    seeds = config['seeds']
    if not seeds or len(seeds) != len(set(seeds)) or any(type(s) is not int or not 0 <= s < 2**32-20001 for s in seeds):
        raise ValueError('Seeds must be distinct nonnegative integers in the supported range')
    if config['primary_metric'] != 'f1_macro':
        raise ValueError('The confirmed protocol requires f1_macro')
    if config['cpus'] != 8 or config['memory_gib'] != 16:
        raise ValueError('Resource configuration must match the fixed Compose profile: 8 CPUs, 16 GiB')
    for key in ['search_budget_seconds', 'preprocessing_budget_seconds', 'postprocessing_budget_seconds', 'qualification_native_search_seconds']:
        if type(config[key]) is not int or config[key] <= 0:
            raise ValueError(f'{key} must be a positive integer')
    for key in ['test_group_fraction', 'validation_group_fraction_within_development',
                'h2o_blending_fraction_of_training', 'native_search_budget_fraction']:
        if not 0 < config[key] < 1:
            raise ValueError(f'{key} must be strictly between zero and one')
    return config


def recipe_hash(code_root, config):
    root = Path(code_root)
    keys = ['protocol_version', 'test_group_fraction', 'validation_group_fraction_within_development',
            'h2o_blending_fraction_of_training', 'cpus']
    payload = {'config': {k: config[k] for k in keys},
               'code': {name: digest_file(root / name) for name in ['prepare.py', 'common.py', 'protocol.py']}}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def verify_snapshot(code_root):
    root = Path(code_root)
    path = root / 'SNAPSHOT.json'
    if not path.exists():
        return None
    snapshot = json.loads(path.read_text())
    for name, expected in snapshot['source_sha256'].items():
        if digest_file(root / name) != expected:
            raise ValueError(f'Frozen source changed: {name}')
    return snapshot


def verify_review(code_root):
    root = Path(code_root)
    path = root / 'REVIEW.json'
    if not path.exists():
        raise ValueError('Official runs require a completed review: REVIEW.json is missing')
    report = json.loads(path.read_text())
    if report.get('status') != 'passed':
        raise ValueError('Code review is not complete')
    for name in REVIEW_FILES:
        if report.get('reviewed_sha256', {}).get(name) != digest_file(root / name):
            raise ValueError(f'Code/configuration changed since review: {name}')
    return report
