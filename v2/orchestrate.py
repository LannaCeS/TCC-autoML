"""Resource-isolated Docker runner. Only the Python standard library is needed on host."""
import argparse
import ast
import csv
import fcntl
from datetime import datetime, timezone
import hashlib
import json
import os
import shutil
import signal
from pathlib import Path
import subprocess
import sys
import time
import threading
import uuid
from v2.protocol import validate_config, capability, TERMINAL, digest_file, verify_snapshot, verify_review

from v2.parallel import schedule, host_slots

ROOT = Path(__file__).resolve().parent
STOP = threading.Event()


def atomic_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def slot_context(context, slot):
    index, cpuset = slot
    return dict(context, environment=dict(context['environment'], BENCHMARK_CPUSET=cpuset, BENCHMARK_SLOT=str(index)))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def docker(service, arguments, log, timeout=None, context=None):
    environment = dict(os.environ, LOCAL_UID=str(os.getuid()), LOCAL_GID=str(os.getgid()))
    compose_args = []
    if context:
        verify_snapshot(context['code'])
        environment.update(context['environment'])
        compose_args = ['-f', str(context['code'] / 'compose.yaml'), '--project-directory', str(ROOT)]
    name = 'lanna-v2-' + uuid.uuid4().hex[:12]
    if STOP.is_set():
        return 130
    with log.open('w') as stream:
        command = ['docker', 'compose', *compose_args, 'run', '--pull', 'never', '--rm', '--no-deps', '--name', name, service, *arguments]
        process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
        started = time.monotonic()
        try:
            while True:
                if STOP.is_set() or (timeout is not None and time.monotonic() - started > timeout):
                    # Retry if cancellation coincides with container creation.
                    deadline = time.monotonic() + 15
                    while process.poll() is None and time.monotonic() < deadline:
                        subprocess.run(['docker', 'stop', '-t', '2', name], stdout=subprocess.DEVNULL,
                                       stderr=subprocess.DEVNULL, timeout=5)
                        try:
                            process.wait(timeout=.5)
                        except subprocess.TimeoutExpired:
                            pass
                    if process.poll() is None:
                        process.terminate()
                        process.wait(timeout=5)
                    subprocess.run(['docker', 'rm', '-f', name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
                    if STOP.is_set():
                        return 130
                    raise subprocess.TimeoutExpired(command, timeout)
                try:
                    return process.wait(timeout=.5)
                except subprocess.TimeoutExpired:
                    pass
        except BaseException:
            subprocess.run(['docker', 'rm', '-f', name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=10)
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=5)
            raise


def freeze_inputs(folder, frameworks, config, require_review=False):
    code, inputs = folder / 'source', folder / 'input'
    code.mkdir()
    inputs.mkdir()
    for path in ROOT.rglob('*'):
        relative = path.relative_to(ROOT)
        if 'outputs' in relative.parts or '__pycache__' in relative.parts:
            continue
        if path.is_file() and path.suffix in {'.py', '.json', '.yaml', '.md'}:
            target = code / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    if json.loads((code / 'config.json').read_text()) != config:
        raise ValueError('Configuration changed while the source snapshot was being made')
    review = verify_review(code) if require_review else None
    dataset = ROOT.parent / 'data' / 'creditcard.csv'
    expected = digest_file(dataset)
    shutil.copy2(dataset, inputs / 'creditcard.csv')
    if digest_file(inputs / 'creditcard.csv') != expected:
        raise ValueError('Dataset changed while it was being copied')
    images = {name: subprocess.check_output(['docker', 'image', 'inspect', f'lanna-automl-{name}:local',
                                            '--format', '{{.Id}}'], text=True).strip()
              for name in set(frameworks) | {'pycaret'}}
    if review and (any(review.get('reviewed_image_ids', {}).get(name) != value for name, value in images.items()) or review.get('dataset_sha256') != expected):
        raise ValueError('Docker images or dataset differ from the reviewed environment; revalidate before official runs')
    sources = {str(p.relative_to(code)): digest_file(p) for p in code.rglob('*') if p.is_file()}
    (code / 'SNAPSHOT.json').write_text(json.dumps({'source_sha256': sources, 'dataset_sha256': expected,
                                                  'image_ids': images}, indent=2) + '\n')
    environment = {'BENCHMARK_CODE_DIR': str(code), 'BENCHMARK_DATA_DIR': str(inputs),
                   'BENCHMARK_OUTPUT_DIR': str(ROOT / 'outputs')}
    environment.update({f'BENCHMARK_IMAGE_{name.upper()}': value for name, value in images.items()})
    return {'code': code, 'environment': environment, 'source_sha256': sources,
            'dataset_sha256': expected, 'image_ids': images}


def import_completed_run(previous, folder, plan, config, context, rerun_frameworks=()):
    """Continue a stopped official run without selecting attempts by test score."""
    previous = Path(previous).resolve()
    if not previous.is_relative_to(ROOT / 'outputs') or previous == folder:
        raise ValueError('Continuation must refer to a previous output directory')
    old_protocol = json.loads((previous / 'protocol.json').read_text())
    if old_protocol['mode'] != 'official' or not (previous / 'finished.json').exists():
        raise ValueError('Continuation requires a stopped or finished official run')
    rerun_frameworks = set(rerun_frameworks)
    ignored = {'implementation_revision'} | ({'tpot_execution'} if 'tpot' in rerun_frameworks else set()) | ({'pycaret_execution'} if 'pycaret' in rerun_frameworks else set())
    def scientific_settings(values):
        return {key: value for key, value in values.items() if key not in ignored}
    if scientific_settings(old_protocol['config']) != scientific_settings(config):
        raise ValueError('Scientific configuration changed; previous results cannot be reused')
    if old_protocol['dataset_sha256'] != context['dataset_sha256'] or any(
            old_protocol['image_ids'].get(key) != value for key, value in context['image_ids'].items()):
        raise ValueError('Dataset or Docker images changed since the previous run')
    snapshot = verify_snapshot(previous / 'source')
    for name in ['common.py', 'prepare.py', 'run.py', 'protocol.py', 'weighting.py', 'h2o_metric.py', 'compose.yaml']:
        if digest_file(ROOT / name) != snapshot['source_sha256'][name]:
            raise ValueError(f'Continuation would mix incompatible code: {name}')
    def model_branches(source):
        search = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == 'search')
        return {ast.unparse(node.test): ast.dump(node, include_attributes=False) for node in search.body if isinstance(node, ast.If)}
    before = model_branches((previous / 'source/adapters.py').read_text())
    after = model_branches((ROOT / 'adapters.py').read_text())
    for tool in ['autogluon', 'tpot', 'pycaret', 'flaml']:
        key = f"framework == '{tool}'"
        if tool not in rerun_frameworks and before[key] != after[key]:
            raise ValueError(f'Cannot reuse results after changing the {tool} adapter')
    def h2o_training_prefix(source):
        search = next(node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == 'search')
        branch = next(node for node in search.body if isinstance(node, ast.If) and ast.unparse(node.test) == "framework == 'h2o'")
        body = next(node for node in branch.body if isinstance(node, ast.Try)).body
        prefix = []
        for node in body:
            prefix.append(ast.dump(node, include_attributes=False))
            if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and ast.unparse(node.value.func) == 'automl.train':
                return prefix
        raise ValueError('H2O training call not found')
    if 'h2o' not in rerun_frameworks and h2o_training_prefix((previous / 'source/adapters.py').read_text()) != h2o_training_prefix((ROOT / 'adapters.py').read_text()):
        raise ValueError('H2O training configuration changed; previous results cannot be reused')
    reused, carried_failures, prepared, attempts = {}, {}, [], []
    for entry in plan:
        origin = previous / 'runs' / entry['name']
        status_path = origin / 'status.json'
        if not status_path.exists():
            continue
        state = json.loads(status_path.read_text())
        if any(state.get(key) != entry[key] for key in ['framework', 'scenario', 'seed', 'mode']):
            raise ValueError('Previous experiment metadata does not match the execution plan')
        if entry['framework'] in rerun_frameworks:
            attempts.append(dict(name=entry['name'], previous_status=state['status'], previous_directory=str(origin),
                                 reason='All cases of the changed framework are repeated under the new execution policy'))
            continue
        failed_terminal = state['status'] in {'failed', 'budget_exceeded', 'postprocessing_timeout', 'container_timeout', 'skipped_preprocessing_failed'}
        if rerun_frameworks and failed_terminal:
            target = folder / 'runs' / entry['name']
            target.symlink_to(os.path.relpath(origin, target.parent), target_is_directory=True)
            carried_failures[entry['name']] = dict(directory=str(origin), status=state['status'], status_sha256=digest_file(status_path))
            continue
        if state['status'] != 'completed':
            attempts.append(dict(name=entry['name'], previous_status=state['status'], previous_directory=str(origin)))
            continue
        if state.get('container_exit_code') != 0:
            raise ValueError('Completed experiment has an unsuccessful container exit')
        required = ['metrics.json', 'test-predictions.csv', 'selection.json', 'selection-frozen.json', 'selected-model.joblib', 'preprocessing.json', 'resources.json']
        if entry['framework'] in {'pycaret', 'tpot'}:
            required.append('training-weight-audit.json')
        signatures = {name: digest_file(origin / name) for name in required}
        metrics = json.loads((origin / 'metrics.json').read_text())
        if metrics.get('validation_only') or not 0 <= metrics['test']['f1_macro'] <= 1:
            raise ValueError('Previous experiment lacks valid final test metrics')
        target = folder / 'runs' / entry['name']
        target.symlink_to(os.path.relpath(origin, target.parent), target_is_directory=True)
        reused[entry['name']] = dict(directory=str(origin), artifact_sha256=signatures)
    # Prepared arrays remain compatible because the preparation recipe is unchanged.
    from v2.protocol import recipe_hash
    for name in sorted({f"{entry['scenario']}-{entry['seed']}" for entry in plan}):
        origin = previous / 'prepared' / name
        if not (origin / 'manifest.json').exists():
            continue
        manifest = json.loads((origin / 'manifest.json').read_text())
        scenario, seed = name.rsplit('-', 1)
        if manifest['scenario'] != scenario or manifest['seed'] != int(seed):
            raise ValueError('Previous preparation metadata does not match its scenario and seed')
        if manifest['recipe_sha256'] != recipe_hash(ROOT, config) or manifest['dataset_sha256'] != context['dataset_sha256'] or manifest['smoke']:
            raise ValueError('Previous preparation recipe or dataset does not match')
        for filename, expected in [('partitions.npz', manifest['partitions_sha256']),
                                   ('data.npz', manifest['data_npz_sha256']),
                                   ('data_h2o.npz', manifest['h2o_blending']['data_npz_sha256'])]:
            if digest_file(origin / filename) != expected:
                raise ValueError(f'Previous prepared artifact is corrupt: {filename}')
        target = folder / 'prepared' / name
        target.symlink_to(os.path.relpath(origin, target.parent), target_is_directory=True)
        prepared.append(name)
    report = dict(previous_run=str(previous), reused_completed=reused, reused_terminal_failures=carried_failures,
                  rerun_frameworks=sorted(rerun_frameworks), reused_preparations=prepared,
                  previous_failed_or_interrupted_attempts=[a for a in attempts if a['previous_status'] != 'completed'],
                  previous_completed_superseded=[a for a in attempts if a['previous_status'] == 'completed'],
                  previous_attempts_to_repeat=attempts,
                  reuse_policy='Repeat all cases of changed frameworks; retain completed compatible cases and existing terminal failures of unchanged frameworks; repeat interrupted cases; never select attempts by metric value',
                  corrected_component=('PyCaret evaluation deadlines and durable candidate checkpoints' if 'pycaret' in rerun_frameworks else
                                       'TPOT worker memory profile' if 'tpot' in rerun_frameworks else 'H2O candidate validation ranking'))
    atomic_json(folder / 'continuation.json', report)
    return set(reused) | set(carried_failures)


def summarize(folder):
    rows = []
    plan = json.loads((folder / 'plan.json').read_text()) if (folder / 'plan.json').exists() else []
    entries = [(folder / 'runs' / entry['name'] / 'status.json', entry) for entry in plan]
    if not plan:
        entries = [(p, {}) for p in sorted((folder / 'runs').glob('*/status.json'))]
    for status, entry in entries:
        row = json.loads(status.read_text()) if status.exists() else dict(entry, status='not_started')
        for key in ['framework', 'scenario', 'seed', 'status', 'smoke', 'test_f1_macro', 'search_wall_seconds',
                    'final_refit_seconds', 'preprocessing_seconds', 'inference_seconds', 'reason', 'error']:
            row.setdefault(key, None)
        metrics_path = status.parent / 'metrics.json'
        row['eligible_for_comparison'] = row['status'] == 'completed'
        row['eligible_for_official_comparison'] = row['eligible_for_comparison'] and row.get('mode') == 'official'
        if metrics_path.exists() and row['eligible_for_comparison']:
            data = json.loads(metrics_path.read_text())
            for key, value in data['test'].items():
                if key != 'confusion_matrix_labels_0_1':
                    row[f'test_{key}'] = value
        rows.append(row)
    fields = sorted({key for row in rows for key in row})
    if fields:
        with (folder / 'summary.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    # Descriptive statistics only; repetitions share observations and are not
    # assumed independent for p-values or standard-error confidence intervals.
    groups = {}
    for row in rows:
        if row['status'] == 'completed':
            groups.setdefault((row['framework'], row['scenario']), []).append(row['test_f1_macro'])
    import statistics
    aggregate = [{'framework': a, 'scenario': b, 'completed_seeds': len(values),
                  'mean_f1_macro': statistics.mean(values),
                  'sample_sd_f1_macro': statistics.stdev(values) if len(values) > 1 else None}
                 for (a, b), values in sorted(groups.items())]
    (folder / 'descriptive-summary.json').write_text(json.dumps(aggregate, indent=2) + '\n')
    coverage = []
    for scenario in sorted({r['scenario'] for r in rows}):
        cases = [r for r in rows if r['scenario'] == scenario and r['status'] != 'unsupported']
        tools = sorted({r['framework'] for r in cases})
        completed = {tool: sorted({r['seed'] for r in cases if r['framework'] == tool and r['status'] == 'completed'}) for tool in tools}
        common = sorted(set.intersection(*(set(v) for v in completed.values()))) if completed else []
        coverage.append({'scenario': scenario, 'completed_seeds_by_framework': completed, 'common_completed_seeds': common,
                         'complete_for_comparison': bool(cases) and all(r['status'] == 'completed' for r in cases)})
    (folder / 'comparison-coverage.json').write_text(json.dumps(coverage, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rerun-frameworks', nargs='+', default=[], help='Repeat every case of these changed frameworks; preserve terminal outcomes of unchanged tools')
    parser.add_argument('--resume-from', help='Reuse completed cases and compatible prepared arrays from a stopped official run')
    parser.add_argument('--retry-first', nargs='+', default=[], help='Prioritize these planned experiment names')
    parser.add_argument('--workers', type=int, help='Concurrent containers, including preparation; default uses all configured slots')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--frameworks', nargs='+')
    parser.add_argument('--scenarios', nargs='+')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--smoke-budget', type=int, default=120)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--qualify', action='store_true', help='Full-data runtime check; validation only, no external test metrics')
    parser.add_argument('--qualification-budget', type=int, default=300)
    args = parser.parse_args()
    STOP.clear()
    def terminate(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, terminate)
    (ROOT / 'outputs').mkdir(exist_ok=True)
    lock = (ROOT / 'outputs' / '.benchmark.lock').open('w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error('Another v2 orchestrator is active; use --workers in a single orchestrator')
    try:
        config = validate_config(json.loads((ROOT / 'config.json').read_text()))
    except ValueError as exc:
        parser.error(str(exc))
    workers = args.workers if args.workers is not None else len(config['execution_slots'])
    if not 1 <= workers <= len(config['execution_slots']):
        parser.error('--workers must be between 1 and the number of configured execution slots')
    slots = config['execution_slots'][:workers]
    try:
        resource_layout = host_slots(slots, config['cpus'], config['memory_gib'])
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    if args.smoke and args.qualify:
        parser.error('--smoke and --qualify are mutually exclusive')
    if args.smoke_budget <= 0 or args.qualification_budget <= 0:
        parser.error('Validation budgets must be positive')
    frameworks = args.frameworks or config['frameworks']
    scenarios = args.scenarios or (['baseline'] if args.smoke or args.qualify else config['scenarios'])
    seeds = [args.seed] if args.seed is not None else ([42] if args.smoke or args.qualify else config['seeds'])
    if not set(frameworks) <= set(config['frameworks']) or not set(scenarios) <= set(config['scenarios']):
        parser.error('Unknown framework or scenario')
    if len(set(frameworks)) != len(frameworks) or len(set(scenarios)) != len(scenarios):
        parser.error('Repeated frameworks/scenarios are not allowed')
    if any(not 0 <= seed < 2**32-20001 for seed in seeds):
        parser.error('Seed out of supported range')
    mode = 'smoke' if args.smoke else ('qualification' if args.qualify else ('preparation' if args.prepare_only else 'official'))
    if args.rerun_frameworks and (not args.resume_from or not set(args.rerun_frameworks) <= set(config['frameworks'])):
        parser.error('--rerun-frameworks requires --resume-from and known tools')
    if args.resume_from and mode != 'official':
        parser.error('--resume-from is supported only for official runs')
    if args.retry_first and mode != 'official':
        parser.error('--retry-first is supported only for official runs')
    if mode == 'official':
        try:
            verify_review(ROOT)
        except ValueError as exc:
            parser.error(str(exc))
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    folder = ROOT / 'outputs' / f'{mode}-{stamp}'
    (folder / 'prepared').mkdir(parents=True)
    (folder / 'runs').mkdir()
    context = freeze_inputs(folder, frameworks, config, require_review=mode == 'official')
    budget = args.qualification_budget if args.qualify else (args.smoke_budget if args.smoke else config['search_budget_seconds'])
    (folder / 'protocol.json').write_text(json.dumps({
        'config': config, 'mode': mode, 'frameworks': frameworks, 'scenarios': scenarios,
        'seeds': seeds, 'source_sha256': context['source_sha256'], 'image_ids': context['image_ids'],
        'dataset_sha256': context['dataset_sha256'], 'source_snapshot': str(context['code']),
        'search_budget_seconds': budget, 'workers': workers, 'resource_layout': resource_layout,
    }, indent=2) + '\n')
    plan = [] if args.prepare_only else [{'name': f'{tool}-{scenario}-{seed}', 'framework': tool,
                                          'scenario': scenario, 'seed': seed, 'smoke': args.smoke, 'mode': mode}
                                         for seed in seeds for scenario in scenarios for tool in frameworks]
    (folder / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    if not set(args.retry_first) <= {entry['name'] for entry in plan}:
        parser.error('--retry-first contains an experiment outside the selected plan')
    reused = import_completed_run(args.resume_from, folder, plan, config, context, args.rerun_frameworks) if args.resume_from else set()
    priority_names = set(args.retry_first)
    print('Output:', folder, flush=True)
    print('Reused terminal experiments:', len(reused), flush=True)
    groups = []
    for seed in seeds:
        for scenario in scenarios:
            active = []
            for framework in frameworks:
                allowed, reason = capability(framework, scenario)
                if allowed and f'{framework}-{scenario}-{seed}' not in reused:
                    active.append(framework)
                elif not allowed and not args.prepare_only:
                    skipped = folder / 'runs' / f'{framework}-{scenario}-{seed}'
                    skipped.mkdir()
                    atomic_json(skipped / 'status.json', dict(framework=framework, scenario=scenario,
                        seed=seed, smoke=args.smoke, mode=mode, status='unsupported', reason=reason))
            if active or args.prepare_only:
                groups.append(dict(seed=seed, scenario=scenario, frameworks=[] if args.prepare_only else active,
                                   active_frameworks=active))

    def prepare_group(group, slot):
        seed, scenario = group['seed'], group['scenario']
        name = f'{scenario}-{seed}'
        prepared = folder / 'prepared' / name
        if (prepared / 'manifest.json').exists():
            print(f'Reusing preparation: {name}', flush=True)
            return 0
        command = ['/code/prepare.py', '--seed', str(seed), '--scenario', scenario,
                   '--output', '/outputs/' + str(prepared.relative_to(ROOT / 'outputs'))]
        if args.smoke:
            command += ['--limit', '2500']
        print(f'Preparing: {name} [slot {slot[0]}, CPUs {slot[1]}]', flush=True)
        try:
            code = docker('prepare', command, folder / f'prepare-{name}.log',
                          timeout=config['preprocessing_budget_seconds'], context=slot_context(context, slot))
        except subprocess.TimeoutExpired:
            code = 124
        if code:
            atomic_json(folder / f'prepare-{name}-failed.json', dict(scenario=scenario, seed=seed, exit_code=code))
            if not args.prepare_only:
                for framework in group['active_frameworks']:
                    skipped = folder / 'runs' / f'{framework}-{scenario}-{seed}'
                    skipped.mkdir()
                    atomic_json(skipped / 'status.json', dict(framework=framework, scenario=scenario,
                        seed=seed, smoke=args.smoke, mode=mode, status='skipped_preprocessing_failed',
                        preprocessing_exit_code=code))
        return code

    def run_experiment(group, slot, framework):
        seed, scenario = group['seed'], group['scenario']
        prepared = folder / 'prepared' / f'{scenario}-{seed}'
        output = folder / 'runs' / f'{framework}-{scenario}-{seed}'
        output.mkdir()
        command = ['/code/run.py', '--framework', framework, '--scenario', scenario, '--seed', str(seed),
                   '--prepared', '/outputs/' + str(prepared.relative_to(ROOT / 'outputs')),
                   '--output', '/outputs/' + str(output.relative_to(ROOT / 'outputs')), '--budget', str(budget)]
        if args.smoke:
            command.append('--smoke')
        if args.qualify:
            command.append('--validation-only')
        print(f'Running: {output.name} [slot {slot[0]}, CPUs {slot[1]}]', flush=True)
        wall_start = time.perf_counter()
        terminal = None
        try:
            code = docker(framework, command, output / 'execution.log', context=slot_context(context, slot),
                          timeout=budget+config['postprocessing_budget_seconds']+120)
        except subprocess.TimeoutExpired:
            code, terminal = 124, 'container_timeout'
        status_path = output / 'status.json'
        status = json.loads(status_path.read_text()) if status_path.exists() else {}
        if STOP.is_set() and status.get('status') not in {'completed', 'qualified'}:
            terminal = 'interrupted'
        if terminal:
            status.update(status=terminal)
        elif code and status.get('status') in {'completed', 'qualified'}:
            status.update(status='failed', error='Container exited with a nonzero code after writing results')
        elif status.get('status') not in TERMINAL:
            status.update(status='failed', error='Container exited without a terminal result; see execution.log')
        status.update(framework=framework, scenario=scenario, seed=seed, smoke=args.smoke, mode=mode,
                      execution_slot=slot[0], cpu_set=slot[1], container_exit_code=code,
                      container_wall_seconds=time.perf_counter()-wall_start)
        atomic_json(status_path, status)
        return bool(code or status['status'] not in {'completed', 'qualified', 'unsupported'})

    interrupted = False
    try:
        def priority(task):
            kind, group, tool = task
            names = [f"{candidate}-{group['scenario']}-{group['seed']}" for candidate in group['frameworks']] if kind == 'prepare' else [f"{tool}-{group['scenario']}-{group['seed']}"]
            return 0 if priority_names.intersection(names) else 1
        failures = schedule(groups, slots, prepare_group, run_experiment, lambda: summarize(folder), STOP, priority=priority)
    except KeyboardInterrupt:
        interrupted, failures = True, 1
        print('Interrupted; all containers owned by this orchestrator have been stopped.', flush=True)
    except BaseException:
        summarize(folder)
        raise
    summarize(folder)
    final_states = []
    for entry in plan:
        status = folder / 'runs' / entry['name'] / 'status.json'
        final_states.append(json.loads(status.read_text()).get('status') if status.exists() else 'not_started')
    continuation = json.loads((folder / 'continuation.json').read_text()) if (folder / 'continuation.json').exists() else {}
    carried_count = len(continuation.get('reused_terminal_failures', {}))
    (folder / 'finished.json').write_text(json.dumps({'failures': failures + carried_count, 'new_execution_failures': failures,
        'carried_failures': carried_count, 'mode': mode, 'workers': workers, 'interrupted': interrupted,
        'reused_completed': len(continuation.get('reused_completed', {})),
        'planned_experiments': len(plan), 'all_planned_terminal': all(s in TERMINAL for s in final_states),
        'status_counts': {s: final_states.count(s) for s in sorted(set(final_states))}}, indent=2) + '\n')
    print('Finished:', folder, 'failures:', failures, flush=True)
    return bool(failures + carried_count)


if __name__ == '__main__':
    sys.exit(main())
