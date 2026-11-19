"""Eight historical timeout cases: full-volume, validation-only, bounded checks."""
import fcntl
import json
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from v2.orchestrate import ROOT, freeze_inputs, docker, slot_context
from v2.protocol import digest_file, verify_snapshot


def main():
    config = json.loads((ROOT/'config.json').read_text())
    lock = (ROOT/'outputs/.benchmark.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    previous = Path(json.loads((ROOT/'outputs/active-official.json').read_text())['output_directory'])
    analysis = json.loads((ROOT/'outputs/pycaret-timeout-analysis.json').read_text())
    cases = [(case['scenario'], case['seed']) for case in analysis['observed_cases']]
    folder = ROOT/'outputs'/('pycaret-deadline-qualification-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'))
    (folder/'runs').mkdir(parents=True)
    context = freeze_inputs(folder, ['pycaret'], config)
    # Only the validation-mode native duration changes in this test snapshot.
    # Official production settings and prepared-data recipes stay unchanged.
    native = 180
    path = context['code']/'config.json'
    validation_config = dict(config, qualification_native_search_seconds=native)
    path.write_text(json.dumps(validation_config, indent=2)+'\n')
    snapshot_path = context['code']/'SNAPSHOT.json'
    snapshot = json.loads(snapshot_path.read_text())
    snapshot['source_sha256']['config.json'] = digest_file(path)
    snapshot_path.write_text(json.dumps(snapshot, indent=2)+'\n')
    context['source_sha256'] = snapshot['source_sha256']
    verify_snapshot(context['code'])
    (folder/'validation-settings.json').write_text(json.dumps(dict(
        production_native_search_seconds=1620, qualification_native_search_seconds=native,
        outer_search_seconds=240, external_test_used=False,
        production_adapter_sha256=digest_file(ROOT/'adapters.py')), indent=2)+'\n')
    print('Output:', folder, flush=True)

    def slot(index):
        checks = []
        for scenario, seed in cases[index::4]:
            output = folder/'runs'/f'pycaret-{scenario}-{seed}'
            output.mkdir()
            prepared = previous/'prepared'/f'{scenario}-{seed}'
            command = ['/code/run.py', '--framework', 'pycaret', '--scenario', scenario,
                       '--seed', str(seed), '--prepared', '/outputs/'+str(prepared.relative_to(ROOT/'outputs')),
                       '--output', '/outputs/'+str(output.relative_to(ROOT/'outputs')),
                       '--budget', '240', '--validation-only']
            code = docker('pycaret', command, output/'execution.log',
                          context=slot_context(context, (index, config['execution_slots'][index])), timeout=1320)
            state = json.loads((output/'status.json').read_text())
            print(scenario, seed, state['status'], flush=True)
            assert code == 0 and state['status'] == 'qualified', (scenario, seed, state)
            selection = json.loads((output/'selection.json').read_text())
            checkpoint = json.loads((output/'pycaret-best-checkpoint.json').read_text())
            history = [json.loads(line) for line in (output/'pycaret-evaluations.jsonl').read_text().splitlines()]
            complete = [row for row in history if row['state'] == 'complete']
            assert checkpoint['f1_macro'] == max(row['f1_macro'] for row in complete)
            assert selection['validation_f1_macro'] == checkpoint['f1_macro']
            assert all(row['completed_monotonic'] < row['evaluation_deadline_monotonic'] for row in complete)
            assert checkpoint['seed'] == seed and checkpoint['weighted'] == (scenario == 'class_weighted')
            assert abs(state['validation_refit_score_delta']) < 1e-10
            assert state['search_wall_seconds'] < 240
            assert not selection['native_fit_includes_unused_train_validation_refit']
            assert selection['execution_resources']['concurrent_evaluations'] == 1
            audit = json.loads((output/'training-weight-audit.json').read_text())
            assert audit['applied'] == (scenario == 'class_weighted')
            assert 'test' not in json.loads((output/'metrics.json').read_text())
            assert not (output/'test-predictions.csv').exists()
            checks.append(dict(scenario=scenario, seed=seed, status='qualified',
                               search_wall_seconds=state['search_wall_seconds'],
                               successful_evaluations=selection['successful_evaluations'],
                               timed_out_evaluations=selection['timed_out_evaluations'],
                               failed_evaluations=selection['failed_evaluations'],
                               selected=selection['selected'], checkpoint_verified=True,
                               validation_score_reproduced=True, weighted=audit['applied'],
                               external_test_used=False, run=str(output)))
        return checks
    with ThreadPoolExecutor(max_workers=4) as pool:
        checks = [check for group in pool.map(slot, range(4)) for check in group]
    assert len(checks) == 8 and any(check['timed_out_evaluations'] for check in checks)
    (folder/'report.json').write_text(json.dumps(dict(status='passed', checks=checks,
        real_candidate_timeouts_observed=True, external_test_used=False), indent=2)+'\n')
    (ROOT/'outputs/pycaret-deadline-qualification-latest.json').write_text(json.dumps(
        dict(folder=str(folder), report=str(folder/'report.json')), indent=2)+'\n')
    print('PASS: eight full-volume cases; timeout recovery, checkpoints and weighting verified', flush=True)


if __name__ == '__main__':
    main()
