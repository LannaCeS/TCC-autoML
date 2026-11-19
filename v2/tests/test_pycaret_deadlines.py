import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from v2.adapters import PyCaretEvaluationWorker, pycaret_resource_profile, pycaret_deadline_search


FAKE_WORKER = '''
import json, sys, time, subprocess
from pathlib import Path
Path(sys.argv[1]).write_text('{}')
for line in sys.stdin:
    r = json.loads(line)
    if r.get('child_pid'):
        child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])
        Path(r['child_pid']).write_text(str(child.pid))
    time.sleep(r.get('sleep', 0))
    result = {'state': 'complete', 'f1_macro': .75, 'completed_monotonic': time.monotonic(), 'completed_epoch': time.time()}
    if r.get('fail'): result.update(state='failed', reason='deliberate invalid candidate')
    Path(r['result_path']).write_text(json.dumps(result))
'''


class PyCaretDeadlineTests(unittest.TestCase):
    def test_completed_checkpoint_survives_an_unfinished_better_tuning_trial(self):
        # Exercise the real scheduler, process isolation, native Optuna TPE,
        # checkpoint serialization and selection without fitting costly models.
        fake = '''
import json, sys, time, joblib
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from weighting import BalancedClassifier, PYCARET_MODELS
from optuna.distributions import FloatDistribution
folder = Path(sys.argv[1])
joblib.dump({'estimators': {n: BalancedClassifier(LogisticRegression()) for n in PYCARET_MODELS},
            'distributions': {n: {'estimator__C': FloatDistribution(.1, 2)} for n in PYCARET_MODELS},
            'columns': ['x0'], 'dtypes': ['float32']}, folder/'catalog.joblib')
Path(sys.argv[2]).write_text('{}')
for line in sys.stdin:
    r = json.loads(line)
    if r['params']: time.sleep(30)
    value = {'state': 'complete', 'f1_macro': .99 if r['params'] else .75,
             'completed_epoch': time.time(), 'completed_monotonic': time.monotonic()}
    Path(r['result_path']).write_text(json.dumps(value))
'''
        with tempfile.TemporaryDirectory() as name:
            output = Path(name)
            settings = dict(native_budget_seconds=7, seed=42, scenario='baseline', cpus=8)
            factory = lambda command, folder: PyCaretEvaluationWorker([sys.executable, '-c', fake, str(folder)], folder)
            with patch('adapters.PyCaretEvaluationWorker', side_effect=factory):
                model, metadata = pycaret_deadline_search(settings, output)
            self.assertEqual(metadata['validation_f1_macro'], .75)
            self.assertGreaterEqual(metadata['timed_out_evaluations'], 1)
            checkpoint = json.loads((output/'pycaret-best-checkpoint.json').read_text())
            self.assertEqual(checkpoint['f1_macro'], .75)
            self.assertTrue(Path(checkpoint['model_path']).exists())
            self.assertEqual(checkpoint['seed'], 42)
            self.assertTrue(metadata['refit_required'])
            self.assertFalse(hasattr(model.model, 'estimator_'))
            history = [json.loads(line) for line in (output/'pycaret-evaluations.jsonl').read_text().splitlines()]
            self.assertEqual(history[-1]['state'], 'timeout')
            self.assertIsNone(history[-1]['f1_macro'])

    def test_slow_evaluation_is_killed_and_next_candidate_succeeds(self):
        with tempfile.TemporaryDirectory() as name:
            worker = PyCaretEvaluationWorker([sys.executable, '-c', FAKE_WORKER], Path(name))
            try:
                self.assertTrue(worker.start(time.monotonic() + 3))
                first_pid = worker.process.pid
                started = time.monotonic()
                result = worker.evaluate({'id': 'slow', 'sleep': 30}, started + .15)
                self.assertEqual(result['state'], 'timeout')
                self.assertLess(time.monotonic() - started, 2)
                self.assertFalse(Path(f'/proc/{first_pid}').exists())
                result = worker.evaluate({'id': 'fast'}, time.monotonic() + 3)
                self.assertEqual(result['f1_macro'], .75)
                self.assertNotEqual(worker.process.pid, first_pid)
            finally:
                worker.stop()

    def test_timed_out_trial_does_not_leave_live_descendants(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            worker = PyCaretEvaluationWorker([sys.executable, '-c', FAKE_WORKER], folder)
            try:
                worker.start(time.monotonic() + 3)
                result = worker.evaluate({'id': 'slow', 'sleep': 30, 'child_pid': str(folder/'pid')}, time.monotonic() + .3)
                self.assertEqual(result['state'], 'timeout')
                pid = int((folder/'pid').read_text())
                state = Path(f'/proc/{pid}/stat')
                self.assertTrue(not state.exists() or state.read_text().split()[2] == 'Z')
            finally:
                worker.stop()

    def test_no_new_worker_starts_after_deadline(self):
        with tempfile.TemporaryDirectory() as name:
            worker = PyCaretEvaluationWorker([sys.executable, '-c', FAKE_WORKER], Path(name))
            try:
                result = worker.evaluate({'id': 'late'}, time.monotonic() - 1)
                self.assertEqual(result['state'], 'timeout')
                self.assertIsNone(worker.process)
                self.assertFalse((Path(name)/'late-result.json').exists())
            finally:
                worker.stop()

    def test_failed_evaluation_is_not_a_zero_score_and_worker_is_reused(self):
        with tempfile.TemporaryDirectory() as name:
            worker = PyCaretEvaluationWorker([sys.executable, '-c', FAKE_WORKER], Path(name))
            try:
                result = worker.evaluate({'id': 'bad', 'fail': True}, time.monotonic() + 3)
                pid = worker.process.pid
                self.assertEqual(result['state'], 'failed')
                result = worker.evaluate({'id': 'good'}, time.monotonic() + 3)
                self.assertEqual(result['state'], 'complete')
                self.assertEqual(worker.process.pid, pid)
            finally:
                worker.stop()

    def test_worker_initialization_failure_is_visible(self):
        with tempfile.TemporaryDirectory() as name:
            worker = PyCaretEvaluationWorker([sys.executable, '-c', 'raise RuntimeError("setup error")'], Path(name))
            try:
                with self.assertRaisesRegex(RuntimeError, 'initialization failed'):
                    worker.start(time.monotonic() + 3)
            finally:
                worker.stop()

    def test_execution_limits_require_positive_finite_values(self):
        config = json.loads((Path(__file__).parents[1]/'config.json').read_text())
        self.assertEqual(pycaret_resource_profile(config)['evaluation_seconds'], 180)
        for key, value in [('evaluation_seconds', 0), ('minimum_start_seconds', -1),
                           ('comparison_max_seconds', True), ('comparison_fraction', 1)]:
            modified = dict(config, pycaret_execution=dict(config['pycaret_execution'], **{key: value}))
            with self.assertRaises(ValueError):
                pycaret_resource_profile(modified)


if __name__ == '__main__':
    unittest.main()
