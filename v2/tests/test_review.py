import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import time
import unittest
import numpy as np
import pandas as pd
from v2.common import predict
from v2.protocol import validate_config, digest_file, verify_snapshot, capability, verify_review, REVIEW_FILES
from v2.prepare import prepare
from v2.run import load_prepared, bounded, PhaseTimeout, finalize
from v2.orchestrate import summarize


class ReviewTests(unittest.TestCase):
    def test_exact_probability_tie_is_negative_in_h2o_metric(self):
        from v2.h2o_metric import MacroF1Loss
        self.assertEqual(MacroF1Loss().map([0, .5, .5], [0], 1, 0, None), [0, 1, 0, 0])

    def test_invalid_predictions_are_rejected(self):
        class Bad:
            def predict(self, X): return np.array([.9, 1])
        with self.assertRaisesRegex(ValueError, 'binary label'):
            predict(Bad(), np.zeros((2, 1)))
        class BadProb:
            classes_ = [0, 1]
            def predict(self, X): return np.array([0, 1])
            def predict_proba(self, X): return np.array([[.9, .9], [.1, .9]])
        with self.assertRaisesRegex(ValueError, 'sum to one'):
            predict(BadProb(), np.zeros((2, 1)))

    def test_optional_score_interfaces_can_be_unavailable(self):
        class HardOnly:
            def predict(self, X): return np.array([0, 1])
            def predict_proba(self, X): raise AttributeError('unavailable')
            def decision_function(self, X): raise AttributeError('unavailable')
        _, score, _ = predict(HardOnly(), np.zeros((2, 1)))
        self.assertIsNone(score)

    def test_unknown_configuration_is_rejected(self):
        config = json.loads((Path(__file__).parents[1] / 'config.json').read_text())
        validate_config(config)
        config['seeds'] = [42, 42]
        with self.assertRaisesRegex(ValueError, 'Seeds'):
            validate_config(config)
        with self.assertRaisesRegex(ValueError, 'Unknown'):
            capability('unknown', 'baseline')

    def test_frozen_code_tampering_is_rejected(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            p = root / 'sample.py'
            p.write_text('original')
            (root / 'SNAPSHOT.json').write_text(json.dumps({'source_sha256': {'sample.py': digest_file(p)}}))
            self.assertIsNotNone(verify_snapshot(root))
            p.write_text('changed')
            with self.assertRaisesRegex(ValueError, 'Frozen source changed'):
                verify_snapshot(root)

    def test_official_review_gate_rejects_changed_code(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            for filename in REVIEW_FILES:
                (root / filename).write_text(filename)
            (root / 'REVIEW.json').write_text(json.dumps({'status': 'passed', 'reviewed_sha256':
                {filename: digest_file(root / filename) for filename in REVIEW_FILES}}))
            self.assertEqual(verify_review(root)['status'], 'passed')
            (root / 'config.json').write_text('changed')
            with self.assertRaisesRegex(ValueError, 'changed since review'):
                verify_review(root)

    def test_search_and_postprocessing_deadlines_terminate_processes(self):
        for phase in ['search', 'postprocessing']:
            status = {}
            with self.assertRaises(PhaseTimeout):
                bounded([sys.executable, '-c', 'import time; time.sleep(30)'], .05, 42, status, phase)
            self.assertGreaterEqual(status[f'{phase}_deadline_trigger_seconds'], .05)
            self.assertLess(status[f'{phase}_wall_seconds'], 6)

    def test_prepared_artifacts_are_checked_before_model_search(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            dataset = root / 'input.csv'
            pd.DataFrame({'x': np.arange(200, dtype=float), 'z': np.sin(np.arange(200)),
                          'Class': np.arange(200) % 2}).to_csv(dataset, index=False)
            folder = root / 'prepared'
            prepare(dataset, folder, 42, 'baseline')
            args = SimpleNamespace(prepared=str(folder), seed=42, scenario='baseline', smoke=False, framework='flaml')
            config = json.loads((Path(__file__).parents[1] / 'config.json').read_text())
            load_prepared(args, config, None)
            with (folder / 'partitions.npz').open('ab') as stream:
                stream.write(b'changed')
            with self.assertRaisesRegex(ValueError, 'Partition checksum'):
                load_prepared(args, config, None)
            with self.assertRaises(FileExistsError):
                prepare(dataset, folder, 42, 'baseline')

    def test_reloaded_score_mismatch_blocks_external_test_evaluation(self):
        import joblib
        from sklearn.dummy import DummyClassifier
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            data = root / 'data.npz'
            np.savez(data, X_validation=np.zeros((2, 1)), y_validation=np.array([0, 1]))
            model = DummyClassifier(strategy='constant', constant=0).fit([[0], [1]], [0, 1])
            joblib.dump(model, root / 'selected-model.joblib')
            (root / 'selection.json').write_text(json.dumps({'refit_required': False, 'validation_f1_macro': .9}))
            job = dict(framework='flaml', scenario='baseline', seed=42, output=str(root), prepared_data=str(data), prepared_data_sha256=digest_file(data), validation_only=False)
            (root / 'job.json').write_text(json.dumps(job))
            with self.assertRaisesRegex(ValueError, 'differs from the score used for selection'):
                finalize(root / 'job.json')
            self.assertFalse((root / 'metrics.json').exists())
            self.assertFalse((root / 'test-predictions.csv').exists())

    def test_reports_include_missing_runs_and_exclude_partial_scores(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            run = folder / 'runs' / 'flaml-baseline-42'
            run.mkdir(parents=True)
            plan = [{'name': 'flaml-baseline-42', 'framework': 'flaml', 'scenario': 'baseline', 'seed': 42},
                    {'name': 'tpot-baseline-42', 'framework': 'tpot', 'scenario': 'baseline', 'seed': 42}]
            (folder / 'plan.json').write_text(json.dumps(plan))
            (run / 'status.json').write_text(json.dumps(dict(plan[0], status='failed')))
            (run / 'metrics.json').write_text(json.dumps({'test': {'f1_macro': .99}}))
            summarize(folder)
            text = (folder / 'summary.csv').read_text()
            self.assertIn('not_started', text)
            self.assertNotIn('0.99', text)
            coverage = json.loads((folder / 'comparison-coverage.json').read_text())[0]
            self.assertFalse(coverage['complete_for_comparison'])
            self.assertEqual(coverage['common_completed_seeds'], [])


if __name__ == '__main__':
    unittest.main()
