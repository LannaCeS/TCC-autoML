import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from v2.adapters import rank_h2o_candidates
from v2.orchestrate import import_completed_run, ROOT
from v2.protocol import digest_file


class H2OCandidateTests(unittest.TestCase):
    def test_missing_custom_candidate_can_win_and_no_candidate_is_dropped(self):
        native = pd.DataFrame({'model_id':['finite','missing'], 'custom':[.2,np.nan]})
        called=[]
        def probability(name):
            called.append(name)
            return [.1,.9] if name=='missing' else [.1,.2]
        table=rank_h2o_candidates(native,[0,1],probability)
        self.assertEqual(called,['finite','missing'])
        self.assertEqual(table.iloc[0].model_id,'missing')
        self.assertEqual(table.iloc[0].f1_macro,1)
        self.assertEqual(int(table.native_custom_missing.sum()),1)
        self.assertTrue(np.isnan(native.iloc[1]['custom']))

    def test_finite_cached_scores_do_not_override_actual_validation_predictions(self):
        table=rank_h2o_candidates(pd.DataFrame({'model_id':['wrong_best','actual_best'],'custom':[0,.9]}),
                                  [0,1],lambda name:[.1,.9] if name=='actual_best' else [.1,.2])
        self.assertEqual(table.iloc[0].model_id,'actual_best')
        self.assertTrue(table['custom'].is_monotonic_increasing)
        self.assertTrue(table.native_custom_discrepancy.all())

    def test_all_metrics_missing_or_column_absent_is_recoverable(self):
        table=rank_h2o_candidates(pd.DataFrame({'model_id':['a','b']}),[0,1],lambda name:[.1,.9])
        self.assertEqual(len(table),2)
        self.assertTrue(table.native_custom_missing.all())
        self.assertTrue(np.isfinite(table['custom']).all())

    def test_invalid_models_do_not_get_silently_excluded(self):
        for probability in [[.1],[.1,np.nan],[.1,1.1]]:
            with self.assertRaises(RuntimeError):
                rank_h2o_candidates(pd.DataFrame({'model_id':['x']}),[0,1],lambda name:probability)
        with self.assertRaises(RuntimeError):
            rank_h2o_candidates(pd.DataFrame({'model_id':['x','x']}),[0,1],lambda name:[.1,.9])


class ContinuationTests(unittest.TestCase):
    def fixture(self, root):
        for name in ['common.py','prepare.py','run.py','protocol.py','weighting.py','h2o_metric.py','compose.yaml','adapters.py']:
            shutil.copy2(ROOT/name,root/name)
        previous=root/'outputs/old';source=previous/'source';source.mkdir(parents=True)
        for p in root.iterdir():
            if p.is_file():shutil.copy2(p,source/p.name)
        (source/'SNAPSHOT.json').write_text(json.dumps({'source_sha256':{p.name:digest_file(p) for p in source.iterdir() if p.is_file()}}))
        config=json.loads((ROOT/'config.json').read_text())
        context={'dataset_sha256':'DATA','image_ids':{'autogluon':'IMAGE'}}
        (previous/'protocol.json').write_text(json.dumps(dict(mode='official',config=config,**context)))
        (previous/'finished.json').write_text('{}')
        target=root/'outputs/new';(target/'runs').mkdir(parents=True);(target/'prepared').mkdir()
        plan=[]
        for tool,status in [('autogluon','completed'),('h2o','failed')]:
            entry=dict(name=f'{tool}-baseline-42',framework=tool,scenario='baseline',seed=42,mode='official');plan.append(entry)
            run=previous/'runs'/entry['name'];run.mkdir(parents=True)
            (run/'status.json').write_text(json.dumps(dict(entry,status=status,container_exit_code=0 if status=='completed' else 1)))
            for filename in ['test-predictions.csv','selection.json','selection-frozen.json','selected-model.joblib','preprocessing.json','resources.json']:
                (run/filename).write_text('artifact')
            (run/'metrics.json').write_text(json.dumps({'validation_only':False,'test':{'f1_macro':.1 if status=='completed' else .99}}))
        return previous,target,plan,config,context

    def test_only_successful_complete_cases_are_reused_regardless_of_test_score(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);previous,target,plan,config,context=self.fixture(root)
            with patch('orchestrate.ROOT',root):
                reused=import_completed_run(previous,target,plan,config,context)
            self.assertEqual(reused,{'autogluon-baseline-42'})
            self.assertTrue((target/'runs/autogluon-baseline-42').is_symlink())
            self.assertFalse((target/'runs/h2o-baseline-42').exists())
            report=json.loads((target/'continuation.json').read_text())
            self.assertEqual(report['previous_failed_or_interrupted_attempts'][0]['previous_status'],'failed')

    def test_changed_h2o_training_space_cannot_reuse_results(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);previous,target,plan,config,context=self.fixture(root)
            old=previous/'source/adapters.py'
            old.write_text(old.read_text().replace("algorithms = ['GBM',", "algorithms = ['GLM',"))
            snapshot=previous/'source/SNAPSHOT.json'
            values=json.loads(snapshot.read_text());values['source_sha256']['adapters.py']=digest_file(old)
            snapshot.write_text(json.dumps(values))
            with patch('orchestrate.ROOT',root),self.assertRaisesRegex(ValueError,'H2O training configuration'):
                import_completed_run(previous,target,plan,config,context)

    def test_tpot_profile_change_requires_explicit_rerun_and_preserves_other_failures(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);previous,target,plan,config,context=self.fixture(root)
            path=previous/'protocol.json';old=json.loads(path.read_text())
            old['config'].pop('tpot_execution',None);path.write_text(json.dumps(old))
            with patch('orchestrate.ROOT',root),self.assertRaisesRegex(ValueError,'Scientific configuration'):
                import_completed_run(previous,target,plan,config,context)
            with patch('orchestrate.ROOT',root):
                reused=import_completed_run(previous,target,plan,config,context,rerun_frameworks=['tpot'])
            self.assertEqual(reused,{'autogluon-baseline-42','h2o-baseline-42'})
            report=json.loads((target/'continuation.json').read_text())
            self.assertEqual(set(report['reused_completed']),{'autogluon-baseline-42'})
            self.assertEqual(set(report['reused_terminal_failures']),{'h2o-baseline-42'})

    def test_all_completed_tpot_cases_are_repeated_under_the_new_profile(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);previous,target,plan,config,context=self.fixture(root)
            entry=dict(name='tpot-baseline-42',framework='tpot',scenario='baseline',seed=42,mode='official')
            plan.append(entry);run=previous/'runs'/entry['name'];run.mkdir()
            (run/'status.json').write_text(json.dumps(dict(entry,status='completed',container_exit_code=0)))
            with patch('orchestrate.ROOT',root):
                import_completed_run(previous,target,plan,config,context,rerun_frameworks=['tpot'])
            self.assertFalse((target/'runs'/entry['name']).exists())
            report=json.loads((target/'continuation.json').read_text())
            self.assertEqual(report['previous_completed_superseded'][0]['name'],entry['name'])

    def test_rerun_flag_does_not_allow_changing_the_container_budget(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);previous,target,plan,config,context=self.fixture(root)
            config['memory_gib']=32
            with patch('orchestrate.ROOT',root),self.assertRaisesRegex(ValueError,'Scientific configuration'):
                import_completed_run(previous,target,plan,config,context,rerun_frameworks=['tpot'])

    def test_pycaret_policy_change_requires_explicit_rerun(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);previous,target,plan,config,context=self.fixture(root)
            path=previous/'protocol.json';old=json.loads(path.read_text())
            old['config'].pop('pycaret_execution');path.write_text(json.dumps(old))
            with patch('orchestrate.ROOT',root),self.assertRaisesRegex(ValueError,'Scientific configuration'):
                import_completed_run(previous,target,plan,config,context)
            with patch('orchestrate.ROOT',root):
                reused=import_completed_run(previous,target,plan,config,context,rerun_frameworks=['pycaret'])
            self.assertEqual(reused,{'autogluon-baseline-42','h2o-baseline-42'})

    def test_completed_and_timed_out_pycaret_cases_are_both_superseded(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);previous,target,plan,config,context=self.fixture(root)
            for seed,status in [(42,'completed'),(101,'budget_exceeded')]:
                entry=dict(name=f'pycaret-baseline-{seed}',framework='pycaret',scenario='baseline',seed=seed,mode='official')
                plan.append(entry);run=previous/'runs'/entry['name'];run.mkdir()
                (run/'status.json').write_text(json.dumps(dict(entry,status=status)))
            with patch('orchestrate.ROOT',root):
                import_completed_run(previous,target,plan,config,context,rerun_frameworks=['pycaret'])
            report=json.loads((target/'continuation.json').read_text())
            self.assertEqual(len(report['previous_completed_superseded']),1)
            self.assertEqual(len(report['previous_attempts_to_repeat']),2)
            self.assertFalse((target/'runs/pycaret-baseline-42').exists())
            self.assertFalse((target/'runs/pycaret-baseline-101').exists())

    def test_changed_scientific_configuration_cannot_reuse_results(self):
        with tempfile.TemporaryDirectory() as name:
            root=Path(name);previous,target,plan,config,context=self.fixture(root)
            config['search_budget_seconds']=900
            with patch('orchestrate.ROOT',root),self.assertRaisesRegex(ValueError,'Scientific configuration'):
                import_completed_run(previous,target,plan,config,context)


if __name__=='__main__':unittest.main()
