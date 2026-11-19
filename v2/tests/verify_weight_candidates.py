"""Native integration checks for every allowed family, including PyCaret tuning."""
import argparse
import functools
import json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import PredefinedSplit
from sklearn.datasets import make_classification
from v2.common import macro_f1, write_json
from v2.weighting import BalancedClassifier, PYCARET_MODELS, TPOT_MODELS, pycaret_tuning_grid, register_pycaret_weighted_model


def main(framework, output):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    X, y = make_classification(n_samples=240, n_features=8, weights=[.85, .15], random_state=13)
    X = (X-X.min(axis=0))/(X.max(axis=0)-X.min(axis=0))
    train, val = np.arange(180), np.arange(180, 240)
    checks = []
    if framework == 'tpot':
        from v2.weighting import tpot_classifier_node
        for name in TPOT_MODELS:
            node = tpot_classifier_node(name, n_classes=2, n_samples=180, n_features=8, random_state=42, n_jobs=1)
            for mode in [False, True]:
                estimator = node.generate(rng=3).export_pipeline()
                model = BalancedClassifier(estimator, weighted=mode).fit(X[train], y[train])
                audit = model.fit_weight_audit_
                assert audit['applied'] == mode and audit['training_rows'] == 180
                assert audit['class_counts'] == np.bincount(y[train]).tolist()
                if mode: np.testing.assert_allclose(audit['class_weight_mass'], [90, 90])
                clone(model).fit(X[train], y[train]).predict(X[val])
                checks.append(dict(family=name, weighted=mode, status='passed', fit_audit=audit))
    else:
        from pycaret.classification import ClassificationExperiment
        from v2.adapters import FrameModel
        exp = ClassificationExperiment()
        dev = pd.DataFrame(X);dev.columns=[f'x{i}' for i in range(X.shape[1])];dev['Class']=y
        cv = PredefinedSplit(np.r_[np.full(180, -1), np.zeros(60, dtype=int)])
        exp.setup(data=dev, test_data=dev.iloc[val], target='Class', preprocess=False,
                  index=False, n_jobs=1, memory=False, html=False, verbose=False, session_id=42,
                  fold_strategy=cv, system_log=str(output/'pycaret.log'))
        exp.add_metric('f1_macro','F1 Macro',macro_f1,target='pred',greater_is_better=True)
        original = BalancedClassifier.fit
        for name in PYCARET_MODELS:
            seen=[]
            @functools.wraps(original)
            def record(self, X, y):
                result=original(self,X,y);seen.append(dict(self.fit_weight_audit_));return result
            native=exp._all_models_internal[name]
            candidate=BalancedClassifier(native.class_def(**native.args),weighted=True)
            with patch.object(BalancedClassifier,'fit',record):
                selected=exp.compare_models(include=[candidate],fold=cv,sort='F1 Macro',errors='raise',verbose=False)
                assert isinstance(selected,BalancedClassifier)
                grid=pycaret_tuning_grid(exp,selected)
                assert not any('class_weight' in k for k in grid)
                register_pycaret_weighted_model(exp, selected)
                tuned,tuner=exp.tune_model(selected,fold=cv,n_iter=1,optimize='F1 Macro',
                    custom_grid=grid,search_library='optuna',search_algorithm='tpe',
                    choose_better=False,return_tuner=True,verbose=False,timeout=10)
                assert tuner.timeout == 10
            train_fits=[a for a in seen if a['training_rows']==180]
            assert len(train_fits)>=2,(name,seen)
            assert all(a['applied'] and a['class_counts']==np.bincount(y[train]).tolist() for a in train_fits)
            for a in train_fits:np.testing.assert_allclose(a['class_weight_mass'],[90,90])
            final=FrameModel(clone(tuned),list(exp.get_config('X_train').columns),
                             [str(d) for d in exp.get_config('X_train').dtypes],weighted=True)
            final.fit(X[train],y[train])
            independent=macro_f1(y[val],final.predict(X[val]))
            assert abs(independent-tuner.best_score_)<1e-10,(name,independent,tuner.best_score_)
            checks.append(dict(family=name,status='passed',cv_and_tuning_training_fits=len(train_fits),
                               final_fit_audit=final.fit_weight_audit_,native_macro_matches_training_only_refit=True))
            print('PASS',name,flush=True)
    write_json(output/'report.json',dict(status='passed',framework=framework,checks=checks))
    print('PASS all allowed families:',framework,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--framework',choices=['pycaret','tpot'],required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();main(args.framework,args.output)
