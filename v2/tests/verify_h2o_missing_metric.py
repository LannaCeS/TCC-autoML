"""Recover missing native metrics using real saved models and validation only."""
import argparse
import json
from pathlib import Path
import joblib
import h2o
import numpy as np
import pandas as pd
from v2.adapters import rank_h2o_candidates
from v2.common import write_json
from v2.h2o_metric import MacroF1Loss


def main(run):
    root=Path(run)
    selection=json.loads((root/'selection.json').read_text())
    job=json.loads((root/'job.json').read_text())
    with np.load(job['search_data']) as data:
        X,y=data['X_validation'],data['y_validation']
    paths=dict(selection['ensemble_paths'])
    saved=joblib.load(root/'selected-model.joblib')
    paths[selection['selected']]=saved.path
    table=pd.read_csv(root/'validation-leaderboard.csv')
    subset=table.loc[table.model_id.isin(paths),['model_id','custom']].copy()
    probability={}
    h2o.init(nthreads=4,max_mem_size='2G')
    try:
        for name in subset.model_id:
            h2o.remove_all()
            h2o.upload_custom_metric(MacroF1Loss,func_name='macro_f1_loss',func_file='h2o_macro_metric.py')
            model=h2o.load_model(paths[name])
            features=pd.DataFrame(X,columns=[f'x{i}' for i in range(X.shape[1])])
            if job['scenario']=='class_weighted':features['__weight']=1.0
            frame=h2o.H2OFrame(features)
            probability[name]=model.predict(frame).as_data_frame()['p1'].to_numpy()
        expected=rank_h2o_candidates(subset,y,lambda name:probability[name])
        missing=subset.copy();missing['custom']=np.nan
        recovered=rank_h2o_candidates(missing,y,lambda name:probability[name])
        assert recovered.model_id.tolist()==expected.model_id.tolist()
        np.testing.assert_array_equal(recovered['custom'],expected['custom'])
        assert recovered.native_custom_missing.all()
        for row in expected.itertuples():
            native=float(subset.loc[subset.model_id==row.model_id,'custom'].iloc[0])
            assert abs(native-row.custom)<1e-10
    finally:
        h2o.cluster().shutdown(prompt=False)
    write_json(root/'missing-metric-recovery-validation.json',dict(status='passed',
        native_metric_deliberately_missing=True,independent_ranking_recovered=True,
        candidates=len(subset),external_test_used=False,reloaded_into_empty_model_store=True))
    print('PASS: missing metrics recovered for',len(subset),'real candidates; validation only')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);args=parser.parse_args();main(args.run)
