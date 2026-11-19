"""Verify each saved ensemble after clearing all models from the H2O cluster."""
import argparse
import json
from pathlib import Path
import h2o
import numpy as np
import pandas as pd
from v2.common import macro_f1, write_json
from v2.h2o_metric import MacroF1Loss


def main(run):
    root = Path(run)
    selection = json.loads((root / 'selection.json').read_text())
    job = json.loads((root / 'job.json').read_text())
    with np.load(job['search_data']) as data:
        X, y = data['X_validation'], data['y_validation']
    table = pd.read_csv(root / 'validation-leaderboard.csv').set_index('model_id')
    assert selection['ensemble_count'] >= 1 and selection['stacking_enabled']
    checks = []
    h2o.init(nthreads=4, max_mem_size='2G')
    try:
        for name, path in selection['ensemble_paths'].items():
            h2o.remove_all()
            h2o.upload_custom_metric(MacroF1Loss, func_name='macro_f1_loss', func_file='h2o_macro_metric.py')
            assert not h2o.api('GET /3/Models')['models']
            model = h2o.load_model(path)
            assert model.algo == 'stackedensemble'
            features = pd.DataFrame(X, columns=[f'x{i}' for i in range(X.shape[1])])
            weight_column = model.actual_params.get('weights_column') or ('__weight' if job['scenario'] == 'class_weighted' else None)
            if weight_column:
                features[weight_column] = 1.0
            frame = h2o.H2OFrame(features)
            probability = model.predict(frame).as_data_frame()['p1'].to_numpy()
            observed = macro_f1(y, (probability > .5).astype(int))
            expected = 1 - float(table.loc[name, 'custom'])
            assert abs(observed - expected) < 1e-10, (observed, expected)
            metalearner_id = model._model_json['output']['metalearner']['name']
            metalearner = h2o.get_model(metalearner_id)
            training_rows = int(metalearner.model_performance(train=True).nobs())
            assert training_rows == selection['blending_rows'], (training_rows, selection['blending_rows'])
            checks.append({'model': name, 'status': 'passed', 'loaded_into_empty_model_store': True,
                           'validation_f1_macro': observed, 'matches_original_leaderboard': True,
                           'metalearner_training_rows': training_rows,
                           'metalearner_stopping_metric': metalearner.actual_params.get('stopping_metric')})
            print('PASS: ensemble loaded independently and predictions verified:', name, flush=True)
    finally:
        h2o.cluster().shutdown(prompt=False)
    write_json(root / 'ensemble-reload-validation.json', {'status': 'passed', 'checks': checks})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    main(parser.parse_args().run)
