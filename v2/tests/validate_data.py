"""Integration checks on the complete CSV and on all sampler code paths."""
import json
import argparse
import time
from pathlib import Path
import numpy as np
import pandas as pd
from v2.common import grouped_partitions, grouped_blending, sha256, write_json
from v2.prepare import prepare


def main(output):
    output = Path(output)
    output.mkdir(exist_ok=True)
    data = pd.read_csv('/data/creditcard.csv')
    features = data.drop(columns='Class')
    groups = pd.util.hash_pandas_object(features, index=False).to_numpy()
    report = {'dataset_sha256': sha256('/data/creditcard.csv'), 'rows': len(data), 'seeds': [], 'samplers': []}
    for seed in [42, 101, 223, 302, 404]:
        parts = grouped_partitions(data.Class.to_numpy(), groups, seed)
        entry = {'seed': seed, 'group_overlap': 0}
        for name, indices in parts.items():
            entry[name] = {'rows': len(indices), 'frauds': int(data.Class.iloc[indices].sum())}
        base, blending = grouped_blending(data.Class.to_numpy(), groups, parts['train'], seed)
        entry['h2o_base_train'] = {'rows': len(base), 'frauds': int(data.Class.iloc[base].sum())}
        entry['h2o_blending'] = {'rows': len(blending), 'frauds': int(data.Class.iloc[blending].sum())}
        four = [base, blending, parts['validation'], parts['test']]
        for i, left in enumerate(four):
            for right in four[i+1:]:
                assert not np.intersect1d(groups[left], groups[right]).size
        entry['h2o_four_way_group_overlap'] = 0
        report['seeds'].append(entry)
    baseline, h2o_baseline = None, None
    for scenario in ['baseline', 'random_over', 'smote', 'editednearestneighbors',
                     'tomeklinks', 'smoteenn', 'smotetomek', 'class_weighted']:
        folder = output / scenario
        prepare('/data/creditcard.csv', folder, 42, scenario, limit=2500)
        manifest = json.loads((folder / 'manifest.json').read_text())
        with np.load(folder / 'data.npz') as archive:
            current = {name: archive[name] for name in ['X_validation', 'y_validation', 'X_test', 'y_test', 'rows_test']}
        if baseline is None:
            baseline = current
        else:
            for name in baseline:
                np.testing.assert_array_equal(current[name], baseline[name])
        train_rows = manifest['scaler_fit_rows']
        np.testing.assert_allclose(manifest['scaler_mean'], features.iloc[train_rows].mean(axis=0), rtol=1e-12, atol=1e-12)
        variant = manifest['h2o_blending']
        with np.load(folder / 'data_h2o.npz') as archive:
            h2o_current = {name: archive[name] for name in ['X_validation', 'y_validation', 'X_test', 'y_test',
                                                          'rows_test', 'X_blending', 'y_blending', 'rows_blending']}
        if h2o_baseline is None:
            h2o_baseline = h2o_current
        else:
            for name in h2o_baseline:
                np.testing.assert_array_equal(h2o_current[name], h2o_baseline[name])
        for name in ['rows_test', 'y_test', 'y_validation']:
            np.testing.assert_array_equal(current[name], h2o_current[name])
        np.testing.assert_allclose(variant['scaler_mean'], features.iloc[variant['scaler_fit_rows']].mean(axis=0),
                                   rtol=1e-12, atol=1e-12)
        assert not np.intersect1d(variant['scaler_fit_rows'], h2o_current['rows_blending']).size
        report['samplers'].append({'scenario': scenario, 'status': 'passed',
                                   'validation_and_test_unchanged': True,
                                   'h2o_blending_validation_and_test_unchanged': True,
                                   'h2o_scaler_fitted_on_base_training_only': True,
                                   'training_counts': manifest['training_class_counts_after']})
    report['status'] = 'passed'
    write_json(output / 'report.json', report)
    print('PASS: all five full-data group partitions and all eight preprocessing scenarios', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default=f'/outputs/data-validation-{time.time_ns()}')
    main(parser.parse_args().output)
