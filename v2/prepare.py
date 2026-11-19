"""Generate common, grouped train/validation/test data in the PyCaret image."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from v2.common import grouped_partitions, grouped_blending, sha256, write_json
from v2.protocol import validate_config, recipe_hash, verify_snapshot


def training_variant(X, y, row_ids, parts, scenario, seed):
    scaler = StandardScaler().fit(X.iloc[parts['train']])
    arrays = {}
    for name, indices in parts.items():
        arrays[f'X_{name}'] = scaler.transform(X.iloc[indices])
        arrays[f'y_{name}'] = y[indices]
        arrays[f'rows_{name}'] = row_ids[indices]
    before = np.bincount(arrays['y_train'], minlength=2)
    if scenario not in {'baseline', 'class_weighted'}:
        from imblearn.over_sampling import RandomOverSampler, SMOTE
        from imblearn.under_sampling import EditedNearestNeighbours, TomekLinks
        from imblearn.combine import SMOTEENN, SMOTETomek
        samplers = {'random_over': RandomOverSampler(random_state=seed), 'smote': SMOTE(random_state=seed),
                    'editednearestneighbors': EditedNearestNeighbours(n_jobs=8), 'tomeklinks': TomekLinks(n_jobs=8),
                    'smoteenn': SMOTEENN(random_state=seed, n_jobs=8),
                    'smotetomek': SMOTETomek(random_state=seed, n_jobs=8)}
        if scenario not in samplers:
            raise ValueError(f'Unknown scenario: {scenario}')
        arrays['X_train'], arrays['y_train'] = samplers[scenario].fit_resample(arrays['X_train'], arrays['y_train'])
    counts = np.bincount(arrays['y_train'], minlength=2)
    if min(counts) == 0:
        raise ValueError('Sampler removed an entire class')
    arrays['weights_train'] = len(arrays['y_train']) / (2 * counts[arrays['y_train']])
    info = {
        'split_sizes_before_sampling': {k: len(v) for k, v in parts.items()},
        'fraud_counts_before_sampling': {k: int(y[v].sum()) for k, v in parts.items()},
        'training_class_counts_before': {str(i): int(v) for i, v in enumerate(before)},
        'training_class_counts_after': {str(i): int(v) for i, v in enumerate(counts)},
        'scaler_fit_rows': row_ids[parts['train']].tolist(),
        'scaler_mean': scaler.mean_.tolist(), 'scaler_scale': scaler.scale_.tolist(),
    }
    return arrays, info


def prepare(dataset, output, seed, scenario, limit=0):
    start = time.perf_counter()
    output = Path(output)
    if (output / 'manifest.json').exists():
        raise FileExistsError(f'Refusing to overwrite prepared data: {output}')
    output.mkdir(parents=True, exist_ok=True)
    code_root = Path(__file__).resolve().parent
    snapshot = verify_snapshot(code_root)
    config = validate_config(json.loads((code_root / 'config.json').read_text()))
    source_sha = sha256(dataset)
    if snapshot and source_sha != snapshot['dataset_sha256']:
        raise ValueError('Dataset differs from the frozen input snapshot')
    data = pd.read_csv(dataset)
    if sha256(dataset) != source_sha:
        raise ValueError('Dataset changed while it was being read')
    if 'Class' not in data or set(data.Class.unique()) != {0, 1}:
        raise ValueError('Expected binary Class labels 0 and 1')
    if data.isna().any().any() or not np.isfinite(data.to_numpy()).all():
        raise ValueError('Missing or non-finite data must be handled explicitly')
    # Smoke subsampling uses whole groups and never produces official metrics.
    X = data.drop(columns='Class')
    hashes = pd.util.hash_pandas_object(X, index=False).to_numpy()
    if pd.Series(hashes).nunique() != len(X.drop_duplicates()):
        raise ValueError('Feature hash collision detected')
    _, first, inverse = np.unique(hashes, return_index=True, return_inverse=True)
    source_labels = data.Class.to_numpy()
    if not np.array_equal(source_labels[first][inverse], source_labels):
        raise ValueError('Identical feature rows have conflicting labels in the source dataset')
    row_ids = np.arange(len(data))
    if limit:
        unique = pd.DataFrame({'group': hashes, 'label': data.Class}).drop_duplicates('group')
        rng = np.random.default_rng(seed)
        chosen = []
        for label in [0, 1]:
            pool = unique.loc[unique.label == label, 'group'].to_numpy()
            count = min(len(pool), max(100 if label else 1, int(limit * len(pool) / len(unique))))
            chosen.extend(rng.choice(pool, count, replace=False))
        mask = np.isin(hashes, chosen)
        data = data.loc[mask].reset_index(drop=True)
        row_ids, hashes = row_ids[mask], hashes[mask]
        X = data.drop(columns='Class')
    y = data.Class.to_numpy(dtype=int)
    parts = grouped_partitions(y, hashes, seed, config['test_group_fraction'],
                               config['validation_group_fraction_within_development'])
    base, blending = grouped_blending(y, hashes, parts['train'], seed,
                                      config['h2o_blending_fraction_of_training'])
    h2o_parts = dict(parts, train=base, blending=blending)
    shared_seconds = time.perf_counter() - start
    stage = time.perf_counter()
    arrays, main_info = training_variant(X, y, row_ids, parts, scenario, seed)
    np.savez_compressed(output / 'data.npz', **arrays)
    main_seconds = shared_seconds + time.perf_counter() - stage
    stage = time.perf_counter()
    h2o_arrays, h2o_info = training_variant(X, y, row_ids, h2o_parts, scenario, seed)
    np.savez_compressed(output / 'data_h2o.npz', **h2o_arrays)
    h2o_info.update(preprocessing_seconds=shared_seconds + time.perf_counter() - stage,
                    data_npz_sha256=sha256(output / 'data_h2o.npz'),
                    blending_fraction_of_original_training=config['h2o_blending_fraction_of_training'],
                    blending_preserves_natural_prevalence=True, blending_weights=1.0,
                    group_overlap=0)
    saved_parts = {k: row_ids[v] for k, v in parts.items()}
    saved_parts.update(h2o_base_train=row_ids[base], h2o_blending=row_ids[blending])
    np.savez_compressed(output / 'partitions.npz', **saved_parts)
    write_json(output / 'manifest.json', {
        'seed': seed, 'scenario': scenario, 'smoke': bool(limit), 'dataset_sha256': source_sha,
        'protocol_version': config['protocol_version'], 'recipe_sha256': recipe_hash(code_root, config),
        'data_npz_sha256': sha256(output / 'data.npz'), 'partitions_sha256': sha256(output / 'partitions.npz'),
        'preprocessing_seconds': main_seconds,
        'shared_input_preparation_seconds': shared_seconds,
        'total_preparation_wall_seconds': time.perf_counter() - start,
        'grouping': 'identical feature rows, independent of labels', 'group_overlap': 0,
        **main_info, 'h2o_blending': h2o_info,
        'columns': X.columns.tolist(),
        'packages': {k: importlib.metadata.version(k) for k in ['pandas', 'numpy', 'scikit-learn', 'imbalanced-learn']},
    })
    print('Prepared', output, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='/data/creditcard.csv')
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--scenario', required=True)
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()
    prepare(args.dataset, args.output, args.seed, args.scenario, args.limit)
