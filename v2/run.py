"""Isolated search and finalization phases, each with an enforced deadline."""
import argparse
import gc
import importlib.metadata
import json
import os
import random
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import joblib
import numpy as np
import pandas as pd
from v2.common import metrics, predict, resource_snapshot, sha256, supported, write_json
from v2.protocol import validate_config, verify_snapshot, recipe_hash

CODE = Path(__file__).resolve().parent


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)


def worker(job_path):
    verify_snapshot(CODE)
    job = json.loads(Path(job_path).read_text())
    seed_all(job['seed'])
    output = Path(job['output'])
    with np.load(job['search_data']) as archive:
        if any('test' in name for name in archive.files):
            raise ValueError('External test data must not be passed to search')
        arrays = {k: archive[k] for k in archive.files}
    from v2.adapters import search
    model, metadata = search(job['framework'], arrays, job, output)
    joblib.dump(model, output / 'selected-model.joblib')
    write_json(output / 'selection.json', metadata)


def finalize(job_path):
    verify_snapshot(CODE)
    job = json.loads(Path(job_path).read_text())
    seed_all(job['seed'])
    output = Path(job['output'])
    if sha256(job['prepared_data']) != job['prepared_data_sha256']:
        raise ValueError('Prepared data changed between search and finalization')
    with np.load(job['prepared_data']) as archive:
        arrays = {k: archive[k] for k in archive.files if not job['validation_only'] or 'test' not in k}
    selection = json.loads((output / 'selection.json').read_text())
    timings = {}
    model = None
    try:
        start = time.perf_counter()
        model = joblib.load(output / 'selected-model.joblib')
        timings['artifact_load_seconds'] = time.perf_counter() - start
        start = time.perf_counter()
        if selection['refit_required']:
            seed_all(job['seed'])
            model.fit(arrays['X_train'], arrays['y_train'])
            joblib.dump(model, output / 'selected-model.joblib')
        if job['framework'] in {'pycaret', 'tpot'}:
            from v2.weighting import weight_audits
            audits = weight_audits(model)
            if len(audits) != 1 or audits[0]['applied'] != (job['scenario'] == 'class_weighted') or audits[0]['training_rows'] != len(arrays['y_train']):
                raise ValueError('Selected-model training weight audit is missing or inconsistent')
            write_json(output / 'training-weight-audit.json', audits[0])
        timings['final_refit_seconds'] = time.perf_counter() - start if selection['refit_required'] else 0
        start = time.perf_counter()
        val_pred, val_score, score_type = predict(model, arrays['X_validation'])
        validation = metrics(arrays['y_validation'], val_pred, val_score)
        timings['validation_seconds'] = time.perf_counter() - start
        native = selection.get('validation_f1_macro')
        if native is None or not np.isfinite(native) or not 0 <= native <= 1:
            raise ValueError('Selected model lacks a valid native macro-F1 selection score')
        if not selection['refit_required'] and not np.isclose(validation['f1_macro'], native, atol=1e-10, rtol=0):
            raise ValueError('Reloaded model validation score differs from the score used for selection')
        timings['validation_refit_score_delta'] = validation['f1_macro'] - native
        write_json(output / 'selection-frozen.json', {'selection': selection, 'validation': validation,
                                                     'frozen_epoch': time.time()})
        result = {'validation': validation, 'score_type': score_type, 'validation_only': job['validation_only']}
        if not job['validation_only']:
            start = time.perf_counter()
            predict(model, arrays['X_validation'][:min(1000, len(val_pred))])
            timings['warmup_seconds'] = time.perf_counter() - start
            start = time.perf_counter()
            test_pred, test_score, score_type = predict(model, arrays['X_test'])
            timings['inference_seconds'] = time.perf_counter() - start
            test = metrics(arrays['y_test'], test_pred, test_score)
            pd.DataFrame({'row_id': arrays['rows_test'], 'y_true': arrays['y_test'], 'y_pred': test_pred,
                          'score_class_1': test_score if test_score is not None else np.nan}).to_csv(output / 'test-predictions.csv', index=False)
            result.update(test=test, majority_baseline=metrics(arrays['y_test'], np.zeros_like(test_pred)), score_type=score_type)
            timings['test_f1_macro'] = test['f1_macro']
        write_json(output / 'metrics.json', result)
    finally:
        if model is not None and hasattr(model, 'close'):
            model.close()
    write_json(output / 'finalization.json', timings)


def load_prepared(args, config, snapshot):
    folder = Path(args.prepared)
    manifest = json.loads((folder / 'manifest.json').read_text())
    if manifest['seed'] != args.seed or manifest['scenario'] != args.scenario or manifest['smoke'] != args.smoke:
        raise ValueError('Prepared-data metadata does not match this run')
    if manifest.get('protocol_version') != config['protocol_version'] or manifest.get('recipe_sha256') != recipe_hash(CODE, config):
        raise ValueError('Prepared data uses an obsolete or different protocol/recipe; regenerate it')
    if snapshot and manifest['dataset_sha256'] != snapshot['dataset_sha256']:
        raise ValueError('Prepared dataset differs from frozen input')
    if sha256(folder / 'partitions.npz') != manifest['partitions_sha256']:
        raise ValueError('Partition checksum mismatch')
    variant = manifest.get('h2o_blending') if args.framework == 'h2o' else manifest
    if variant is None:
        raise ValueError('H2O blending data is missing')
    path = folder / ('data_h2o.npz' if args.framework == 'h2o' else 'data.npz')
    if sha256(path) != variant['data_npz_sha256']:
        raise ValueError('Prepared data checksum mismatch')
    with np.load(path) as archive:
        arrays = {k: archive[k] for k in archive.files}
    names = ['train', 'validation', 'test'] + (['blending'] if args.framework == 'h2o' else [])
    for name in names:
        X, y = arrays[f'X_{name}'], arrays[f'y_{name}']
        if X.ndim != 2 or X.shape != (len(y), len(manifest['columns'])) or not np.isfinite(X).all():
            raise ValueError(f'Invalid feature matrix: {name}')
        if y.ndim != 1 or set(np.unique(y)) != {0, 1}:
            raise ValueError(f'Invalid binary target: {name}')
        if name != 'train' and len(arrays[f'rows_{name}']) != len(y):
            raise ValueError(f'Row ID mismatch: {name}')
    counts = np.bincount(arrays['y_train'], minlength=2)
    expected_weights = len(arrays['y_train']) / (2 * counts[arrays['y_train']])
    if not np.allclose(arrays['weights_train'], expected_weights, rtol=1e-12):
        raise ValueError('Training weights are inconsistent with training class counts')
    return manifest, variant, path, arrays


class PhaseTimeout(TimeoutError):
    pass


def bounded(command, seconds, seed, status, phase):
    started = time.perf_counter()
    process = subprocess.Popen(command, start_new_session=True,
                               env=dict(os.environ, PYTHONHASHSEED=str(seed)))
    def stop():
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        except ProcessLookupError:
            process.wait()
    try:
        return process.wait(timeout=seconds)
    except subprocess.TimeoutExpired:
        status[f'{phase}_deadline_trigger_seconds'] = time.perf_counter() - started
        stop()
        raise PhaseTimeout(phase)
    except BaseException:
        stop()
        raise
    finally:
        status[f'{phase}_wall_seconds'] = time.perf_counter() - started


def experiment(args):
    output = Path(args.output)
    if (output / 'status.json').exists():
        raise FileExistsError(f'Refusing to overwrite an existing run: {output}')
    output.mkdir(parents=True, exist_ok=True)
    config = validate_config(json.loads((CODE / 'config.json').read_text()))
    snapshot = verify_snapshot(CODE)
    mode = 'qualification' if args.validation_only else ('smoke' if args.smoke else 'official')
    status = {'framework': args.framework, 'scenario': args.scenario, 'seed': args.seed,
              'status': 'running', 'mode': mode, 'protocol_version': config['protocol_version'],
              'primary_metric': 'f1_macro', 'smoke': args.smoke,
              'requested_search_budget_seconds': args.budget, 'started_epoch': time.time(),
              'execution_slot': os.environ.get('BENCHMARK_SLOT'),
              'cpu_set': os.environ.get('BENCHMARK_CPUSET'),
              'actual_cpu_affinity': sorted(os.sched_getaffinity(0))}
    write_json(output / 'status.json', status)
    allowed, reason = supported(args.framework, args.scenario)
    if not allowed:
        status.update(status='unsupported', reason=reason)
        write_json(output / 'status.json', status)
        return 0
    try:
        if args.budget <= 0:
            raise ValueError('Search budget must be positive')
        if mode == 'official' and snapshot is None:
            raise ValueError('Official experiments require the frozen snapshot created by orchestrate.py')
        manifest, variant, data_path, arrays = load_prepared(args, config, snapshot)
        status.update(dataset_sha256=manifest['dataset_sha256'], partitions_sha256=manifest['partitions_sha256'],
                      preprocessing_seconds=variant['preprocessing_seconds'],
                      training_variant='h2o_base_and_blending' if args.framework == 'h2o' else 'common_training')
        write_json(output / 'preprocessing.json', {'columns': manifest['columns'],
                    'scaler_mean': variant['scaler_mean'], 'scaler_scale': variant['scaler_scale'],
                    'training_variant': status['training_variant'], 'recipe_sha256': manifest['recipe_sha256'],
                    'dataset_sha256': manifest['dataset_sha256'], 'partitions_sha256': manifest['partitions_sha256']})
        keys = ['X_train', 'y_train', 'X_validation', 'y_validation', 'weights_train']
        if args.framework == 'h2o':
            keys += ['X_blending', 'y_blending']
        np.savez_compressed(output / 'search-data.npz', **{k: arrays[k] for k in keys})
        del arrays
        gc.collect()
        search_job = dict(status, output=str(output), search_data=str(output / 'search-data.npz'),
                          native_budget_seconds=max(1, min(args.budget * config['native_search_budget_fraction'],
                              config['qualification_native_search_seconds']) if args.validation_only
                              else args.budget * config['native_search_budget_fraction']),
                          cpus=config['cpus'], require_stacked_candidate=config['h2o_require_stacked_candidate'])
        write_json(output / 'search-job.json', search_job)
        job = dict(search_job, prepared_data=str(data_path), prepared_data_sha256=variant['data_npz_sha256'],
                   validation_only=args.validation_only)
        write_json(output / 'job.json', job)
        write_json(output / 'packages.json', {d.metadata['Name']: d.version for d in importlib.metadata.distributions()})
        status['phase'] = 'search'
        write_json(output / 'status.json', status)
        code = bounded([sys.executable, __file__, '--worker', str(output / 'search-job.json')],
                       args.budget, args.seed, status, 'search')
        if code:
            raise RuntimeError(f'Search subprocess failed with exit code {code}; see execution.log')
        status['phase'] = 'postprocessing'
        write_json(output / 'status.json', status)
        code = bounded([sys.executable, __file__, '--finalize', str(output / 'job.json')],
                       config['postprocessing_budget_seconds'], args.seed, status, 'postprocessing')
        if code:
            raise RuntimeError(f'Finalization subprocess failed with exit code {code}; see execution.log')
        status.update(json.loads((output / 'finalization.json').read_text()))
        status.update(status='qualified' if args.validation_only else 'completed', phase='finished')
    except PhaseTimeout as exc:
        status.update(status='budget_exceeded' if str(exc) == 'search' else 'postprocessing_timeout', error=str(exc))
    except KeyboardInterrupt:
        status.update(status='interrupted', error='Interrupted by user')
    except BaseException as exc:
        status.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        traceback.print_exc()
    finally:
        status['ended_epoch'] = time.time()
        status['total_wall_seconds'] = status['ended_epoch'] - status['started_epoch']
        write_json(output / 'resources.json', resource_snapshot())
        write_json(output / 'status.json', status)
    return 0 if status['status'] in {'completed', 'qualified'} else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker')
    parser.add_argument('--finalize')
    parser.add_argument('--framework')
    parser.add_argument('--scenario', default='baseline')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--budget', type=int, default=1800)
    parser.add_argument('--prepared')
    parser.add_argument('--output')
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--validation-only', action='store_true')
    args = parser.parse_args()
    if args.worker:
        worker(args.worker)
    elif args.finalize:
        finalize(args.finalize)
    else:
        sys.exit(experiment(args))
