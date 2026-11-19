"""Framework adapters; imports are lazy so every tool runs in its own image."""
import time
from contextlib import contextmanager
import numpy as np
import pandas as pd
from v2.common import macro_f1


class H2OModel:
    """Persist a native model path, not a live connection to the worker's JVM."""
    def __init__(self, path, cpus, weight_column=None):
        self.path, self.cpus, self._model = path, cpus, None
        self.weight_column = weight_column

    @property
    def classes_(self):
        return np.array([0, 1])

    def _probability(self, X):
        import h2o
        if self._model is None:
            from v2.h2o_metric import MacroF1Loss
            h2o.init(nthreads=self.cpus, max_mem_size='10G')
            h2o.upload_custom_metric(MacroF1Loss, func_name='macro_f1_loss', func_file='h2o_macro_metric.py')
            self._model = h2o.load_model(self.path)
        features = pd.DataFrame(X, columns=[f'x{i}' for i in range(X.shape[1])])
        weight_column = getattr(self, 'weight_column', None) or self._model.actual_params.get('weights_column')
        if weight_column:
            features[weight_column] = 1.0
        frame = h2o.H2OFrame(features)
        predicted = None
        try:
            predicted = self._model.predict(frame)
            return predicted.as_data_frame()['p1'].to_numpy()
        finally:
            h2o.remove(frame)
            if predicted is not None:
                h2o.remove(predicted)

    def predict(self, X):
        return (self._probability(X) > .5).astype(int)

    def predict_proba(self, X):
        positive = self._probability(X)
        return np.c_[1 - positive, positive]

    def close(self):
        if self._model is not None:
            import h2o
            h2o.cluster().shutdown(prompt=False)
            self._model = None


class AutoGluonModel:
    def __init__(self, predictor, name):
        self.predictor, self.name = predictor, name

    @property
    def classes_(self):
        return self.predictor.class_labels

    def predict(self, X):
        return self.predictor.predict(pd.DataFrame(X), model=self.name).to_numpy()

    def predict_proba(self, X):
        return self.predictor.predict_proba(pd.DataFrame(X), model=self.name).to_numpy()


class FrameModel:
    """Keep PyCaret's native feature names and numeric precision at refit/inference."""
    def __init__(self, model, columns, dtypes, weighted=False):
        self.model, self.columns, self.dtypes = model, columns, dtypes
        self.weighted = weighted

    def _frame(self, X):
        return pd.DataFrame(X, columns=self.columns).astype(dict(zip(self.columns, self.dtypes)))

    def fit(self, X, y):
        from v2.weighting import fit_with_weights, BalancedClassifier
        if isinstance(self.model, BalancedClassifier):
            if self.model.weighted != self.weighted:
                raise ValueError('Inconsistent class weighting at final refit')
            self.model.fit(self._frame(X), y)
            self.fit_weight_audit_ = self.model.fit_weight_audit_
        else:
            self.fit_weight_audit_ = fit_with_weights(self.model, self._frame(X), y, self.weighted)
        return self

    @property
    def classes_(self):
        return self.model.classes_

    def predict(self, X):
        return self.model.predict(self._frame(X))

    def predict_proba(self, X):
        return self.model.predict_proba(self._frame(X))

    def decision_function(self, X):
        return self.model.decision_function(self._frame(X))


def pycaret_resource_profile(config):
    profile = dict(config['pycaret_execution'])
    for name in ['evaluation_seconds', 'comparison_max_seconds', 'minimum_start_seconds']:
        if type(profile[name]) is not int or profile[name] <= 0:
            raise ValueError(f'PyCaret {name} must be a positive integer')
    if not 0 < profile['comparison_fraction'] < 1:
        raise ValueError('PyCaret comparison_fraction must be between zero and one')
    return profile


def pycaret_atomic_json(path, value):
    import json
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


class PyCaretEvaluationWorker:
    """One reusable evaluator process; replace its entire process group on timeout."""
    def __init__(self, command, folder):
        self.command, self.folder, self.process, self.stream = command, folder, None, None
        self.sequence = 0

    def start(self, deadline):
        import subprocess
        if time.monotonic() >= deadline:
            return False
        if self.process is not None and self.process.poll() is None:
            return True
        self.stop()
        self.sequence += 1
        ready = self.folder / f'worker-{self.sequence}-ready.json'
        self.stream = (self.folder / f'worker-{self.sequence}.log').open('w')
        self.process = subprocess.Popen([*self.command, str(ready)], stdin=subprocess.PIPE,
                                        stdout=self.stream, stderr=subprocess.STDOUT,
                                        text=True, start_new_session=True)
        while time.monotonic() < deadline:
            if ready.exists():
                return True
            if self.process.poll() is not None:
                raise RuntimeError(f'PyCaret evaluator initialization failed; see worker-{self.sequence}.log')
            time.sleep(.02)
        self.stop()
        return False

    def stop(self):
        import os
        import signal
        import subprocess
        if self.process is not None:
            # Kill the group even if its leader exited with surviving children.
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                self.process.wait(timeout=.5)
            except subprocess.TimeoutExpired:
                pass
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self.process.wait(timeout=2)
            if self.process.stdin:
                self.process.stdin.close()
            self.process = None
        if self.stream is not None:
            self.stream.close()
            self.stream = None

    def evaluate(self, request, deadline):
        import json
        try:
            if not self.start(deadline):
                return {'state': 'timeout', 'reason': 'worker_initialization_deadline'}
        except RuntimeError as exc:
            self.stop()
            return {'state': 'failed', 'reason': str(exc)}
        result = self.folder / (request['id'] + '-result.json')
        request = dict(request, result_path=str(result))
        try:
            self.process.stdin.write(json.dumps(request, allow_nan=False) + '\n')
            self.process.stdin.flush()
        except BrokenPipeError:
            self.stop()
            return {'state': 'failed', 'reason': 'worker_pipe_closed'}
        while time.monotonic() < deadline:
            if result.exists():
                value = json.loads(result.read_text())
                # An observation is eligible only after its durable publication.
                if value['completed_monotonic'] <= deadline and time.monotonic() < deadline:
                    return value
                break
            if self.process.poll() is not None:
                self.stop()
                return {'state': 'failed', 'reason': 'worker_process_exited'}
            time.sleep(.02)
        self.stop()
        return {'state': 'timeout', 'reason': 'evaluation_deadline'}


def pycaret_evaluation_worker(job_path, ready_path):
    """Evaluate using PyCaret's native CV, with no redundant full-data refit."""
    import json
    import sys
    import traceback
    import joblib
    from pathlib import Path
    from sklearn.base import clone
    from sklearn.model_selection import PredefinedSplit
    from pycaret.classification import ClassificationExperiment
    from pycaret.internal.distributions import get_optuna_distributions
    from v2.weighting import PYCARET_MODELS, BalancedClassifier, assert_weight_support, pycaret_tuning_grid
    from v2.protocol import verify_snapshot
    verify_snapshot(Path(__file__).parent)
    job = json.loads(Path(job_path).read_text())
    output = Path(job['output'])
    with np.load(job['search_data']) as archive:
        if any('test' in name for name in archive.files):
            raise ValueError('External test data cannot enter PyCaret search')
        train, valid = archive['X_train'], archive['X_validation']
        y_train, y_valid = archive['y_train'], archive['y_validation']
    frame, validation = pd.DataFrame(np.vstack([train, valid])), pd.DataFrame(valid)
    frame.columns = validation.columns = [f'x{i}' for i in range(train.shape[1])]
    frame['Class'], validation['Class'] = np.r_[y_train, y_valid], y_valid
    cv = PredefinedSplit(np.r_[np.full(len(train), -1), np.zeros(len(valid), dtype=int)])
    exp = ClassificationExperiment()
    # One validation fold and one concurrent trial. Parallelism is inside the
    # estimator, bounded by the same eight assigned CPUs.
    exp.setup(data=frame, test_data=validation, target='Class', preprocess=False,
              session_id=job['seed'], n_jobs=1, index=False, html=False, verbose=False,
              system_log=str(Path(ready_path).with_suffix('.pycaret.log')), memory=False, fold_strategy=cv)
    exp.add_metric('f1_macro', 'F1 Macro', macro_f1, target='pred', greater_is_better=True)
    catalog = {'estimators': {}, 'distributions': {},
               'columns': list(exp.get_config('X_train').columns),
               'dtypes': [str(dtype) for dtype in exp.get_config('X_train').dtypes]}
    for family in PYCARET_MODELS:
        container = exp._all_models_internal[family]
        estimator = container.class_def(**container.args)
        if 'n_jobs' in estimator.get_params(deep=False):
            estimator.set_params(n_jobs=job['cpus'])
        assert_weight_support(estimator)
        model = BalancedClassifier(estimator, weighted=job['scenario'] == 'class_weighted')
        catalog['estimators'][family] = model
        catalog['distributions'][family] = get_optuna_distributions(pycaret_tuning_grid(exp, model))
    catalog_path = output / 'pycaret-evaluations' / 'catalog.joblib'
    temporary = catalog_path.with_suffix('.tmp')
    joblib.dump(catalog, temporary)
    temporary.replace(catalog_path)
    pycaret_atomic_json(Path(ready_path), {'ready': True})
    for line in sys.stdin:
        request = json.loads(line)
        result = {'state': 'failed'}
        try:
            model = clone(catalog['estimators'][request['family']]).set_params(**request['params'])
            # Explicit private option is pinned to PyCaret 3.3.2 and covered by
            # integration checks: no fit on combined train+validation rows.
            exp._create_model(model, fold=cv, round=12, cross_validation=True,
                              refit=False, verbose=False, system=False,
                              add_to_model_list=False, error_score='raise',
                              metrics={'f1_macro': exp._all_metrics['f1_macro']})
            table = exp.pull()
            score = float(table.loc['Mean', 'F1 Macro'])
            if not np.isfinite(score) or not 0 <= score <= 1:
                raise ValueError('Invalid PyCaret validation macro-F1')
            result = {'state': 'complete', 'f1_macro': score}
        except Exception as exc:
            result = {'state': 'failed', 'reason': f'{type(exc).__name__}: {exc}'}
            traceback.print_exc()
        result.update(completed_monotonic=time.monotonic(), completed_epoch=time.time())
        pycaret_atomic_json(Path(request['result_path']), result)


def pycaret_deadline_search(settings, output):
    """Keep each completed native PyCaret evaluation, then tune its best family."""
    started = time.monotonic()
    import json
    import signal
    import sys
    import joblib
    import optuna
    from pathlib import Path
    from sklearn.base import clone
    from v2.weighting import PYCARET_MODELS
    config = json.loads((Path(__file__).parent / 'config.json').read_text())
    profile = pycaret_resource_profile(config)
    deadline = started + settings['native_budget_seconds']
    folder = output / 'pycaret-evaluations'
    folder.mkdir()
    command = [sys.executable, '-c',
               'import sys; from adapters import pycaret_evaluation_worker; pycaret_evaluation_worker(sys.argv[1], sys.argv[2])',
               str(output / 'search-job.json')]
    worker = PyCaretEvaluationWorker(command, folder)
    prior_handler = signal.getsignal(signal.SIGTERM)
    def terminate(signum, frame):
        worker.stop()
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, terminate)
    history, best, catalog = [], None, None

    def evaluate(family, params, phase, budget):
        nonlocal best
        evaluation_started = time.monotonic()
        evaluation_deadline = min(deadline, evaluation_started + budget)
        request = {'id': f'evaluation-{len(history):05d}', 'family': family, 'params': params}
        result = worker.evaluate(request, evaluation_deadline)
        row = dict(request, seed=settings['seed'], weighted=settings['scenario'] == 'class_weighted',
                   phase=phase, state=result['state'],
                   elapsed_seconds=time.monotonic() - evaluation_started,
                   evaluation_budget_seconds=budget, started_monotonic=evaluation_started,
                   evaluation_deadline_monotonic=evaluation_deadline,
                   completed_epoch=result.get('completed_epoch'),
                   completed_monotonic=result.get('completed_monotonic'),
                   f1_macro=result.get('f1_macro'), error=result.get('reason'))
        history.append(row)
        # All parameters are JSON primitives from native Optuna distributions.
        with (output / 'pycaret-evaluations.jsonl').open('a') as stream:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
            stream.flush()
        if result['state'] == 'complete' and (best is None or row['f1_macro'] > best['f1_macro']):
            candidate = clone(catalog['estimators'][family]).set_params(**params)
            model = FrameModel(candidate, catalog['columns'], catalog['dtypes'],
                               weighted=settings['scenario'] == 'class_weighted')
            checkpoint = folder / (request['id'] + '-model.joblib')
            temporary = checkpoint.with_suffix('.tmp')
            joblib.dump(model, temporary)
            temporary.replace(checkpoint)
            best = dict(row, model_path=str(checkpoint))
            pycaret_atomic_json(output / 'pycaret-best-checkpoint.json', best)
        return row

    try:
        if not worker.start(deadline):
            raise RuntimeError('PyCaret initialization exhausted its native search budget')
        catalog = joblib.load(folder / 'catalog.joblib')
        comparison_deadline = min(deadline, time.monotonic() + min(
            profile['comparison_max_seconds'], settings['native_budget_seconds'] * profile['comparison_fraction']))
        for index, family in enumerate(PYCARET_MODELS):
            remaining = comparison_deadline - time.monotonic()
            if remaining < profile['minimum_start_seconds']:
                break
            # Reserve a fair opportunity for every family, including those after
            # a slow GBC. A timeout never becomes an artificial zero score.
            budget = min(profile['evaluation_seconds'], remaining / (len(PYCARET_MODELS) - index))
            evaluate(family, {}, 'comparison', budget)
        comparison_seconds = time.monotonic() - started
        if best is None:
            raise RuntimeError('No PyCaret candidate completed successfully within the comparison budget')
        family = best['family']
        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(
            seed=settings['seed'], multivariate=True, constant_liar=True), pruner=optuna.pruners.NopPruner())
        while deadline - time.monotonic() >= profile['minimum_start_seconds'] and len(study.trials) < 100000:
            trial = study.ask(fixed_distributions=catalog['distributions'][family])
            remaining = deadline - time.monotonic()
            if remaining < profile['minimum_start_seconds']:
                study.tell(trial, state=optuna.trial.TrialState.FAIL)
                break
            row = evaluate(family, trial.params, 'tuning', min(profile['evaluation_seconds'], remaining))
            if row['state'] == 'complete':
                study.tell(trial, row['f1_macro'])
            else:
                study.tell(trial, state=optuna.trial.TrialState.FAIL)
            # Durable after EVERY completed, failed or interrupted trial.
            temporary = output / 'tuning-trials.tmp'
            study.trials_dataframe().to_csv(temporary, index=False)
            temporary.replace(output / 'tuning-trials.csv')
        pd.DataFrame(history).drop(columns=['params']).to_csv(output / 'validation-leaderboard.csv', index=False)
        model = joblib.load(best['model_path'])
        return model, dict(selected=type(model.model.estimator).__name__,
                          candidate_families=list(PYCARET_MODELS), refit_required=True,
                          validation_f1_macro=best['f1_macro'], input_columns=catalog['columns'],
                          input_dtypes=catalog['dtypes'], compare_wall_seconds=comparison_seconds,
                          training_class_weights=settings['scenario'] == 'class_weighted',
                          weighting_policy='BalancedClassifier computes weights only on each training fit; macro-F1 is unweighted',
                          validation_policy='Native PyCaret CV uses the shared PredefinedSplit; no external test data',
                          execution_policy='Sequential native PyCaret evaluations in a reusable killable process; native Optuna TPE on the best comparison family',
                          execution_resources=dict(profile, concurrent_evaluations=1, estimator_cpus=settings['cpus']),
                          native_fit_includes_unused_train_validation_refit=False,
                          search_budget_seconds=settings['native_budget_seconds'],
                          search_elapsed_seconds=time.monotonic() - started,
                          successful_evaluations=sum(row['state'] == 'complete' for row in history),
                          timed_out_evaluations=sum(row['state'] == 'timeout' for row in history),
                          failed_evaluations=sum(row['state'] == 'failed' for row in history),
                          best_checkpoint=str(output / 'pycaret-best-checkpoint.json'),
                          selection_completed_epoch=best['completed_epoch'],
                          tuning=dict(attempted=bool(study.trials), trials=len(study.trials)))
    finally:
        worker.stop()
        signal.signal(signal.SIGTERM, prior_handler)


def tpot_resource_profile(config):
    """Reserve half the container memory for the driver and unmanaged data."""
    profile = config['tpot_execution']
    workers, gib = profile['workers'], profile['worker_memory_gib']
    if type(workers) is not int or type(gib) is not int or workers < 1 or gib < 1:
        raise ValueError('TPOT worker count and memory must be positive integers')
    if workers > config['cpus'] or workers * gib > config['memory_gib'] - 8:
        raise ValueError('TPOT workers must leave at least 8 GiB for the driver and other allocations')
    return dict(workers=workers, worker_memory_bytes=gib * 1024**3, threads_per_worker=1,
                driver_memory_reserve_bytes=(config['memory_gib'] - workers * gib) * 1024**3,
                memory_target=.60, memory_spill=.70, memory_pause=.80, memory_terminate=.90,
                monitor_interval='100ms')


@contextmanager
def tpot_dask_client(profile, output):
    """Two process workers with Dask spilling, pausing and nanny restarts."""
    import json
    import logging
    import threading
    from pathlib import Path
    import dask
    from distributed import Client, LocalCluster
    from v2.common import write_json
    settings = {f'distributed.worker.memory.{key}': profile['memory_' + key]
                for key in ['target', 'spill', 'pause', 'terminate']}
    settings['distributed.worker.memory.monitor-interval'] = profile['monitor_interval']
    settings['distributed.nanny.environ'] = dict(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    with dask.config.set(settings):
        with LocalCluster(n_workers=profile['workers'], threads_per_worker=profile['threads_per_worker'],
                          processes=True, memory_limit=profile['worker_memory_bytes'],
                          local_directory='/tmp/tpot-dask', dashboard_address=None,
                          silence_logs=logging.WARNING) as cluster:
            with Client(cluster) as client:
                client.wait_for_workers(profile['workers'], timeout=60)
                info = client.scheduler_info()['workers']
                if len(info) != profile['workers'] or any(
                        worker['memory_limit'] != profile['worker_memory_bytes'] or worker['nthreads'] != 1
                        for worker in info.values()):
                    raise RuntimeError('Dask did not apply the requested TPOT worker limits')
                write_json(output / 'tpot-worker-configuration.json', dict(profile=profile, workers=info,
                           memory_limit_semantics='Dask-managed RSS thresholds and nanny restarts; 16 GiB hard cap remains at container level'))
                stop = threading.Event()
                def monitor():
                    with (output / 'tpot-memory-samples.jsonl').open('a') as stream:
                        while not stop.is_set():
                            try:
                                workers = client.scheduler_info()['workers']
                                record = dict(epoch=time.time(), workers={address: {
                                    'memory_limit': worker['memory_limit'],
                                    'rss_bytes': worker.get('metrics', {}).get('memory'),
                                    'status': worker.get('status')}
                                    for address, worker in workers.items()})
                                for name in ['memory.current', 'memory.peak', 'memory.events']:
                                    path = Path('/sys/fs/cgroup') / name
                                    record[name] = path.read_text().strip() if path.exists() else None
                                stream.write(json.dumps(record) + '\n');stream.flush()
                            except Exception as exc:
                                stream.write(json.dumps({'monitor_error':str(exc)}) + '\n');stream.flush()
                            stop.wait(2)
                thread = threading.Thread(target=monitor, daemon=True)
                thread.start()
                try:
                    yield client
                finally:
                    stop.set();thread.join(timeout=5)
                    path = Path('/sys/fs/cgroup/memory.events')
                    write_json(output / 'tpot-memory-events.json', dict(
                        memory_events=path.read_text().strip() if path.exists() else None,
                        driver_memory_reserve_bytes=profile['driver_memory_reserve_bytes']))


def rank_h2o_candidates(native_table, y_validation, positive_probability):
    """Recompute the primary objective for every candidate, using validation only.

    H2O can leave custom metrics missing even for usable candidates. Neither
    missing nor discrepant cached metrics may silently eliminate a candidate.
    """
    if native_table.empty or 'model_id' not in native_table:
        raise RuntimeError('H2O returned no candidate models')
    if native_table['model_id'].isna().any() or native_table['model_id'].duplicated().any():
        raise RuntimeError('H2O candidate IDs are missing or duplicated')
    table = native_table.copy()
    native = pd.to_numeric(table.get('custom', pd.Series(np.nan, index=table.index)), errors='coerce')
    observed = []
    for candidate_id in table['model_id']:
        probability = np.asarray(positive_probability(candidate_id))
        if probability.shape != (len(y_validation),) or not np.isfinite(probability).all():
            raise RuntimeError(f'Invalid H2O validation probability output: {candidate_id}')
        if (probability < 0).any() or (probability > 1).any():
            raise RuntimeError(f'H2O validation probabilities outside [0, 1]: {candidate_id}')
        observed.append(macro_f1(y_validation, (probability > .5).astype(int)))
    table['native_custom'] = native
    table['f1_macro'] = observed
    table['custom'] = 1 - table['f1_macro']
    table['native_custom_missing'] = ~np.isfinite(native)
    table['native_custom_discrepancy'] = np.isfinite(native) & ~np.isclose(native, table['custom'], atol=1e-10, rtol=0)
    return table.sort_values('custom', ascending=True, kind='stable').reset_index(drop=True)


def flaml_metric(X_val, y_val, estimator, labels, X_train, y_train,
                 weight_val=None, weight_train=None, *args, **kwargs):
    # Primary objective is unweighted macro-F1, including in weighted scenarios.
    value = macro_f1(y_val, estimator.predict(X_val))
    return 1 - value, {'f1_macro': value}


def search(framework, arrays, settings, output):
    train, valid = arrays['X_train'], arrays['X_validation']
    y_train, y_valid = arrays['y_train'], arrays['y_validation']
    seed, seconds = settings['seed'], settings['native_budget_seconds']
    weighted = settings['scenario'] == 'class_weighted'
    cpus = settings['cpus']
    meta = {'selection_metric': 'f1_macro', 'refit_required': False,
            'search_data': 'training and validation only', 'native_budget_seconds': seconds}
    if framework == 'h2o':
        import h2o
        from h2o.automl import H2OAutoML
        from v2.h2o_metric import MacroF1Loss
        h2o.init(nthreads=cpus, max_mem_size='10G')
        try:
            reference = h2o.upload_custom_metric(MacroF1Loss, func_name='macro_f1_loss', func_file='h2o_macro_metric.py')
            names = [f'x{i}' for i in range(train.shape[1])]
            if not all(k in arrays for k in ['X_blending', 'y_blending']):
                raise ValueError('A separate training-only blending partition is required')
            train_df, valid_df = pd.DataFrame(train, columns=names), pd.DataFrame(valid, columns=names)
            blend_df = pd.DataFrame(arrays['X_blending'], columns=names)
            train_df['Class'], valid_df['Class'], blend_df['Class'] = y_train, y_valid, arrays['y_blending']
            if weighted:
                train_df['__weight'], valid_df['__weight'] = arrays['weights_train'], 1.0
                # The blending frame preserves natural prevalence; only base
                # learners receive class-balanced training weights.
                blend_df['__weight'] = 1.0
            train_h, valid_h, blend_h = h2o.H2OFrame(train_df), h2o.H2OFrame(valid_df), h2o.H2OFrame(blend_df)
            for frame in [train_h, valid_h, blend_h]:
                frame['Class'] = frame['Class'].asfactor()
            if train_h['Class'].levels()[0] != ['0', '1']:
                raise ValueError('H2O factor order must map Class=1 to p1')
            runtime = max(1, int(seconds - min(15, seconds * .2)))
            algorithms = ['GBM', 'DRF', 'XGBoost', 'GLM', 'DeepLearning', 'StackedEnsemble']
            automl = H2OAutoML(include_algos=algorithms, nfolds=0,
                               balance_classes=False, max_runtime_secs=runtime,
                               max_models=5 if settings['smoke'] else None,
                               sort_metric='custom', stopping_metric='custom',
                               custom_metric_func=reference, seed=seed)
            kwargs = {'weights_column': '__weight'} if weighted else {}
            automl.train(x=names, y='Class', training_frame=train_h,
                         blending_frame=blend_h, validation_frame=valid_h,
                         leaderboard_frame=valid_h, **kwargs)
            automl.event_log.as_data_frame().to_csv(output / 'h2o-events.csv', index=False)
            native_table = automl.leaderboard.as_data_frame()
            native_table.to_csv(output / 'native-leaderboard.csv', index=False)
            candidates = {}
            def validation_probability(candidate_id):
                candidate = h2o.get_model(candidate_id)
                candidates[candidate_id] = candidate
                predicted = candidate.predict(valid_h)
                try:
                    return predicted.as_data_frame()['p1'].to_numpy()
                finally:
                    h2o.remove(predicted)
            table = rank_h2o_candidates(native_table, y_valid, validation_probability)
            table.to_csv(output / 'validation-leaderboard.csv', index=False)
            model = candidates[table.iloc[0]['model_id']]
            actual_stopping = {}
            ensemble_paths = {}
            for candidate_id in table['model_id']:
                candidate = candidates[candidate_id]
                actual_stopping[candidate_id] = {
                    'algorithm': candidate.algo,
                    'stopping_metric': candidate.actual_params.get('stopping_metric'),
                    'custom_metric_func': candidate.actual_params.get('custom_metric_func'),
                }
                if candidate.algo.lower() == 'stackedensemble':
                    actual_stopping[candidate_id]['metalearner'] = candidate._model_json['output'].get('metalearner')
                    ensemble_paths[candidate_id] = h2o.save_model(candidate, path=str(output / 'h2o-ensembles'), force=True)
            if settings['require_stacked_candidate'] and not ensemble_paths:
                raise RuntimeError('No Stacked Ensemble was built within this run; see AutoML events')
            value = float(table.iloc[0]['f1_macro'])
            path = h2o.save_model(model, path=str(output / 'h2o-model'), force=True)
            meta.update(selected=model.model_id, validation_f1_macro=value,
                        native_objective='1 - f1_macro', sort_metric='custom', stopping_metric='custom',
                        custom_metric_func=reference, threshold=.5, include_algos=algorithms,
                        automl_runtime_budget_seconds=runtime,
                        actual_candidate_stopping=actual_stopping,
                        search_data='base training, training-only blending, and validation; no external test',
                        stacking_enabled=True, ensemble_count=len(ensemble_paths), ensemble_paths=ensemble_paths,
                        blending_rows=len(blend_df), blending_fraud_count=int(arrays['y_blending'].sum()),
                        base_training_rows=len(train_df), blending_class_weight_policy='uniform weights, natural prevalence',
                        scope='Five base-model families and Stacked Ensembles; group-disjoint training-only blending; GLM/metalearner may retain internal AUTO',
                        leaderboard_custom_loss_checked=True, independent_validation_ranking=True,
                        candidates_recomputed=len(table),
                        native_custom_missing_candidates=table.loc[table.native_custom_missing, 'model_id'].tolist(),
                        native_custom_discrepant_candidates=table.loc[table.native_custom_discrepancy, 'model_id'].tolist(),
                        ranking_policy='All candidates ranked by independently recomputed validation macro-F1; missing native metrics retained')
            return H2OModel(path, cpus, '__weight' if weighted else None), meta
        finally:
            h2o.cluster().shutdown(prompt=False)
    if framework == 'flaml':
        from flaml import AutoML
        automl = AutoML()
        kwargs = {'sample_weight': arrays['weights_train']} if weighted else {}
        automl.fit(X_train=train, y_train=y_train, X_val=valid, y_val=y_valid,
                   task='classification', metric=flaml_metric, eval_method='holdout',
                   time_budget=seconds, train_time_limit=min(seconds, 300),
                   retrain_full=False, sample=False, auto_augment=False, seed=seed, n_jobs=cpus,
                   estimator_list=['lgbm', 'rf', 'extra_tree', 'xgboost', 'lrl1'],
                   log_file_name=str(output / 'flaml-search.jsonl'), log_type='all', **kwargs)
        # Retain the actual selected FLAML wrapper. No manual reconstruction.
        meta.update(selected=automl.best_estimator, selected_config=automl.best_config,
                    validation_f1_macro=float(1 - automl.best_loss))
        return automl.model, meta

    if framework == 'autogluon':
        from autogluon.tabular import TabularPredictor
        from autogluon.features.generators import IdentityFeatureGenerator
        train_df, valid_df = pd.DataFrame(train), pd.DataFrame(valid)
        train_df['Class'], valid_df['Class'] = y_train, y_valid
        weight_args = {}
        if weighted:
            train_df['__weight'] = arrays['weights_train']
            valid_df['__weight'] = 1.0
            weight_args = {'sample_weight': '__weight', 'weight_evaluation': False}
        predictor = TabularPredictor(label='Class', eval_metric='f1_macro',
                                     path=str(output / 'autogluon'), **weight_args)
        # Inputs are already numeric and scaled on training only. Pre-fit the
        # identity generator to prevent AutoGluon fitting feature processing
        # on its internally concatenated training and validation frames.
        generator = IdentityFeatureGenerator()
        generator.fit_transform(pd.DataFrame(train))
        predictor.fit(train_data=train_df, tuning_data=valid_df, time_limit=seconds,
                      feature_generator=generator,
                      presets='medium_quality', num_cpus=cpus, num_gpus=0,
                      num_bag_folds=0, num_stack_levels=0, fit_weighted_ensemble=False,
                      hyperparameters={'GBM': {'seed': seed}, 'RF': {'random_state': seed},
                                       'XT': {'random_state': seed}, 'XGB': {'random_state': seed},
                                       'CAT': {'random_seed': seed}},
                      hyperparameter_tune_kwargs={'scheduler': 'local', 'searcher': 'random', 'num_trials': 1000},
                      calibrate_decision_threshold=False)
        native = predictor.leaderboard(silent=True)
        native.to_csv(output / 'native-leaderboard.csv', index=False)
        # In this installed image RF/XT cached scores can differ from fresh
        # predictions. Rank every fitted candidate again on the SAME validation
        # rows, within the search deadline, using its actual inference output.
        values = []
        for name in native.loc[native['can_infer'], 'model']:
            observed = macro_f1(y_valid, predictor.predict(pd.DataFrame(valid), model=name))
            values.append({'model': name, 'f1_macro': observed,
                           'native_score_val': float(native.set_index('model').loc[name, 'score_val'])})
        table = pd.DataFrame(values).sort_values('f1_macro', ascending=False, kind='stable')
        table.to_csv(output / 'validation-leaderboard.csv', index=False)
        name = table.iloc[0]['model']
        predictor.set_model_best(name, save_trainer=True)
        meta.update(selected=name, validation_f1_macro=float(table.iloc[0]['f1_macro']),
                    native_reported_validation_f1_macro=float(table.iloc[0]['native_score_val']),
                    independent_validation_ranking=True,
                    candidates_with_native_score_discrepancy=int((abs(table.f1_macro - table.native_score_val) > 1e-10).sum()),
                    search_space='GBM,RF,XT,XGB,CAT; identical across scenarios')
        return AutoGluonModel(predictor, name), meta

    if framework == 'pycaret':
        model, selection = pycaret_deadline_search(settings, output)
        meta.update(selection)
        return model, meta

    if framework == 'tpot':
        from tpot import TPOTClassifier
        from v2.weighting import tpot_weight_space, TPOT_MODELS
        from sklearn.metrics import make_scorer
        from sklearn.model_selection import PredefinedSplit
        from sklearn.base import clone
        # Disable nested estimators with private CV that cannot receive group splits.
        space = tpot_weight_space(weighted, seed, cpus, len(train), train.shape[1])
        cv = PredefinedSplit(np.r_[np.full(len(train), -1), np.zeros(len(valid), dtype=int)])
        import json
        from pathlib import Path
        profile = tpot_resource_profile(json.loads((Path(__file__).parent / 'config.json').read_text()))
        with tpot_dask_client(profile, output) as client:
            estimator = TPOTClassifier(
                search_space=space, scorers=[make_scorer(macro_f1)], scorers_weights=[1],
                cv=cv,
                max_time_mins=seconds * .75 / 60,
                max_eval_time_mins=(min(seconds, 60) if settings.get('mode') == 'qualification' else min(seconds * .1, 180)) / 60,
                n_jobs=profile['workers'], memory_limit=profile['worker_memory_bytes'],
                client=client, processes=True, scatter=False, random_state=seed, validation_strategy='none', preprocessing=False,
                population_size=8 if settings['smoke'] else 32,
                initial_population_size=4 if settings['smoke'] else 32,
                generations=1 if settings['smoke'] else None,
                allow_inner_classifiers=False, verbose=2,
            )
            estimator.fit(np.vstack([train, valid]), np.r_[y_train, y_valid])
        table = estimator.evaluated_individuals.rename(columns={'macro_f1': 'f1_macro'})
        table.drop(columns=['Individual', 'Instance'], errors='ignore').to_csv(output / 'validation-leaderboard.csv')
        if 'f1_macro' not in table:
            raise RuntimeError(f'TPOT macro-F1 column missing: {list(table.columns)}')
        candidates = table.dropna(subset=['f1_macro', 'Instance']).sort_values('f1_macro', ascending=False)
        if candidates.empty:
            raise RuntimeError('No TPOT pipeline evaluated successfully within the budget')
        row = candidates.iloc[0]
        model = clone(row['Instance'])
        meta.update(candidate_families=list(TPOT_MODELS), training_class_weights=weighted,
                    execution_resources=profile, scatter_input=False, container_cpu_allocation=cpus, container_memory_gib=16,
                    weighting_policy='BalancedClassifier computes weights on each fit target; preprocessing remains training-only and unweighted',
                    selected=str(model), validation_f1_macro=float(row['f1_macro']),
                    optimization_budget_seconds=seconds * .75,
                    max_pipeline_eval_seconds=min(seconds, 60) if settings.get('mode') == 'qualification' else min(seconds * .1, 180),
                    refit_required=True, native_fit_includes_unused_train_validation_refit=True,
                    search_space='linear-light with five weight-compatible final classifiers, no inner classifiers or private inner CV; identical families in all scenarios')
        return model, meta
    raise ValueError(f'Unsupported framework: {framework}')
