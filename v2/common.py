"""Shared protocol. This module never imports an AutoML framework."""
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import train_test_split
from v2.protocol import capability


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str, allow_nan=False) + '\n')
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def macro_f1(y_true, y_pred, sample_weight=None):
    # Some library scorers pass sample_weight=None automatically. Evaluation
    # remains unweighted even when training uses balanced class weights.
    return float(f1_score(y_true, y_pred, labels=[0, 1], average='macro', zero_division=0))


def grouped_partitions(y, groups, seed, test_fraction=.2, validation_fraction=.2):
    y, groups = np.asarray(y), np.asarray(groups)
    unique, first, inverse = np.unique(groups, return_index=True, return_inverse=True)
    labels = y[first]
    if not np.array_equal(labels[inverse], y):
        raise ValueError('Identical feature rows have conflicting labels; resolve dataset provenance before splitting')
    group_ids = np.arange(len(unique))
    development, test = train_test_split(group_ids, test_size=test_fraction,
                                        random_state=seed, stratify=labels)
    train, validation = train_test_split(development, test_size=validation_fraction,
                                       random_state=seed + 10000, stratify=labels[development])
    parts = {name: np.flatnonzero(np.isin(inverse, indices))
             for name, indices in [('train', train), ('validation', validation), ('test', test)]}
    for name, indices in parts.items():
        if set(np.unique(y[indices])) != {0, 1}:
            raise ValueError(f'{name} does not contain both classes')
    for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
        if np.intersect1d(groups[parts[a]], groups[parts[b]]).size:
            raise AssertionError(f'Duplicate group crossed {a}/{b}')
        if np.intersect1d(parts[a], parts[b]).size:
            raise AssertionError('Row index overlap')
    if sum(map(len, parts.values())) != len(y):
        raise AssertionError('Partitions do not cover the dataset')
    return parts


def grouped_blending(y, groups, training_indices, seed, fraction=.2):
    """Carve blending exclusively from the original training partition."""
    if not 0 < fraction < 1:
        raise ValueError('Blending fraction must be strictly between zero and one')
    y, groups, training_indices = np.asarray(y), np.asarray(groups), np.asarray(training_indices)
    if training_indices.ndim != 1 or len(np.unique(training_indices)) != len(training_indices):
        raise ValueError('Training indices must be unique and one-dimensional')
    if np.any(training_indices < 0) or np.any(training_indices >= len(y)):
        raise ValueError('Training index out of range')
    unique, first, inverse = np.unique(groups[training_indices], return_index=True, return_inverse=True)
    labels = y[training_indices[first]]
    if not np.array_equal(labels[inverse], y[training_indices]):
        raise ValueError('Identical feature rows have conflicting labels')
    base_groups, blending_groups = train_test_split(unique, test_size=fraction,
                                                   random_state=seed + 20000, stratify=labels)
    base = training_indices[np.isin(groups[training_indices], base_groups)]
    blending = training_indices[np.isin(groups[training_indices], blending_groups)]
    if np.intersect1d(groups[base], groups[blending]).size:
        raise AssertionError('Duplicate group crossed base training/blending')
    if not np.array_equal(np.sort(np.r_[base, blending]), np.sort(training_indices)):
        raise AssertionError('Base training and blending do not cover original training')
    if any(set(np.unique(y[indices])) != {0, 1} for indices in [base, blending]):
        raise ValueError('Base training and blending must each contain both classes')
    return base, blending


def metrics(y, prediction, score=None):
    class_f1 = f1_score(y, prediction, labels=[0, 1], average=None, zero_division=0)
    result = {'f1_macro': macro_f1(y, prediction), 'f1_class_0': float(class_f1[0]),
              'f1_class_1': float(class_f1[1]), 'precision_class_1': float(precision_score(y, prediction, zero_division=0)),
              'recall_class_1': float(recall_score(y, prediction, zero_division=0)),
              'accuracy': float(accuracy_score(y, prediction)),
              'confusion_matrix_labels_0_1': confusion_matrix(y, prediction, labels=[0, 1]).tolist(),
              'average_precision_class_1': None}
    if score is not None:
        if not np.isfinite(score).all():
            raise ValueError('Non-finite prediction scores')
        result['average_precision_class_1'] = float(average_precision_score(y, score))
    return result


def predict(model, X):
    prediction = np.asarray(model.predict(X)).reshape(-1)
    if len(prediction) != len(X) or not np.isin(prediction, [0, 1]).all():
        raise ValueError('Predictions must contain exactly one binary label per row')
    prediction = prediction.astype(int)
    if hasattr(model, 'predict_proba'):
        try:
            probability = model.predict_proba(X)
            classes = list(model.classes_)
            probability = np.asarray(probability)
            if probability.shape != (len(X), 2) or set(classes) != {0, 1}:
                raise ValueError('Binary probability output has invalid shape or class labels')
            if not np.isfinite(probability).all() or (probability < 0).any() or (probability > 1).any():
                raise ValueError('Invalid predicted probabilities')
            if not np.allclose(probability.sum(axis=1), 1, atol=1e-6):
                raise ValueError('Class probabilities do not sum to one')
            score = probability[:, classes.index(1)]
            return prediction, score, 'probability_class_1'
        except (AttributeError, NotImplementedError):
            pass
    if hasattr(model, 'decision_function'):
        try:
            score = np.asarray(model.decision_function(X)).reshape(-1)
            if len(score) != len(X) or not np.isfinite(score).all():
                raise ValueError('Invalid decision scores')
            if list(model.classes_).index(1) == 0:
                score = -score
            return prediction, score, 'decision_function_class_1'
        except (AttributeError, NotImplementedError):
            pass
    return prediction, None, 'unavailable; AP is not computed from hard labels'


def resource_snapshot():
    # cgroup measurements include child processes, unlike Python-process RSS.
    result = {}
    for filename in ['memory.current', 'memory.peak', 'memory.max', 'cpu.stat']:
        path = Path('/sys/fs/cgroup') / filename
        result[filename] = path.read_text().strip() if path.exists() else None
    return result


def supported(framework, scenario):
    return capability(framework, scenario)
