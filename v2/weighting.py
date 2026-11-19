"""Genuine training-only class weights and compatible TPOT search spaces."""
import inspect
from functools import partial
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.metaestimators import available_if

PYCARET_MODELS = ('lr', 'dt', 'rf', 'et', 'gbc', 'ada', 'nb', 'lightgbm')
TPOT_MODELS = ('BernoulliNB', 'DecisionTreeClassifier', 'GaussianNB', 'LogisticRegression', 'MultinomialNB')
INTERNAL_WEIGHT_PARAMETERS = {'class_weight', 'scale_pos_weight', 'is_unbalance'}


def balanced_weights(y):
    y = np.asarray(y)
    if y.ndim != 1 or not np.isin(y, [0, 1]).all():
        raise ValueError('Weights require a binary one-dimensional training target')
    counts = np.bincount(y.astype(int), minlength=2)
    if min(counts) == 0:
        raise ValueError('Both training classes are required for balanced weights')
    return len(y) / (2 * counts[y.astype(int)])


def assert_weight_support(estimator):
    if 'sample_weight' not in inspect.signature(estimator.fit).parameters:
        raise TypeError(f'{type(estimator).__name__} does not explicitly support sample_weight')
    params = estimator.get_params(deep=False)
    if params.get('class_weight') not in (None, {}):
        raise ValueError('Internal class_weight would duplicate the external class weighting')
    if params.get('scale_pos_weight', 1) != 1 or params.get('is_unbalance', False):
        raise ValueError('Internal boosting class balance would duplicate the external class weighting')


def fit_with_weights(estimator, X, y, weighted):
    assert_weight_support(estimator)
    weights = balanced_weights(y) if weighted else None
    kwargs = {'sample_weight': weights} if weighted else {}
    estimator.fit(X, y, **kwargs)
    counts = np.bincount(np.asarray(y, dtype=int), minlength=2)
    audit = dict(applied=bool(weighted), training_rows=len(y), class_counts=counts.tolist(),
                 estimator=type(estimator).__name__, formula='n_train / (2 * n_train_class)',
                 class_weight_mass=[float(weights[np.asarray(y) == c].sum()) for c in [0, 1]] if weighted else None)
    return audit


def _has(method):
    return lambda self: hasattr(self.estimator, method)


class BalancedClassifier(ClassifierMixin, BaseEstimator):
    """Compute weights from this fit's target; validation rows never enter it."""
    def __init__(self, estimator, weighted=False):
        self.estimator = estimator
        self.weighted = weighted

    def fit(self, X, y):
        self.estimator_ = clone(self.estimator)
        self.fit_weight_audit_ = fit_with_weights(self.estimator_, X, y, self.weighted)
        self.classes_ = self.estimator_.classes_
        if hasattr(self.estimator_, 'n_features_in_'):
            self.n_features_in_ = self.estimator_.n_features_in_
        return self

    def predict(self, X):
        return self.estimator_.predict(X)

    @available_if(_has('predict_proba'))
    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)

    @available_if(_has('decision_function'))
    def decision_function(self, X):
        return self.estimator_.decision_function(X)


def neutral_tpot_parameters(configuration, parser):
    params = parser(configuration) if parser is not None else dict(configuration)
    # TPOT's native LogisticRegression configuration fixes class_weight to
    # balanced. Override this constant in BOTH scenarios to prevent double
    # balancing when sample_weight is applied and preserve a true baseline.
    if 'class_weight' in params:
        params['class_weight'] = None
    if 'scale_pos_weight' in params:
        params['scale_pos_weight'] = 1.0
    if 'is_unbalance' in params:
        params['is_unbalance'] = False
    return params


def tpot_classifier_node(name, **kwargs):
    from tpot.config.get_configspace import get_search_space
    if name not in TPOT_MODELS:
        raise ValueError('Classifier is outside the weight-compatible TPOT space')
    node = get_search_space(name, **kwargs)
    node.hyperparameter_parser = partial(neutral_tpot_parameters, parser=node.hyperparameter_parser)
    return node


def tpot_weight_space(weighted, seed, cpus, n_samples, n_features):
    from tpot.config.template_search_spaces import get_template_search_spaces
    from tpot.config.get_configspace import get_search_space
    from tpot.search_spaces.pipelines import WrapperPipeline, ChoicePipeline
    kwargs = dict(random_state=seed, n_jobs=cpus, n_samples=n_samples, n_features=n_features, n_classes=2)
    space = get_template_search_spaces('linear-light', classification=True, inner_predictors=False,
                                      cross_val_predict_cv=0, **kwargs)
    classifiers = ChoicePipeline([tpot_classifier_node(name, **kwargs) for name in TPOT_MODELS])
    space.search_spaces[-1] = WrapperPipeline(BalancedClassifier, {'weighted': weighted},
                                            classifiers, wrapped_param_name='estimator')
    return space


def pycaret_tuning_grid(exp, selected):
    if not isinstance(selected, BalancedClassifier):
        raise TypeError("PyCaret must return a weight-aware classifier")
    for name in PYCARET_MODELS:
        container = exp._all_models_internal[name]
        if type(selected.estimator) is container.class_def:
            return {'estimator__' + key: value for key, value in container.tune_distribution.items()
                    if key not in INTERNAL_WEIGHT_PARAMETERS}
    raise TypeError(f'Unexpected PyCaret candidate: {type(selected).__name__}')


def register_pycaret_weighted_model(exp, selected):
    # PyCaret 3.3.2 silently discards tune_model keyword arguments (including
    # timeout) for unknown estimators. Register AFTER comparison so distinct
    # custom candidates keep their identities during comparison.
    from pycaret.containers.models.classification import ClassifierContainer
    identifier = 'balanced_classifier'
    exp._all_models_internal[identifier] = ClassifierContainer(
        id=identifier, name='Training-weight-aware classifier', class_def=BalancedClassifier,
        args={'estimator': clone(selected.estimator), 'weighted': selected.weighted},
        tune_distribution=pycaret_tuning_grid(exp, selected), is_gpu_enabled=False,
        is_boosting_supported=False, is_soft_voting_supported=True)
    if exp._get_model_id(selected) != identifier:
        raise RuntimeError('PyCaret did not register the classifier; timeout would be ignored')


def weight_audits(model):
    if hasattr(model, 'fit_weight_audit_'):
        return [model.fit_weight_audit_]
    if hasattr(model, 'steps'):
        return [audit for _, step in model.steps for audit in weight_audits(step)]
    if hasattr(model, 'model'):
        return weight_audits(model.model)
    return []
