import tempfile
from pathlib import Path
import unittest
import joblib
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import PredefinedSplit, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.metrics import make_scorer
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from v2.common import macro_f1
from v2.weighting import BalancedClassifier, balanced_weights, assert_weight_support, weight_audits


class WeightRecorder(ClassifierMixin, BaseEstimator):
    def fit(self, X, y, sample_weight=None):
        self.weights_ = np.asarray(sample_weight).copy()
        self.target_ = np.asarray(y).copy()
        self.classes_ = np.array([0, 1])
        return self
    def predict(self, X):
        return np.zeros(len(X), dtype=int)


class WeightingTests(unittest.TestCase):
    def test_exact_training_formula_and_equal_class_mass(self):
        y = np.r_[np.zeros(90, dtype=int), np.ones(10, dtype=int)]
        w = balanced_weights(y)
        np.testing.assert_allclose(w[y == 0], 100/180)
        np.testing.assert_allclose(w[y == 1], 5)
        self.assertAlmostEqual(w[y == 0].sum(), 50)
        self.assertAlmostEqual(w[y == 1].sum(), 50)
        with self.assertRaises(ValueError): balanced_weights([0, 0])

    def test_predefined_split_weights_use_training_only(self):
        y = np.r_[np.zeros(90, dtype=int), np.ones(10, dtype=int), np.zeros(1, dtype=int), np.ones(99, dtype=int)]
        X = np.arange(200).reshape(-1, 1)
        model = make_pipeline(BalancedClassifier(WeightRecorder(), weighted=True))
        cv = PredefinedSplit(np.r_[np.full(100, -1), np.zeros(100)])
        result = cross_validate(model, X, y, cv=cv, scoring=make_scorer(macro_f1), return_estimator=True, error_score='raise')
        fitted = result['estimator'][0].steps[-1][1]
        np.testing.assert_array_equal(fitted.estimator_.target_, y[:100])
        np.testing.assert_array_equal(fitted.estimator_.weights_, balanced_weights(y[:100]))
        self.assertEqual(weight_audits(result['estimator'][0])[0]['class_counts'], [90, 10])

    def test_real_weights_change_prior_without_resampling(self):
        X = np.arange(200).reshape(100, 2)
        y = np.r_[np.zeros(90, dtype=int), np.ones(10, dtype=int)]
        weighted = BalancedClassifier(GaussianNB(), weighted=True).fit(X, y)
        baseline = BalancedClassifier(GaussianNB(), weighted=False).fit(X, y)
        np.testing.assert_allclose(weighted.estimator_.class_prior_, [.5, .5])
        np.testing.assert_allclose(baseline.estimator_.class_prior_, [.9, .1])
        self.assertEqual(weighted.fit_weight_audit_['training_rows'], 100)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'model.joblib';joblib.dump(weighted, path)
            loaded = joblib.load(path)
            np.testing.assert_array_equal(loaded.predict(X), weighted.predict(X))
            self.assertEqual(loaded.fit_weight_audit_, weighted.fit_weight_audit_)
        self.assertTrue(clone(weighted).weighted)

    def test_unsupported_and_duplicate_class_weighting_are_rejected(self):
        with self.assertRaisesRegex(TypeError, 'sample_weight'):
            assert_weight_support(KNeighborsClassifier())
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            assert_weight_support(RandomForestClassifier(class_weight='balanced'))


if __name__ == '__main__':
    unittest.main()
