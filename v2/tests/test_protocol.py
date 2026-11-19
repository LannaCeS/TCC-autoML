import unittest
import numpy as np
from v2.common import grouped_partitions, grouped_blending, macro_f1, metrics, predict, supported


class ProtocolTests(unittest.TestCase):
    def test_group_separation_and_reproducibility(self):
        groups = np.repeat(np.arange(300), 3)
        y = np.repeat(np.r_[np.zeros(240), np.ones(60)], 3)
        parts = grouped_partitions(y, groups, 42)
        again = grouped_partitions(y, groups, 42)
        for name, indices in parts.items():
            np.testing.assert_array_equal(indices, again[name])
            self.assertEqual(set(y[indices]), {0, 1})
        for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
            self.assertFalse(set(groups[parts[a]]) & set(groups[parts[b]]))
        self.assertEqual(len(set(np.concatenate(list(parts.values())))), len(y))

    def test_macro_is_not_positive_class_f1_or_weighted_f1(self):
        y, pred = [0, 0, 0, 1], [0, 0, 0, 0]
        value = metrics(y, pred)
        self.assertAlmostEqual(value['f1_class_0'], 6 / 7)
        self.assertEqual(value['f1_class_1'], 0)
        self.assertAlmostEqual(macro_f1(y, pred), 3 / 7)
        self.assertAlmostEqual(macro_f1(y, pred, sample_weight=[100, 100, 100, 1]), 3 / 7)
        self.assertIsNone(value['average_precision_class_1'])

    def test_blending_uses_only_training_and_keeps_duplicate_groups_intact(self):
        groups = np.repeat(np.arange(300), 3)
        y = np.repeat(np.r_[np.zeros(240), np.ones(60)], 3)
        parts = grouped_partitions(y, groups, 42)
        base, blend = grouped_blending(y, groups, parts['train'], 42)
        np.testing.assert_array_equal(np.sort(np.r_[base, blend]), parts['train'])
        self.assertEqual(set(y[base]), {0, 1})
        self.assertEqual(set(y[blend]), {0, 1})
        four = [base, blend, parts['validation'], parts['test']]
        for i, left in enumerate(four):
            for right in four[i+1:]:
                self.assertFalse(set(groups[left]) & set(groups[right]))
        np.testing.assert_array_equal(base, grouped_blending(y, groups, parts['train'], 42)[0])
        with self.assertRaisesRegex(ValueError, 'fraction'):
            grouped_blending(y, groups, parts['train'], 42, fraction=0)

    def test_conflicting_labels_are_not_silently_split_or_relabelled(self):
        with self.assertRaisesRegex(ValueError, 'conflicting labels'):
            grouped_partitions([0, 1, 0, 1], [10, 10, 20, 30], 42)

    def test_positive_probability_column_follows_class_order(self):
        class Model:
            classes_ = [1, 0]
            def predict(self, X): return np.array([1, 0])
            def predict_proba(self, X): return np.array([[.9, .1], [.2, .8]])
        _, score, _ = predict(Model(), np.zeros((2, 1)))
        np.testing.assert_array_equal(score, [.9, .2])

    def test_unsupported_scenarios_are_not_silently_replaced(self):
        self.assertTrue(supported('h2o', 'baseline')[0])
        self.assertTrue(supported('h2o', 'class_weighted')[0])
        self.assertTrue(supported('tpot', 'class_weighted')[0])
        self.assertTrue(supported('pycaret', 'class_weighted')[0])
        self.assertTrue(supported('flaml', 'class_weighted')[0])

    def test_h2o_custom_metric_aggregates_counts_before_computing_macro(self):
        from v2.h2o_metric import MacroF1Loss
        metric = MacroF1Loss()
        left = metric.map([0, .8, .2], [0], 100, 0, None)
        right = metric.map([0, .8, .2], [1], 1, 0, None)
        aggregate = metric.reduce(metric.reduce(left, left), metric.reduce(left, right))
        self.assertAlmostEqual(metric.metric(aggregate), 1 - 3 / 7)


if __name__ == '__main__':
    unittest.main()
