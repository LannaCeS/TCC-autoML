"""Additive custom H2O metric; keep class syntax compatible with Jython."""


class MacroF1Loss:
    def map(self, pred, act, w, o, model):
        # Use p(class=1) and the shared fixed threshold. Evaluation is unweighted.
        positive = pred[2] > 0.5
        actual = act[0] == 1
        return [1.0 if positive and actual else 0.0,
                1.0 if not positive and not actual else 0.0,
                1.0 if positive and not actual else 0.0,
                1.0 if not positive and actual else 0.0]

    def reduce(self, left, right):
        return [left[i] + right[i] for i in range(4)]

    def metric(self, values):
        tp, tn, fp, fn = values
        d1, d0 = 2 * tp + fp + fn, 2 * tn + fp + fn
        f1 = 2 * tp / d1 if d1 else 0.0
        f0 = 2 * tn / d0 if d0 else 0.0
        return 1.0 - (f1 + f0) / 2.0
