import unittest

from data import Sample, balanced_weights, patient_cv_folds, patient_counts, split
from run_experiment import AGGREGATE_METRIC_NAMES, aggregate_metric_rows


class ExperimentSplitTests(unittest.TestCase):
    def setUp(self):
        self.samples = [Sample(patient, f"{patient}/{i}.png", f"{patient}/{i}.png") for patient, count in (("p1", 3), ("p2", 1), ("p3", 2)) for i in range(count)]

    def test_test_patient_is_excluded_from_training(self):
        train, test = split(self.samples, "p3")
        self.assertEqual(patient_counts(train), {"p1": 3, "p2": 1})
        self.assertEqual({s.patient for s in test}, {"p3"})

    def test_cv_is_patient_disjoint(self):
        for fold in patient_cv_folds(self.samples, seed=7):
            self.assertTrue(set(fold["train_patients"]).isdisjoint(fold["val_patients"]))

    def test_balancing_equalizes_patient_total_weight(self):
        weights = balanced_weights(self.samples)
        self.assertAlmostEqual(weights[:3].sum(), weights[3:4].sum())
        self.assertAlmostEqual(weights[3:4].sum(), weights[4:].sum())

    def test_aggregate_metric_schema_is_shared_across_architectures(self):
        fold_rows = []
        for fold, offset in enumerate((0.0, 0.1)):
            row = {metric: 0.5 + offset for metric in AGGREGATE_METRIC_NAMES}
            row["fold"] = fold
            fold_rows.append(row)
        aggregate = aggregate_metric_rows(fold_rows)
        self.assertEqual([row["metric"] for row in aggregate], list(AGGREGATE_METRIC_NAMES))
        self.assertTrue(all(row["folds"] == 2 for row in aggregate))


if __name__ == "__main__":
    unittest.main()
