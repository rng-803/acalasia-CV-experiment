import unittest

import numpy as np

from sam2_points import compose_semantic_prediction, sample_class_prompts, sample_positive_points


class Sam2PointPromptTests(unittest.TestCase):
    def test_eroded_random_points_are_foreground_and_one_per_label(self):
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[2:9, 2:9] = 1
        mask[11:18, 11:18] = 2
        target, points, labels = sample_positive_points(mask, seed=3)
        self.assertEqual(points.shape, (2, 2))
        self.assertEqual(labels.tolist(), [1, 1])
        self.assertTrue(all(target[int(y), int(x)] == 1 for x, y in points))

    def test_empty_foreground_is_rejected(self):
        with self.assertRaises(ValueError):
            sample_positive_points(np.zeros((8, 8), dtype=np.uint8))

    def test_class_prompts_use_class_specific_targets_and_points(self):
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[2:9, 2:9] = 1
        mask[11:18, 11:18] = 2
        targets, points, point_labels, class_ids = sample_class_prompts(mask, seed=4)
        self.assertEqual(class_ids.tolist(), [1, 2])
        self.assertEqual(targets.shape, (2, 20, 20))
        self.assertEqual(point_labels.tolist(), [1, 1])
        for index, (x, y) in enumerate(points.astype(int)):
            self.assertEqual(mask[y, x], class_ids[index])
            self.assertEqual(targets[index, y, x], 1)
            self.assertEqual(targets[1 - index, y, x], 0)

    def test_semantic_composition_thresholds_and_resolves_overlap(self):
        probabilities = np.asarray([
            [[0.9, 0.4], [0.8, 0.7]],
            [[0.6, 0.3], [0.95, 0.7]],
        ], dtype=np.float32)
        prediction = compose_semantic_prediction(probabilities, np.asarray([1, 2]), threshold=0.5)
        self.assertEqual(prediction.tolist(), [[1, 0], [2, 1]])
