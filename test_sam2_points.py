import unittest

import numpy as np

from sam2_points import sample_positive_points


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
