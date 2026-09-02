import unittest
import numpy as np

from sufh_rl.models import build_model
from sufh_rl.datasets.loveda import LoveDAMulticlassDataset


class TestReleaseModelVariants(unittest.TestCase):

    def test_vaihingen_delta_head(self):
        model = build_model(
            method="suf_hrl",
            num_classes=6,
            delta_hidden_dims=(128, 128),
        )
        self.assertEqual(tuple(model.delta_head[0].weight.shape), (128, 256, 3, 3))
        self.assertEqual(tuple(model.delta_head[3].weight.shape), (128, 128, 3, 3))
        self.assertEqual(tuple(model.delta_head[6].weight.shape), (1, 128, 1, 1))

    def test_loveda_delta_head(self):
        model = build_model(
            method="suf_hrl",
            num_classes=7,
            delta_hidden_dims=(128, 64),
        )
        self.assertEqual(tuple(model.delta_head[0].weight.shape), (128, 256, 3, 3))
        self.assertEqual(tuple(model.delta_head[3].weight.shape), (64, 128, 3, 3))
        self.assertEqual(tuple(model.delta_head[6].weight.shape), (1, 64, 1, 1))

    def test_loveda_full_image_eval(self):
        ds = LoveDAMulticlassDataset.__new__(LoveDAMulticlassDataset)
        ds.crop_size = None

        image = np.zeros((1024, 1024, 3), dtype=np.uint8)
        label = np.zeros((1024, 1024), dtype=np.uint8)

        out_image, out_label = ds._center_crop(image, label)

        self.assertEqual(out_image.shape, image.shape)
        self.assertEqual(out_label.shape, label.shape)


if __name__ == "__main__":
    unittest.main()
