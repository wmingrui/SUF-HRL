import unittest


class TestPublicMetricsImport(unittest.TestCase):
    def test_public_metrics_import(self):
        from sufh_rl.metrics import (
            compute_miou_macc_oa,
            confusion_matrix,
            update_hist_from_logits,
        )

        self.assertTrue(callable(compute_miou_macc_oa))
        self.assertTrue(callable(confusion_matrix))
        self.assertTrue(callable(update_hist_from_logits))


if __name__ == "__main__":
    unittest.main()
