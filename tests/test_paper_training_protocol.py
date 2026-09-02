import importlib.util
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


class TestPaperTrainingProtocol(unittest.TestCase):

    def test_public_configs_match_paper_protocol(self):
        for dataset in ["potsdam", "vaihingen", "loveda"]:
            with self.subTest(dataset=dataset):
                cfg_path = ROOT / "configs" / f"{dataset}.yaml"
                cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

                self.assertEqual(cfg["dataset"]["crop_size"], 512)

                t = cfg["training"]

                self.assertEqual(t["batch_size"], 16)
                self.assertAlmostEqual(float(t["lr"]), 6e-5)
                self.assertAlmostEqual(float(t["weight_decay"]), 0.01)

                self.assertEqual(t["max_iters"], 80000)
                self.assertEqual(t["warmup_iters"], 1500)
                self.assertEqual(t["lr_scheduler"], "poly")

                self.assertNotIn("epochs", t)
                self.assertNotIn("topk_warmup_epochs", t)

    def test_training_is_iteration_based(self):
        text = (ROOT / "tools" / "train.py").read_text(encoding="utf-8")

        self.assertIn("max_iters", text)
        self.assertIn("global_iter", text)
        self.assertIn("warmup_iters", text)
        self.assertIn("topk_start_iter", text)

        self.assertNotIn("for epoch in range(epochs)", text)
        self.assertNotIn("epoch >= warmup_epochs", text)

    def test_poly_lr_function(self):
        train_path = ROOT / "tools" / "train.py"

        spec = importlib.util.spec_from_file_location(
            "sufhrl_public_train",
            train_path,
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        base_lr = 6e-5

        lr0 = module.compute_poly_lr(
            base_lr=base_lr,
            current_iter=0,
            max_iters=80000,
            warmup_iters=1500,
            warmup_start_factor=1e-6,
            power=1.0,
        )

        lr1500 = module.compute_poly_lr(
            base_lr=base_lr,
            current_iter=1500,
            max_iters=80000,
            warmup_iters=1500,
            warmup_start_factor=1e-6,
            power=1.0,
        )

        lr40000 = module.compute_poly_lr(
            base_lr=base_lr,
            current_iter=40000,
            max_iters=80000,
            warmup_iters=1500,
            warmup_start_factor=1e-6,
            power=1.0,
        )

        self.assertGreater(lr0, 0.0)
        self.assertLess(lr0, base_lr)

        self.assertAlmostEqual(lr1500, base_lr)

        self.assertGreater(lr40000, 0.0)
        self.assertLess(lr40000, base_lr)

    def test_training_checkpoint_is_public_eval_compatible(self):
        text = (ROOT / "tools" / "train.py").read_text(encoding="utf-8")

        self.assertIn(
            'torch.save(model.state_dict(), out_dir / "checkpoints" / "best.pth")',
            text,
        )


if __name__ == "__main__":
    unittest.main()
