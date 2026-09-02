import unittest
from pathlib import Path
import yaml


ROOT = Path(__file__).resolve().parents[1]


class TestPublicTrainingContract(unittest.TestCase):

    def test_train_does_not_load_custom_checkpoint(self):
        src = (ROOT / "tools/train.py").read_text()

        self.assertNotIn("init_checkpoint", src)
        self.assertNotIn("torch.load(", src)
        self.assertNotIn("load_state_dict(", src)

    def test_public_configs_do_not_expose_pretrained_switch(self):
        for name in ["potsdam", "vaihingen", "loveda"]:
            cfg = yaml.safe_load((ROOT / f"configs/{name}.yaml").read_text())

            self.assertNotIn("init_checkpoint", cfg)
            self.assertNotIn("pretrained", cfg["model"])

    def test_training_does_not_forward_pretrained_flag(self):
        src = (ROOT / "tools/train.py").read_text()
        self.assertNotIn(
            'pretrained=bool(cfg["model"].get("pretrained", True))',
            src,
        )

    def test_segformer_backbone_always_uses_official_pretrained_weights(self):
        src = (
            ROOT / "sufh_rl/models/segformer_suf_hrl.py"
        ).read_text()

        self.assertIn(
            "self.backbone = SegformerModel.from_pretrained(backbone_name)",
            src,
        )

        self.assertNotIn(
            "if pretrained else SegformerModel.from_pretrained",
            src,
        )


if __name__ == "__main__":
    unittest.main()
