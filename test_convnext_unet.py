from __future__ import annotations

import json
import unittest
from pathlib import Path


IMPORT_ERROR: Exception | None = None
try:
    import torch
    from convnext_unet import ENCODER_STAGE_CHANNELS, ConvNeXtTinyUNet
    from run_experiment import build_model
except Exception as exc:  # dependency-aware: local planning environments may lack torch
    IMPORT_ERROR = exc


@unittest.skipIf(IMPORT_ERROR is not None, f"PyTorch/torchvision unavailable: {IMPORT_ERROR}")
class ConvNeXtTinyUNetTests(unittest.TestCase):
    def test_forward_preserves_square_and_non_square_shapes(self):
        model = ConvNeXtTinyUNet(num_classes=3, pretrained=False).eval()
        with torch.no_grad():
            for height, width in ((256, 256), (256, 320)):
                output = model(torch.randn(1, 3, height, width))
                self.assertEqual(tuple(output.shape), (1, 3, height, width))

    def test_encoder_stage_layout_is_stable(self):
        model = ConvNeXtTinyUNet(num_classes=3, pretrained=False).eval()
        with torch.no_grad():
            stages = model.forward_features(torch.randn(1, 3, 64, 96))
        self.assertEqual(tuple(stage.shape[1] for stage in stages), ENCODER_STAGE_CHANNELS)
        self.assertEqual([tuple(stage.shape[-2:]) for stage in stages], [(16, 24), (8, 12), (4, 6), (2, 3)])

    def test_gradients_reach_encoder_and_decoder(self):
        model = ConvNeXtTinyUNet(num_classes=3, pretrained=False, decoder_channels=(64, 32, 16))
        model(torch.randn(1, 3, 64, 64)).mean().backward()
        self.assertIsNotNone(model.encoder[0][0].weight.grad)
        self.assertIsNotNone(model.classifier.weight.grad)

    def test_checkpoint_round_trip_without_pretrained_download(self):
        options = {"variant": "tiny", "pretrained": False, "decoder_channels": [64, 32, 16], "decoder_norm": "groupnorm", "dropout": 0.0}
        original = build_model("convnext_tiny_unet", options)
        restored = build_model("convnext_tiny_unet", options)
        restored.load_state_dict(original.state_dict(), strict=True)
        self.assertEqual(tuple(restored.classifier.weight.shape), (3, 16, 1, 1))

    def test_shared_config_registers_architecture(self):
        config = json.loads((Path(__file__).resolve().parent / "configs.json").read_text())
        self.assertIn("convnext_tiny_unet", config["architectures"])
        self.assertEqual(config["architectures"]["convnext_tiny_unet"]["variant"], "tiny")


if __name__ == "__main__":
    unittest.main()
