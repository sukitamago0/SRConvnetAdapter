import torch
import torch.nn as nn


class CLIPSemanticExtractor(nn.Module):
    """Minimal wrapper around the official SeD CLIP RN50 visual trunk."""

    def __init__(self, layers=(3, 4, 6, 3), pretrained=True, path=None, output_dim=1024, heads=32):
        super().__init__()
        try:
            import clip
            from clip.model import ModifiedResNet
        except ImportError as exc:
            raise ImportError(
                "vision_semantic_extractor.py requires the OpenAI CLIP package. "
                "Install with `pip install git+https://github.com/openai/CLIP.git`."
            ) from exc

        class _Extractor(ModifiedResNet):
            def __init__(self):
                super().__init__(layers=layers, output_dim=output_dim, heads=heads)

        self.backbone = _Extractor()
        ckpt = "RN50" if path is None else path
        model = None
        if pretrained:
            model, _ = clip.load(ckpt, device="cpu")
            self.backbone.load_state_dict(model.visual.state_dict())

        self.register_buffer("mean", torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1))
        self.requires_grad_(False)
        if model is not None:
            del model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bb = self.backbone

        def stem(inp):
            inp = bb.relu1(bb.bn1(bb.conv1(inp)))
            inp = bb.relu2(bb.bn2(bb.conv2(inp)))
            inp = bb.relu3(bb.bn3(bb.conv3(inp)))
            inp = bb.avgpool(inp)
            return inp

        x = ((x - self.mean) / self.std).type(bb.conv1.weight.dtype)
        x = stem(x)
        x = bb.layer1(x)
        x = bb.layer2(x)
        x = bb.layer3(x)
        return x
