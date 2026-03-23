import torch


try:
    import clip
    from clip.model import ModifiedResNet
except ImportError:
    clip = None
    ModifiedResNet = object


class CLIPSemanticExtractor(ModifiedResNet):
    """Official CLIP_Semantic_extractor transplant with minimal adaptation."""

    def __init__(self, layers=(3, 4, 6, 3), pretrained=True, path=None, output_dim=1024, heads=32):
        if clip is None:
            raise ImportError(
                "vision_semantic_extractor.py requires the OpenAI CLIP package. "
                "Install with `pip install git+https://github.com/openai/CLIP.git`."
            )
        super().__init__(layers=layers, output_dim=output_dim, heads=heads)

        ckpt = 'RN50' if path is None else path
        model = None
        if pretrained:
            model, _ = clip.load(ckpt, device='cpu')
            self.load_state_dict(model.visual.state_dict())
        self.register_buffer('mean', torch.Tensor([0.48145466, 0.4578275, 0.40821073]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.Tensor([0.26862954, 0.26130258, 0.27577711]).view(1, 3, 1, 1))
        self.requires_grad_(False)
        if model is not None:
            del model

    def forward(self, x):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        x = (x - self.mean) / self.std
        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x
