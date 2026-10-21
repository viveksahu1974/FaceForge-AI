"""Pretrained PyTorch ViT age *range* classifier, not chronological age determination."""
from __future__ import annotations
from PIL import Image
from .core import rgb_image


class AgeEstimator:
    def __init__(self, device: str | None = None):
        import torch
        from transformers import AutoImageProcessor, AutoModelForImageClassification
        self.torch = torch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        name = "nateraw/vit-age-classifier"
        self.processor = AutoImageProcessor.from_pretrained(name)
        self.model = AutoModelForImageClassification.from_pretrained(name).to(self.device).eval()

    def predict(self, img: Image.Image | str, box: list[float] | None = None) -> tuple[str, float]:
        picture = rgb_image(img)
        if box is not None:
            x1, y1, x2, y2 = [int(v) for v in box]
            picture = picture.crop((max(0, x1), max(0, y1), min(picture.width, x2), min(picture.height, y2)))
        if picture.width < 8 or picture.height < 8:
            raise ValueError("Face crop is too small")
        batch = self.processor(images=picture, return_tensors="pt")
        batch = {k: v.to(self.device) for k, v in batch.items()}
        with self.torch.inference_mode():
            probabilities = self.model(**batch).logits.softmax(dim=-1)[0]
        i = int(probabilities.argmax().item())
        return self.model.config.id2label[i], float(probabilities[i].item())
