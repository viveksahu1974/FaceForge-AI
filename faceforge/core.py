"""PyTorch MTCNN detection, 5-point alignment, and FaceNet embeddings."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import cv2
import numpy as np
from PIL import Image, ImageDraw

@dataclass
class Face:
    box: np.ndarray
    landmarks: np.ndarray
    confidence: float

    @property
    def area(self) -> float:
        return float(max(0, self.box[2] - self.box[0]) * max(0, self.box[3] - self.box[1]))

def rgb_image(value: Image.Image | np.ndarray | str) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, str):
        with Image.open(value) as img:
            return img.convert("RGB")
    arr = np.asarray(value)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("Expected an RGB HxWx3 image")
    return Image.fromarray(arr.astype(np.uint8), "RGB")

class FaceEngine:
    def __init__(self, device: Optional[str] = None):
        import torch
        from facenet_pytorch import MTCNN, InceptionResnetV1
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.mtcnn = MTCNN(keep_all=True, image_size=160, margin=0, device=self.device)
        self.encoder = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)

    def detect(self, img: Image.Image | np.ndarray | str, minimum_confidence: float = 0.90) -> list[Face]:
        boxes, probs, points = self.mtcnn.detect(rgb_image(img), landmarks=True)
        if boxes is None or points is None:
            return []
        faces = []
        for box, prob, landmarks in zip(boxes, probs, points):
            if prob is not None and np.isfinite(prob) and prob >= minimum_confidence:
                faces.append(Face(np.asarray(box, dtype=np.float32), np.asarray(landmarks, dtype=np.float32), float(prob)))
        return sorted(faces, key=lambda f: f.area, reverse=True)

    def embed(self, img: Image.Image | np.ndarray | str, index: int = 0) -> np.ndarray:
        import torch
        picture = rgb_image(img)
        faces = self.detect(picture)
        if index < 0 or index >= len(faces):
            raise ValueError("No face at this index (check that the face is clearly visible).")
        crop = self.mtcnn.extract(picture, np.asarray([faces[index].box]), None)
        if crop is None:
            raise ValueError("Could not crop detected face")
        if crop.ndim == 3:
            crop = crop.unsqueeze(0)
        with torch.inference_mode():
            vector = self.encoder(crop.to(self.device))[0].cpu().numpy().astype(np.float32)
        return vector / max(np.linalg.norm(vector), 1e-10)

    def align(self, img: Image.Image | np.ndarray | str, index: int = 0, size: int = 160) -> Image.Image:
        picture = np.asarray(rgb_image(img))
        faces = self.detect(Image.fromarray(picture))
        if not faces or index >= len(faces) or index < 0:
            raise ValueError("No face available for alignment")
        template = np.asarray([
            [38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
            [41.5493, 92.3655], [70.7299, 92.2041],
        ], dtype=np.float32) * (size / 112.0)
        src = faces[index].landmarks.astype(np.float32)
        matrix, _ = cv2.estimateAffinePartial2D(src, template, method=cv2.LMEDS)
        if matrix is None:
            raise ValueError("Alignment transform could not be estimated")
        result = cv2.warpAffine(picture, matrix, (size, size), flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_REFLECT_101)
        return Image.fromarray(result)

    def annotate(self, img: Image.Image | np.ndarray | str, show_landmarks: bool = True) -> Image.Image:
        picture = rgb_image(img).copy()
        canvas = ImageDraw.Draw(picture)
        for n, face in enumerate(self.detect(picture), start=1):
            x1, y1, x2, y2 = face.box.tolist()
            canvas.rectangle((x1, y1, x2, y2), outline="#00eeaf", width=3)
            canvas.text((x1 + 3, max(0, y1 - 18)), f"Face {n} ({face.confidence:.2f})", fill="#00eeaf")
            if show_landmarks:
                for x, y in face.landmarks:
                    r = 3
                    canvas.ellipse((x-r, y-r, x+r, y+r), fill="#ffca3a")
        return picture

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float32).reshape(-1)
    y = np.asarray(b, dtype=np.float32).reshape(-1)
    if x.shape != y.shape or x.size == 0:
        raise ValueError("Embeddings must have the same nonzero length")
    return float(np.dot(x, y) / max(float(np.linalg.norm(x) * np.linalg.norm(y)), 1e-10))
