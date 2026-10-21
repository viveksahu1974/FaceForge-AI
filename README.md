# FaceForge AI — 2D & 3D Face Analysis Toolkit

A local-first Python 3.10/3.11 + PyTorch face-analysis toolkit with five Gradio browser demos. Source code is available here without pretrained weights, private biometric data, or example face images.

## Features

| Feature | Implementation |
|---|---|
| Detection and 5-point landmarks | facenet-pytorch MTCNN |
| Face alignment | OpenCV similarity transform |
| Recognition of enrolled consenting people | FaceNet (VGGFace2) embeddings, local gallery |
| Age range (approximate) | `nateraw/vit-age-classifier` |
| 3D reconstruction | Optional locally installed 3DDFA_V2 pretrained extension |
| Photo/video face swapping | Geometric OpenCV baseline; optional licensed InsightFace ONNX extension |
| UI | Gradio 4.44.1, five tabs |

The built-in geometric swap is a simple baseline, not high-quality deepfake generation. 3DDFA_V2 must be installed separately and may require building native extensions. Neural swapping requires appropriate model licensing and extra dependencies. Face matching and age prediction can be inaccurate; biometric gallery data is sensitive.

## Windows 10/11 quick start (Python 3.11)

Double-click `START_WINDOWS.bat`, or in VS Code PowerShell from the repository root run:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e .
$env:NO_PROXY="localhost,127.0.0.1,::1"
$env:no_proxy="localhost,127.0.0.1,::1"
.\.venv\Scripts\python.exe app.py
```

Then open http://127.0.0.1:7860 . First use requires an internet connection to download pretrained weights. If you use Python 3.10, replace `py -3.11` with `py -3.10`.

We pin Gradio 4.44.1, Pydantic 2.10.6, FastAPI 0.112.2, and Starlette 0.38.6 to avoid a previously observed Gradio JSON-schema / ASGI startup conflict. For the localhost error, first check proxy settings and ensure `NO_PROXY` covers localhost. **Do not switch to `share=True` just to work around a proxy issue.**

## CLI commands

```bash
faceforge detect --image examples/person.jpg --out results/faces.png
faceforge align --image examples/person.jpg --out results/aligned.png
faceforge age --image examples/person.jpg
faceforge enroll --name "Example Person" --image examples/person.jpg
faceforge identify --image examples/query.jpg --threshold 0.68
faceforge swap --source examples/source.jpg --target examples/target.jpg --out results/swapped.png
faceforge swap --source examples/source.jpg --target examples/video.mp4 --video --out results/swapped.mp4
```

The example filenames are placeholders — supply your own authorized images/videos. Results are stored locally and should not be published without the subjects' permission. FFmpeg installed on PATH is recommended for browser-compatible video encoding and audio preservation.

## Optional pretrained extensions

For 3D mesh reconstruction follow the upstream installation/build instructions at https://github.com/cleardusk/3DDFA_V2 and install its pretrained weights. The local app accepts the absolute path to that repository and can export OBJ or PLY. The extension may need a separate Python environment on Windows.

For neural swapping, verify the model license before supplying `inswapper_128.onnx`. Then optionally install `pip install -e ".[neural]"` and choose neural mode. The neural backend uses ONNX Runtime, while the main FaceNet detection and recognition backend uses PyTorch.

## API example

```python
from faceforge.core import FaceEngine, cosine
from faceforge.swap import swap_photo
engine = FaceEngine()
faces = engine.detect("person.jpg")
engine.annotate("person.jpg").save("faces.png")
engine.align("person.jpg").save("aligned.png")
score = cosine(engine.embed("person1.jpg"), engine.embed("person2.jpg"))
print(score)
swap_photo("source.jpg", "target.jpg", engine).save("swapped.png")
```

## Privacy and testing

- Never add `.venv/`, `private/`, `.env`, biometric galleries, face photos or generated model weights to GitHub.
- The gallery is stored locally in `private/gallery.json` and is excluded by `.gitignore`.
- Use face swapping only when you have consent. AI-edited outputs receive a visible disclosure label.
- The similarity threshold is a demo default, not a production-validated identity decision.
- Run tests with `python -m pip install -e ".[test]"` followed by `pytest -q`. Tests do not require pretrained weight downloads.
- This repository provides code, not a hosted web demo. The Gradio UI runs locally.

## Upstream models

- https://github.com/timesler/facenet-pytorch
- https://huggingface.co/nateraw/vit-age-classifier
- https://github.com/cleardusk/3DDFA_V2
- https://github.com/deepinsight/insightface/blob/master/examples/in_swapper/README.md

Respect each upstream model's license and the rights of people whose likenesses are processed.
