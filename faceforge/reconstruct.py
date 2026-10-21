"""Optional official 3DDFA_V2 integration for actual single-image 3DMM meshes."""
from __future__ import annotations
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from .core import rgb_image


def reconstruct_mesh(image, repo_dir: str | Path, destination: str | Path, fmt: str = "obj", mode: str = "cpu") -> Path:
    if fmt not in {"obj", "ply"}:
        raise ValueError("3D format must be obj or ply")
    if mode not in {"cpu", "gpu"}:
        raise ValueError("Mode must be cpu or gpu")
    repo = Path(repo_dir).expanduser().resolve()
    if not (repo / "demo.py").is_file() or not (repo / "weights").is_dir():
        raise FileNotFoundError("Install the official 3DDFA_V2 repository and pretrained weights, then specify its folder")
    dest = Path(destination).expanduser().resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)
    (repo / "examples" / "results").mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / "faceforge_input.jpg"
        rgb_image(image).save(path, quality=95)
        cmd = [sys.executable, "demo.py", "-f", str(path), "-o", fmt, "-m", mode,
               "--show_flag", "false"]
        result = subprocess.run(cmd, cwd=repo, capture_output=True, text=True, timeout=240)
        if result.returncode != 0:
            raise RuntimeError("3DDFA_V2 failed:\n" + (result.stderr or result.stdout)[-2400:])
        generated = repo / "examples" / "results" / f"faceforge_input_{fmt}.{fmt}"
        if not generated.is_file():
            raise RuntimeError("3DDFA_V2 did not produce a mesh; confirm a face is visible.\n" + result.stdout[-1000:])
        shutil.copyfile(generated, dest)
        generated.unlink(missing_ok=True)
    return dest
