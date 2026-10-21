"""Frame-wise image-to-video face swapping with FFmpeg audio preservation when installed."""
from __future__ import annotations
import shutil
import subprocess
import tempfile
from pathlib import Path
import cv2
from .swap import make_swapper, mark_generated
from .core import FaceEngine

def swap_video(source_image, video_path: str|Path, output_path: str|Path,
               engine: FaceEngine|None=None, mode="geometric",weights=None,
               all_faces=False,max_frames=900)->Path:
    if max_frames<1: raise ValueError("max_frames must be positive")
    video=Path(video_path)
    if not video.is_file(): raise FileNotFoundError(video)
    output=Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True,exist_ok=True)
    swapper=make_swapper(engine,mode=mode,neural_weights=weights)
    swapper.prepare_source(source_image)
    cap=cv2.VideoCapture(str(video))
    if not cap.isOpened(): raise ValueError(f"Cannot open video: {video}")
    fps=cap.get(cv2.CAP_PROP_FPS)
    fps=float(fps) if fps and 1<=fps<=240 else 25.0
    w,h=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if w<2 or h<2:
        cap.release()
        raise ValueError("Invalid video dimensions")
    with tempfile.TemporaryDirectory() as tmp:
        silent=Path(tmp)/"silent.mp4"
        writer=cv2.VideoWriter(str(silent),cv2.VideoWriter_fourcc(*"mp4v"),fps,(w,h))
        if not writer.isOpened():
            cap.release()
            raise RuntimeError("Cannot initialize MP4 encoder")
        count=0
        try:
            while count<max_frames:
                ok,frame=cap.read()
                if not ok: break
                writer.write(mark_generated(swapper.swap_frame(frame,all_faces=all_faces)))
                count+=1
        finally:
            cap.release()
            writer.release()
        if count==0: raise ValueError("Video contains no decodable frames")
        ffmpeg=shutil.which("ffmpeg")
        if ffmpeg:
            args=[ffmpeg,"-hide_banner","-loglevel","error","-y","-i",str(silent),"-i",str(video),
                  "-map","0:v:0","-map","1:a?","-c:v","libx264","-pix_fmt","yuv420p",
                  "-crf","21","-preset","veryfast","-c:a","aac","-shortest","-movflags","+faststart",str(output)]
            res=subprocess.run(args,capture_output=True,text=True)
            if res.returncode==0 and output.exists(): return output
        shutil.copy2(silent,output)
    return output
