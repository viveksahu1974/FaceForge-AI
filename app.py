"""Local Gradio demos for PyTorch face analysis and consented swapping."""
from __future__ import annotations
from functools import lru_cache
from pathlib import Path
import tempfile
import gradio as gr
from faceforge.core import FaceEngine
from faceforge.gallery import FaceGallery

@lru_cache(maxsize=1)
def engine():
    return FaceEngine()

@lru_cache(maxsize=1)
def age_model():
    from faceforge.age import AgeEstimator
    return AgeEstimator()

def _status(ex):
    return f"Error: {type(ex).__name__}: {ex}"

def detect(picture,landmarks):
    if picture is None: return None,"Upload a photo."
    try:
        eng=engine()
        faces=eng.detect(picture)
        return eng.annotate(picture,landmarks),f"Detected **{len(faces)}** face(s)."
    except Exception as exc: return None,_status(exc)

def align(picture):
    if picture is None: return None,"Upload a photo."
    try: return engine().align(picture),"Aligned the largest face using five landmarks."
    except Exception as exc: return None,_status(exc)

def enroll(picture,name):
    if picture is None or not name or not name.strip():
        return "Upload a photo and a name with the person's permission."
    try:
        gallery=FaceGallery()
        gallery.enroll(name,engine().embed(picture))
        return f"Enrolled **{name.strip()}** locally. Total identities: **{len(gallery.entries)}**."
    except Exception as exc: return _status(exc)

def identify(picture,threshold):
    if picture is None: return "Upload a photo."
    try:
        name,score=FaceGallery().search(engine().embed(picture),float(threshold))
        return f"**Closest match:** {name}\n\n**Cosine similarity:** {score:.3f} (threshold {float(threshold):.2f}). Not a verified identity determination."
    except Exception as exc: return _status(exc)

def delete_identity(name):
    try: return "Identity deleted locally." if FaceGallery().delete(name.strip()) else "Name not in local gallery."
    except Exception as exc: return _status(exc)

def estimate_age(picture):
    if picture is None: return "Upload a photo."
    try:
        faces=engine().detect(picture)
        if not faces: return "No face detected."
        label,confidence=age_model().predict(picture,faces[0].box.tolist())
        return f"**Estimated age range:** {label}\n\nModel class probability: {confidence:.1%}. Age prediction can be inaccurate."
    except Exception as exc: return _status(exc)

def reconstruct(picture,repo,fmt):
    if picture is None or not repo.strip(): return None,"Upload a photo and provide the installed 3DDFA_V2 directory."
    try:
        from faceforge.reconstruct import reconstruct_mesh
        output=str(Path(tempfile.mkdtemp(prefix="faceforge_mesh_"))/f"reconstruction.{fmt}")
        result=reconstruct_mesh(picture,repo,output,fmt=fmt)
        return str(result),f"Exported {fmt.upper()} facial mesh."
    except Exception as exc: return None,_status(exc)

def swap_img(source,target,backend,weights,all_faces,confirmed):
    if source is None or target is None: return None,"Upload source and target pictures."
    if not confirmed: return None,"Confirm you have permission before creating a face swap."
    try:
        from faceforge.swap import swap_photo
        mode="neural" if backend=="Neural (licensed model)" else "geometric"
        result=swap_photo(source,target,engine() if mode=="geometric" else None,
                          mode=mode,weights=weights.strip() or None,all_faces=all_faces)
        return result,"AI-edited image created with a visible disclosure label."
    except Exception as exc: return None,_status(exc)

def swap_vid(source,target_path,backend,weights,all_faces,confirmed,limit):
    if source is None or not target_path: return None,"Upload a source photo and a target video."
    if not confirmed: return None,"Confirm you have permission before creating a face swap."
    try:
        from faceforge.video import swap_video
        mode="neural" if backend=="Neural (licensed model)" else "geometric"
        output=Path(tempfile.mkdtemp(prefix="faceforge_video_"))/"result.mp4"
        result=swap_video(source,target_path,output,engine=engine() if mode=="geometric" else None,
                          mode=mode,weights=weights.strip() or None,
                          all_faces=all_faces,max_frames=int(limit))
        return str(result),"Processed video; FFmpeg preserves audio if installed."
    except Exception as exc: return None,_status(exc)

with gr.Blocks(title="FaceForge AI • Local face analysis",theme=gr.themes.Soft()) as demo:
    gr.Markdown("# FaceForge AI — 2D/3D Face Lab\nLocal-first PyTorch face analysis. Models download on first use. Only process people who consent.")
    with gr.Tab("01 · Detection & Alignment"):
        with gr.Row():
            det_input=gr.Image(type="pil",label="Input image")
            det_output=gr.Image(type="pil",label="Detected faces & landmarks")
        show_marks=gr.Checkbox(value=True,label="Show five facial landmarks")
        det_button=gr.Button("Detect faces",variant="primary")
        det_text=gr.Markdown()
        det_button.click(detect,[det_input,show_marks],[det_output,det_text])
        with gr.Row():
            aligned=gr.Image(type="pil",label="Aligned largest face")
            align_text=gr.Markdown()
        align_button=gr.Button("Align largest face")
        align_button.click(align,[det_input],[aligned,align_text])
    with gr.Tab("02 · Recognition (enrolled only)"):
        gr.Markdown("Enroll **only consenting people**. Biometric embeddings stay in local `private/gallery.json`. This does not identify arbitrary strangers.")
        with gr.Row():
            enroll_img=gr.Image(type="pil",label="Enrollment portrait")
            with gr.Column():
                enroll_name=gr.Textbox(label="Person's display name")
                enroll_btn=gr.Button("Enroll person",variant="primary")
                enroll_output=gr.Markdown()
                enroll_btn.click(enroll,[enroll_img,enroll_name],enroll_output)
                delete_btn=gr.Button("Delete person's record")
                delete_btn.click(delete_identity,enroll_name,enroll_output)
        with gr.Row():
            find_img=gr.Image(type="pil",label="Query image")
            with gr.Column():
                match_threshold=gr.Slider(0.4,0.95,value=.68,step=.01,label="Similarity threshold")
                find_btn=gr.Button("Match against enrolled gallery")
                match_result=gr.Markdown()
                find_btn.click(identify,[find_img,match_threshold],match_result)
    with gr.Tab("03 · Age range"):
        with gr.Row():
            age_img=gr.Image(type="pil",label="Portrait")
            with gr.Column():
                age_btn=gr.Button("Estimate age range",variant="primary")
                age_result=gr.Markdown()
                age_btn.click(estimate_age,age_img,age_result)
        gr.Markdown("Age ranges are estimates, not proof of actual age.")
    with gr.Tab("04 · 3D reconstruction"):
        gr.Markdown("Requires installation of pretrained [3DDFA_V2](https://github.com/cleardusk/3DDFA_V2), including build steps.")
        with gr.Row():
            mesh_img=gr.Image(type="pil",label="Single face photo")
            with gr.Column():
                repo_path=gr.Textbox(label="3DDFA_V2 folder",placeholder="/path/to/3DDFA_V2")
                fmt=gr.Radio(["obj","ply"],value="obj",label="Mesh format")
                mesh_btn=gr.Button("Reconstruct face mesh",variant="primary")
                mesh_file=gr.File(label="Download OBJ/PLY mesh")
                mesh_status=gr.Markdown()
                mesh_btn.click(reconstruct,[mesh_img,repo_path,fmt],[mesh_file,mesh_status])
    with gr.Tab("05 · Face swap — photo & video"):
        gr.Markdown("Only edit images with permission. Output is labeled **AI FACE SWAP**. Geometric mode is a baseline; neural mode requires a licensed ONNX model.")
        with gr.Row():
            src_img=gr.Image(type="pil",label="Source face")
            target_img=gr.Image(type="pil",label="Target photo")
        with gr.Row():
            backend=gr.Radio(["Geometric (no extra model)","Neural (licensed model)"],value="Geometric (no extra model)",label="Swap method")
            weights=gr.Textbox(label="Licensed neural ONNX weights path",placeholder="/path/to/licensed/inswapper_128.onnx")
        all_faces=gr.Checkbox(value=False,label="Replace every target face")
        confirmed=gr.Checkbox(value=False,label="I have permission to edit the likenesses in these images/video.")
        photo_btn=gr.Button("Swap faces in photo",variant="primary")
        with gr.Row():
            swapped=gr.Image(type="pil",label="AI-edited output")
            swapped_status=gr.Markdown()
        photo_btn.click(swap_img,[src_img,target_img,backend,weights,all_faces,confirmed],[swapped,swapped_status])
        gr.Markdown("### Video face swap")
        video_input=gr.Video(label="Target video")
        frame_limit=gr.Slider(30,1500,value=450,step=30,label="Maximum output frames")
        video_btn=gr.Button("Swap faces in video",variant="primary")
        with gr.Row():
            video_out=gr.Video(label="AI-edited video")
            video_status=gr.Markdown()
        video_btn.click(swap_vid,[src_img,video_input,backend,weights,all_faces,confirmed,frame_limit],[video_out,video_status])
    gr.Markdown("Local by default. Never publish biometric galleries. Pretrained weights may download on first use.")

if __name__=="__main__":
    demo.launch(server_name="127.0.0.1",server_port=7860,share=False,show_error=True)
