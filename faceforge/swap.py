"""Consent-based geometric face compositing plus optional licensed ONNX neural swapper."""
from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from .core import Face, FaceEngine, rgb_image

def mark_generated(bgr: np.ndarray) -> np.ndarray:
    out = bgr.copy()
    h, w = out.shape[:2]
    scale = max(0.45, min(w / 900, 1.0))
    label = "AI FACE SWAP"
    font, weight = cv2.FONT_HERSHEY_SIMPLEX, max(1, int(2 * scale))
    (tw, th), _ = cv2.getTextSize(label, font, scale, weight)
    x, y = max(5, w - tw - 12), max(th + 7, h - 12)
    cv2.rectangle(out, (x-5, y-th-7), (min(w-1, x+tw+6), min(h-1, y+7)), (18,18,18), -1)
    cv2.putText(out, label, (x,y), font, scale, (245,245,245), weight, cv2.LINE_AA)
    return out

def _pil_to_bgr(img: Image.Image | np.ndarray | str) -> np.ndarray:
    return cv2.cvtColor(np.asarray(rgb_image(img)), cv2.COLOR_RGB2BGR)

def _box_ellipse(shape: tuple[int,int], box: np.ndarray) -> np.ndarray:
    h,w = shape
    x1,y1,x2,y2 = [float(v) for v in box]
    cx,cy = int((x1+x2)/2),int(y1 + .51*(y2-y1))
    ax,ay = max(1,int((x2-x1)*.43)),max(1,int((y2-y1)*.49))
    mask = np.zeros((h,w),dtype=np.uint8)
    cv2.ellipse(mask,(cx,cy),(ax,ay),0,0,360,255,-1)
    return mask

def geometric_swap_frame(source_bgr: np.ndarray, target_bgr: np.ndarray,
                         source_face: Face, target_face: Face) -> np.ndarray:
    src,dst = np.asarray(source_bgr,dtype=np.uint8),np.asarray(target_bgr,dtype=np.uint8)
    matrix,_ = cv2.estimateAffinePartial2D(source_face.landmarks.astype(np.float32),
                    target_face.landmarks.astype(np.float32),method=cv2.LMEDS)
    if matrix is None: raise ValueError("Could not estimate face alignment")
    h,w = dst.shape[:2]
    warped = cv2.warpAffine(src,matrix,(w,h),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
    mask = cv2.warpAffine(_box_ellipse(src.shape[:2],source_face.box),matrix,(w,h),flags=cv2.INTER_NEAREST)
    k = max(3,int(min(h,w)/250)|1)
    mask = cv2.erode(mask,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(k,k)))
    if cv2.countNonZero(mask)<30: raise ValueError("Warped face is outside target")
    x,y,mw,mh = cv2.boundingRect(mask)
    try:
        return cv2.seamlessClone(warped,dst,mask,(x+mw//2,y+mh//2),cv2.NORMAL_CLONE)
    except cv2.error as exc:
        raise ValueError("Unable to blend; try clearer frontal portraits") from exc

class GeometricSwapper:
    def __init__(self, engine: FaceEngine):
        self.engine=engine
        self.source_bgr=None
        self.source_face=None
    def prepare_source(self,image):
        self.source_bgr=_pil_to_bgr(image)
        faces=self.engine.detect(rgb_image(image))
        if not faces: raise ValueError("No source face detected")
        self.source_face=faces[0]
    def swap_frame(self,target_bgr,index=0,all_faces=False):
        if self.source_face is None: raise RuntimeError("Call prepare_source first")
        faces=self.engine.detect(cv2.cvtColor(target_bgr,cv2.COLOR_BGR2RGB))
        if not faces: return target_bgr.copy()
        result=target_bgr.copy()
        for face in (faces if all_faces else faces[index:index+1]):
            result=geometric_swap_frame(self.source_bgr,result,self.source_face,face)
        return result

class NeuralSwapper:
    def __init__(self,weights):
        path=Path(weights).expanduser().resolve()
        if not path.is_file(): raise FileNotFoundError("Supply a licensed INSwapper ONNX model")
        try:
            from insightface.app import FaceAnalysis
            from insightface.model_zoo import get_model
        except ImportError as exc:
            raise RuntimeError("Install optional extras: pip install '.[neural]'") from exc
        providers=["CPUExecutionProvider"]
        self.analyser=FaceAnalysis(name="buffalo_l",providers=providers)
        self.analyser.prepare(ctx_id=-1,det_size=(640,640))
        self.swapper=get_model(str(path),providers=providers)
        self.source_face=None
    @staticmethod
    def _sorted(faces):
        return sorted(faces,key=lambda f:float((f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1])),reverse=True)
    def prepare_source(self,image):
        faces=self._sorted(self.analyser.get(_pil_to_bgr(image)))
        if not faces: raise ValueError("No source face detected")
        self.source_face=faces[0]
    def swap_frame(self,target_bgr,index=0,all_faces=False):
        if self.source_face is None: raise RuntimeError("Call prepare_source first")
        faces=self._sorted(self.analyser.get(target_bgr))
        result=target_bgr.copy()
        for face in (faces if all_faces else faces[index:index+1]):
            result=self.swapper.get(result,face,self.source_face,paste_back=True)
        return result

def make_swapper(engine:FaceEngine|None=None,mode="geometric",neural_weights=None):
    if mode=="geometric": return GeometricSwapper(engine or FaceEngine())
    if mode=="neural":
        if not neural_weights: raise ValueError("Neural mode requires licensed ONNX weights")
        return NeuralSwapper(neural_weights)
    raise ValueError("Mode must be geometric or neural")

def swap_photo(source,target,engine=None,mode="geometric",weights=None,all_faces=False)->Image.Image:
    swapper=make_swapper(engine,mode,weights)
    swapper.prepare_source(source)
    result=swapper.swap_frame(_pil_to_bgr(target),all_faces=all_faces)
    return Image.fromarray(cv2.cvtColor(mark_generated(result),cv2.COLOR_BGR2RGB))
