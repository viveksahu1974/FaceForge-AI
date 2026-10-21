"""FaceForge AI command-line interface."""
import argparse
import json
from pathlib import Path
from .core import FaceEngine
from .gallery import FaceGallery

def main():
    parser=argparse.ArgumentParser(prog="faceforge",description="PyTorch face analysis and authorized swaps")
    sub=parser.add_subparsers(dest="command",required=True)
    for name in ("detect","align","age","enroll","identify","mesh"):
        p=sub.add_parser(name)
        p.add_argument("--image",required=True)
        if name in ("detect","align","mesh"): p.add_argument("--out",required=True)
        if name=="enroll": p.add_argument("--name",required=True)
        if name in ("enroll","identify"): p.add_argument("--gallery",default="private/gallery.json")
        if name=="identify": p.add_argument("--threshold",type=float,default=.68)
        if name=="mesh":
            p.add_argument("--repo",required=True)
            p.add_argument("--format",choices=["obj","ply"],default="obj")
    p=sub.add_parser("swap")
    p.add_argument("--source",required=True)
    p.add_argument("--target",required=True)
    p.add_argument("--out",required=True)
    p.add_argument("--mode",choices=["geometric","neural"],default="geometric")
    p.add_argument("--weights")
    p.add_argument("--all-faces",action="store_true")
    p.add_argument("--video",action="store_true")
    p.add_argument("--max-frames",type=int,default=900)
    args=parser.parse_args()
    if args.command=="mesh":
        from .reconstruct import reconstruct_mesh
        print(reconstruct_mesh(args.image,args.repo,args.out,fmt=args.format))
        return
    if args.command=="swap":
        from .swap import swap_photo
        from .video import swap_video
        engine=FaceEngine() if args.mode=="geometric" else None
        if args.video:
            result=swap_video(args.source,args.target,args.out,engine=engine,mode=args.mode,
                              weights=args.weights,all_faces=args.all_faces,max_frames=args.max_frames)
        else:
            result=Path(args.out)
            result.parent.mkdir(parents=True,exist_ok=True)
            swap_photo(args.source,args.target,engine,args.mode,args.weights,args.all_faces).save(result)
        print(result)
        return
    engine=FaceEngine()
    if args.command=="detect":
        Path(args.out).parent.mkdir(parents=True,exist_ok=True)
        engine.annotate(args.image).save(args.out)
        print(json.dumps([{"box":f.box.tolist(),"landmarks":f.landmarks.tolist(),
                           "confidence":f.confidence} for f in engine.detect(args.image)],indent=2))
    elif args.command=="align":
        Path(args.out).parent.mkdir(parents=True,exist_ok=True)
        engine.align(args.image).save(args.out)
    elif args.command=="age":
        from .age import AgeEstimator
        faces=engine.detect(args.image)
        if not faces: raise ValueError("No face found")
        label,confidence=AgeEstimator().predict(args.image,faces[0].box.tolist())
        print(json.dumps({"age_range":label,"classification_confidence":confidence}))
    elif args.command=="enroll":
        FaceGallery(args.gallery).enroll(args.name,engine.embed(args.image))
        print("Locally enrolled:",args.name)
    elif args.command=="identify":
        name,score=FaceGallery(args.gallery).search(engine.embed(args.image),args.threshold)
        print(json.dumps({"match":name,"cosine_similarity":score,"threshold":args.threshold}))

if __name__=="__main__":
    main()
