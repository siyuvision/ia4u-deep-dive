"""One-to-one, class-agnostic centroid matching before classification scoring."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


def match_points(gt, pred, radius):
    if not len(gt) or not len(pred):
        return []
    distances = cdist(np.asarray(gt), np.asarray(pred))
    # Invalid edges cost more than every valid edge combined: maximize matches first.
    penalty = (min(len(gt), len(pred)) + 1) * (radius + 1)
    rows, cols = linear_sum_assignment(np.where(distances <= radius, distances, penalty))
    return [(int(r), int(c)) for r,c in zip(rows,cols) if distances[r,c] <= radius]


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--instances', type=Path, required=True)
    p.add_argument('--image', type=Path, required=True)
    p.add_argument('--classes', type=Path, required=True)
    p.add_argument('--predictions', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--radius', type=float, default=12)
    a=p.parse_args()
    if a.radius <= 0:
        raise ValueError('radius must be positive')
    inst=np.asarray(Image.open(a.instances)); cls=np.asarray(Image.open(a.classes))
    if inst.ndim != 2 or cls.shape != inst.shape:
        raise ValueError('Expected equal-sized 2D annotation maps')
    if not set(np.unique(cls)).issubset(set(range(7))):
        raise ValueError('Invalid reference classes; use raw annotation bytes, not viewer previews')
    gt=[]; types=[]
    for iid in np.unique(inst):
        if iid == 0: continue
        y,x=np.where(inst==iid)
        gt.append([float(x.mean()),float(y.mean())])
        types.append(int(np.bincount(cls[y,x].astype(int)).argmax()))
    results=[]
    for path in a.predictions:
        item=json.loads(path.read_text(encoding='utf-8'))
        if item.get('image_sha256') != hashlib.sha256(a.image.read_bytes()).hexdigest():
            raise ValueError('Predictions were not generated from this exact reference image')
        pred=item['points']; pairs=match_points(gt,[[v['x'],v['y']] for v in pred],a.radius)
        tp=len(pairs); fp=len(pred)-tp; fn=len(gt)-tp
        cm=np.zeros((6,6),dtype=int)
        for g,q in pairs:
            c=pred[q].get('class_id_conic',pred[q]['class_id'])
            if c not in range(1,7) or types[g] not in range(1,7):
                raise ValueError('Class IDs must be mapped to CoNIC 1..6 before evaluation')
            cm[types[g]-1,c-1]+=1
        counts_gt=[types.count(i) for i in range(1,7)]
        counts_pred=[sum(v.get('class_id_conic',v['class_id'])==i for v in pred) for i in range(1,7)]
        results.append(dict(model=item['model'],tp=tp,fp=fp,fn=fn,precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None,f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,matched_class_accuracy=float(cm.trace()/tp) if tp else None,matched_confusion_matrix=cm.tolist(),counts_gt=counts_gt,counts_pred=counts_pred,counts_absolute_error=np.abs(np.array(counts_gt)-counts_pred).tolist(),warm_patch_seconds=item['warm_patch_seconds']))
    report=dict(scope='One public training-release patch; possible model training overlap. Demonstration only, not held-out performance or whole-slide speed.',radius_pixels=a.radius,reference_instances=len(gt),results=results)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
