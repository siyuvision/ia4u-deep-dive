"""Same-patch smoke test using each upstream model; not a WSI benchmark."""
import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from export_qupath import export_prediction_json

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ['neutrophil', 'epithelial', 'lymphocyte', 'plasma', 'eosinophil', 'connective']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', choices=['kongnet', 'classpose'], required=True)
    p.add_argument('--image', type=Path, required=True)
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; refusing to report CPU as GPU performance')
    image = np.asarray(Image.open(args.image).convert('RGB'))
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    if args.model == 'kongnet':
        sys.path.insert(0, str(ROOT / 'vendor/KongNet_Inference_Main'))
        from model.KongNet import get_KongNet
        from skimage.feature import peak_local_max
        model = get_KongNet(num_heads=6, decoders_out_channels=[3] * 6)
        state = torch.load(args.weights, map_location='cpu', weights_only=False)
        model.load_state_dict(state['model'], strict=True)
        del state
        model = model.eval().cuda()
        tensor = torch.as_tensor(image.copy()).permute(2, 0, 1)[None].float().cuda() / 255
        tensor = (tensor - tensor.new_tensor([.485,.456,.406])[None,:,None,None]) / tensor.new_tensor([.229,.224,.225])[None,:,None,None]
        torch.cuda.synchronize()
        load_s = time.perf_counter() - started
        def infer():
            with torch.inference_mode(), torch.autocast('cuda', enabled=False):
                return model(tensor).sigmoid()[0, [2,5,8,11,14,17]].cpu().numpy()
        infer()
        torch.cuda.synchronize()
        t = time.perf_counter()
        probs = infer()
        points = []
        for c, prob in enumerate(probs, 1):
            coords = peak_local_max(prob, min_distance=5, threshold_abs=.5, exclude_border=False)
            points.extend({'x':float(x), 'y':float(y), 'class_id':c, 'class_id_conic':c, 'score':float(prob[y,x])} for y,x in coords)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t
        np.save(args.output / 'probabilities.npy', probs)
        note = '256px single-patch adapter: official detection channels and peak settings; no WSI overlap NMS, no TTA, fp32. Not the full WSI pipeline.'
    else:
        from classpose.models import ClassposeModel
        state = torch.load(args.weights, map_location='cpu', weights_only=False)
        structure = [state[k].shape[0] for k in state if re.search(r'out_class\.encoder_blocks\.[0-9]+\.block.conv1.weight', k)] or None
        nclasses = state['W3'].shape[1]
        del state
        model = ClassposeModel(gpu=True, device=torch.device('cuda'), pretrained_model=str(args.weights), nclasses=nclasses, feature_transformation_structure=structure, precision='fp32')
        torch.cuda.synchronize()
        load_s = time.perf_counter() - started
        def infer():
            return model.eval(image, batch_size=1, resample=True, normalize=True, invert=False, diameter=None, augment=False, min_size=15, tile_overlap=.1, bsize=256, compute_masks=True)
        infer()
        torch.cuda.synchronize()
        t = time.perf_counter()
        masks, _, class_masks, _ = infer()
        points = []
        for instance in np.unique(masks):
            if instance == 0:
                continue
            yy, xx = np.where(masks == instance)
            label = int(np.bincount(class_masks[yy,xx].astype(int)).argmax())
            points.append({'x':float(xx.mean()), 'y':float(yy.mean()), 'class_id':label, 'class_id_raw':label, 'class_id_conic':label})
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t
        np.save(args.output / 'instances.npy', masks)
        np.save(args.output / 'classes.npy', class_masks)
        note = 'Upstream ClassposeModel.eval, fp32, no TTA; instance centroids used for detection comparison. Not WSI performance.'
    result = dict(model=args.model, image=str(args.image.resolve()), image_sha256=hashlib.sha256(args.image.read_bytes()).hexdigest(), image_shape=list(image.shape), weights=str(args.weights.resolve()), class_names=CLASSES, points=points, model_load_seconds=load_s, warm_patch_seconds=elapsed, torch_version=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0), notes=note)
    (args.output / 'predictions.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    export_prediction_json(args.output / 'predictions.json')
    print(json.dumps({'model':args.model, 'cells':len(points), 'warm_patch_seconds':elapsed}, indent=2))


if __name__ == '__main__':
    main()
