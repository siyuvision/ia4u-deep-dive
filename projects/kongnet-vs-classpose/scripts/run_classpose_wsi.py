"""In-process whole-slide ClassPose adapter for Windows benchmarking.

The upstream predict_wsi entrypoint relies on torch.multiprocessing plus a
SyncManager and stalls on Windows, so this adapter feeds non-overlapping
slide tiles to the same ClassposeModel.eval call used by run_patch.py (same
weights, fp32, no TTA).  Detections are instance centroids in level-0 slide
pixels with both the raw model label and the candidate CoNIC label (raw+1).
This is an adapted local pipeline benchmark, not the upstream Linux pipeline.
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import openslide
from PIL import Image
import torch
from export_qupath import export_prediction_json

ROOT = Path(__file__).resolve().parents[1]
CLASSES = ['neutrophil', 'epithelial', 'lymphocyte', 'plasma', 'eosinophil', 'connective']
TARGET_MPP = 0.5
TILE_PX = 1024


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--wsi', type=Path, required=True)
    p.add_argument('--mask', type=Path, required=True)
    p.add_argument('--weights', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; refusing to report CPU as GPU performance')

    started = time.perf_counter()
    slide = openslide.OpenSlide(str(args.wsi))
    base_mpp = float(slide.properties.get('openslide.mpp-x') or 0.25)
    target_downsample = TARGET_MPP / base_mpp
    level = min(
        range(slide.level_count),
        key=lambda l: abs(slide.level_downsamples[l] - target_downsample),
    )
    downsample = slide.level_downsamples[level]
    mpp_at_level = base_mpp * downsample
    resize_factor = downsample / target_downsample
    w, h = slide.level_dimensions[level]
    dims0 = list(slide.level_dimensions[0])
    print(f'Slide MPP: {base_mpp}, target MPP: {TARGET_MPP}, '
          f'level: {level}, level downsample: {downsample}, '
          f'MPP at level: {mpp_at_level:.4f}, '
          f'resize factor: {resize_factor:.4f}', flush=True)
    mask = np.asarray(Image.open(args.mask).convert('L'))
    grid_w = int(np.ceil(w / TILE_PX))
    grid_h = int(np.ceil(h / TILE_PX))
    mask_small = np.asarray(
        Image.fromarray(mask).resize((grid_w, grid_h), Image.BOX), dtype=np.float32
    )
    keep = mask_small > 255 * 0.04  # mean mask value in the tile's pixels

    sys.path.insert(0, str(ROOT / 'vendor/classpose/src'))
    from classpose.models import ClassposeModel

    state = torch.load(args.weights, map_location='cpu', weights_only=False)
    structure = [state[k].shape[0] for k in state if re.search(r'out_class\.encoder_blocks\.[0-9]+\.block.conv1.weight', k)] or None
    nclasses = state['W3'].shape[1]
    del state
    model = ClassposeModel(
        gpu=True,
        device=torch.device('cuda'),
        pretrained_model=str(args.weights),
        nclasses=nclasses,
        feature_transformation_structure=structure,
        precision='fp32',
    )
    torch.cuda.synchronize()
    model_load_s = time.perf_counter() - started

    points = []
    tiles_done = 0
    tiles_skipped = 0
    infer_started = time.perf_counter()
    for gy in range(grid_h):
        y0 = gy * TILE_PX
        for gx in range(grid_w):
            if not keep[gy, gx]:
                tiles_skipped += 1
                continue
            x0 = gx * TILE_PX
            tw = min(TILE_PX, w - x0)
            th = min(TILE_PX, h - y0)
            if tw <= 0 or th <= 0:
                continue
            tile = np.asarray(
                slide.read_region((int(x0 * downsample), int(y0 * downsample)), level, (tw, th))
                .convert('RGB')
            )
            tile = np.ascontiguousarray(tile, dtype=np.uint8)
            if abs(resize_factor - 1.0) > 0.01:
                tile = cv2.resize(
                    tile, None,
                    fx=resize_factor, fy=resize_factor,
                    interpolation=cv2.INTER_LINEAR,
                )
            masks, _, class_masks, _ = model.eval(
                tile,
                batch_size=1,
                resample=True,
                normalize=True,
                invert=False,
                diameter=None,
                augment=False,
                min_size=15,
                tile_overlap=.1,
                bsize=256,
                compute_masks=True,
            )
            pred_to_level = 1.0 / resize_factor if abs(resize_factor - 1.0) > 0.01 else 1.0
            for instance in np.unique(masks):
                if instance == 0:
                    continue
                yy, xx = np.where(masks == instance)
                label = int(np.bincount(class_masks[yy, xx].astype(int)).argmax())
                cx_level = x0 + float(xx.mean()) * pred_to_level
                cy_level = y0 + float(yy.mean()) * pred_to_level
                points.append({
                    'x': cx_level * downsample,
                    'y': cy_level * downsample,
                    'class_id': label,
                    'class_id_raw': label,
                    'class_id_conic': label,
                })
            tiles_done += 1
    torch.cuda.synchronize()
    infer_s = time.perf_counter() - infer_started
    slide.close()
    wall_s = time.perf_counter() - started

    counts = [sum(pt['class_id_conic'] == i for pt in points) for i in range(1, 7)]
    args.output.mkdir(parents=True, exist_ok=True)
    report = {
        'model': 'classpose',
        'mode': 'windows in-process whole-slide adapter',
        'slide': str(args.wsi.resolve()),
        'mask': str(args.mask.resolve()),
        'slide_dimensions': dims0,
        'level_used': level,
        'downsample': downsample,
        'mpp_at_level': mpp_at_level,
        'target_mpp': TARGET_MPP,
        'resize_factor': resize_factor,
        'tile_px': TILE_PX,
        'tiles_total': int(keep.sum()),
        'tiles_processed': tiles_done,
        'tiles_skipped': tiles_skipped,
        'cells': len(points),
        'counts_by_class_conic': counts,
        'class_names': CLASSES,
        'model_load_seconds': model_load_s,
        'inference_seconds': infer_s,
        'wall_seconds_total': wall_s,
        'torch_version': torch.__version__,
        'cuda': torch.version.cuda,
        'gpu': torch.cuda.get_device_name(0),
        'points': points,
        'notes': 'Adapted local benchmark on Windows; per-tile in-process '
                 'ClassposeModel.eval identical to run_patch.py; '
                 'tiles resized to target MPP before inference; '
                 'class_id_conic = raw model label (CoNIC identity).',
    }
    out_json = args.output / 'case01_classpose.json'
    out_json.write_text(json.dumps(report, indent=2), encoding='utf-8')
    export_prediction_json(out_json)
    print(json.dumps({k: report[k] for k in
                      ['cells', 'tiles_processed', 'inference_seconds',
                       'model_load_seconds', 'wall_seconds_total',
                       'counts_by_class_conic']}, indent=2))
if __name__ == '__main__':
    main()
