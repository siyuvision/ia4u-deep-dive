from project_env import configure, PROJECT, DATASET
configure()
import argparse
import json
import random
import time
import numpy as np
import torch

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--profile', choices=['pilot','smoke'], default='pilot')
    ap.add_argument('--resume', action='store_true')
    args = ap.parse_args()
    seed=20260926
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU is required for this experiment')
    from nnunetv2.run.run_training_from_pretrained import get_trainer_from_args
    name = f'TumSeg_nnFoundationCNN_{args.profile}_128'
    trainer = get_trainer_from_args(DATASET, '3d_fullres', 'all', 'TumSegPretrainedTrainer', name,
                                    device=torch.device('cuda'), continue_training=args.resume)
    if args.resume:
        trainer.use_pretrained_weights=False
        latest = __import__('pathlib').Path(trainer.output_folder)/'checkpoint_latest.pth'
        trainer.load_checkpoint(str(latest))
    elif (__import__('pathlib').Path(trainer.output_folder)/'checkpoint_final.pth').exists():
        raise RuntimeError('Run already finished. Preserve it; use a new profile for a new experiment.')
    start = time.perf_counter()
    trainer.run_training()
    summary = dict(profile=args.profile, seconds=time.perf_counter()-start,
        model_folder=trainer.output_folder_base, fold='all', checkpoint='checkpoint_final.pth',
        epochs=trainer.num_epochs, optimizer_steps=trainer.num_epochs*trainer.num_iterations_per_epoch,
        peak_gpu_allocated_gb=torch.cuda.max_memory_allocated()/1024**3,
        torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
        checkpoint_selection='fixed final checkpoint; test unused')
    (PROJECT/'experiments'/f'training-{args.profile}.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print('TRAINING_COMPLETE', json.dumps(summary), flush=True)

if __name__ == '__main__':
    main()
