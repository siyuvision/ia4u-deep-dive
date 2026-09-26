"""Four-case fine-tuning followed by all three development evaluations, preserving prior runs."""
from fourshot_config import *
configure()
import hashlib
import json
import os
import random
import shutil
import time
from pathlib import Path
import numpy as np
import torch

def main():
    protocol=json.loads((EXP/'protocol.json').read_text())
    assert hashlib.sha256((EXP/'protocol.json').read_bytes()).hexdigest()==(EXP/'protocol.sha256').read_text().strip()
    assert hashlib.sha256(WEIGHTS.read_bytes()).hexdigest()==WEIGHTS_SHA
    from nnunetv2.experiment_planning.plan_and_preprocess_api import extract_fingerprints,plan_experiments
    from nnunetv2.experiment_planning.like_nnssl import preprocess_like_nnssl
    from nnunetv2.preprocessing.sampling_locations.extract_sampling_locations import extract_sampling_locations_dataset
    pre=PROJECT/'data/nnUNet_preprocessed'/DATASET
    state=dict(stage='preprocessing',pid=os.getpid(),started=time.time(),run=RUN);status=EXP/'status.json'
    def save():status.write_text(json.dumps(state,indent=2),encoding='utf-8')
    save()
    try:
        if not (pre/'dataset_fingerprint.json').exists():extract_fingerprints([DATASET_ID],num_processes=1,check_dataset_integrity=True)
        if not (pre/'nnUNetPlans.json').exists():plan_experiments([DATASET_ID],gpu_memory_target_in_gb=10)
        preprocess_like_nnssl(DATASET_ID,'nnFoundationCNN',str(WEIGHTS),'default_nnunet',None,num_processes=1,verbose=False)
        plan=json.loads(next(pre.glob('ptPlans__nnFoundationCNN*.json')).read_text());plan['plans_name']=PLAN
        plan['configurations']['3d_fullres']['patch_size']=[128]*3;plan['configurations']['3d_fullres']['batch_size']=2
        plan['tumseg_experiment']=dict(profile=RUN,epochs=30,iterations_per_epoch=50,monitoring_iterations=5,warmup_epochs=3,
             seed=20260926,training_cases=TRAIN,test_cases=TEST,checkpoint_selection='fixed final',monitoring='training cases only')
        (pre/(PLAN+'.json')).write_text(json.dumps(plan,indent=2),encoding='utf-8')
        extract_sampling_locations_dataset(DATASET,plans_identifier=PLAN,configurations=['3d_fullres'],num_processes=1,overwrite=False)
        import nnunetv2
        shutil.copy2(Path(__file__).with_name('tumseg_trainer.py'),Path(nnunetv2.__file__).parent/'training/nnUNetTrainer/tumseg_trainer.py')
        from nnunetv2.run.run_training_from_pretrained import get_trainer_from_args
        random.seed(20260926);np.random.seed(20260926);torch.manual_seed(20260926);torch.cuda.manual_seed_all(20260926)
        torch.set_num_threads(4);torch.backends.cudnn.benchmark=True
        trainer=get_trainer_from_args(DATASET,'3d_fullres','all','TumSegPretrainedTrainer',PLAN,device=torch.device('cuda'))
        if (Path(trainer.output_folder)/'checkpoint_final.pth').exists():raise FileExistsError('Four-shot model already exists; preserve it')
        state.update(stage='training',model_folder=trainer.output_folder_base);save()
        start=time.perf_counter();trainer.run_training()
        record=dict(model_folder=trainer.output_folder_base,seconds=time.perf_counter()-start,epochs=30,iterations=1500,
            gpu=torch.cuda.get_device_name(),torch=torch.__version__,cuda=torch.version.cuda,peak_gpu_allocated_gb=torch.cuda.max_memory_allocated()/1024**3,
            initialization_sha256=WEIGHTS_SHA)
        (EXP/'training.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
        state.update(stage='training_complete',finished=time.time());save()
        print('FOURSHOT_TRAIN_COMPLETE',json.dumps(record),flush=True)
    except BaseException as e:state.update(stage='failed',error=str(e));save();raise

if __name__=='__main__':main()
