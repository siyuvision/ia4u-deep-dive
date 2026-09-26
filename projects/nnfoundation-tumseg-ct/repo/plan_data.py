from project_env import configure,PROJECT,DATASET,WEIGHTS,WEIGHTS_SHA
configure()
import argparse
import hashlib
import json
import shutil
from pathlib import Path

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--profile',choices=['pilot','smoke'],default='pilot');a=ap.parse_args()
    protocol=json.loads((PROJECT/'experiments/protocol.json').read_text())
    assert hashlib.sha256((PROJECT/'experiments/protocol.json').read_bytes()).hexdigest()==(PROJECT/'experiments/protocol.sha256').read_text().strip()
    assert hashlib.sha256(WEIGHTS.read_bytes()).hexdigest()==WEIGHTS_SHA
    raw=PROJECT/'data/nnUNet_raw'/DATASET
    assert sorted(p.name for p in (raw/'labelsTr').glob('*.nii.gz'))==sorted(c+'.nii.gz' for c in protocol['training_cases'])
    from nnunetv2.experiment_planning.plan_and_preprocess_api import extract_fingerprints,plan_experiments
    from nnunetv2.experiment_planning.like_nnssl import preprocess_like_nnssl
    pre=PROJECT/'data/nnUNet_preprocessed'/DATASET
    if not (pre/'dataset_fingerprint.json').exists():extract_fingerprints([902],num_processes=1,check_dataset_integrity=True)
    if not (pre/'nnUNetPlans.json').exists():plan_experiments([902],gpu_memory_target_in_gb=10)
    preprocess_like_nnssl(902,'nnFoundationCNN',str(WEIGHTS),'default_nnunet',None,num_processes=1,verbose=False)
    original=next(pre.glob('ptPlans__nnFoundationCNN*.json'));plan=json.loads(original.read_text())
    name=f'TumSeg_nnFoundationCNN_{a.profile}_128';plan['plans_name']=name
    cfg=plan['configurations']['3d_fullres'];cfg['patch_size']=[128]*3;cfg['batch_size']=2
    assert all(abs(s-.21)<1e-5 for s in cfg['spacing'])
    plan['tumseg_experiment']=dict(profile=a.profile,epochs=30 if a.profile=='pilot' else 1,
        iterations_per_epoch=50 if a.profile=='pilot' else 2,warmup_epochs=3 if a.profile=='pilot' else 1,
        monitoring_iterations=5 if a.profile=='pilot' else 1,seed=20260926,training_cases=protocol['training_cases'],
        test_cases=protocol['test_cases'],checkpoint_selection='fixed final',
        monitoring='training cases only; no independent validation',postprocessing='none')
    (pre/(name+'.json')).write_text(json.dumps(plan,indent=2),encoding='utf-8')
    from nnunetv2.preprocessing.sampling_locations.extract_sampling_locations import extract_sampling_locations_dataset
    extract_sampling_locations_dataset(DATASET,plans_identifier=name,configurations=['3d_fullres'],num_processes=1,overwrite=False)
    import nnunetv2
    shutil.copy2(Path(__file__).with_name('tumseg_trainer.py'),Path(nnunetv2.__file__).parent/'training/nnUNetTrainer/tumseg_trainer.py')
    print('PLAN_READY',name,flush=True)

if __name__=='__main__':main()
