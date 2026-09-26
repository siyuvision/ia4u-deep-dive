"""Infer every frozen unseen mouse CT from the fixed final checkpoint, with no postprocessing."""
from project_env import configure,PROJECT,DATASET
configure()
import argparse
import hashlib
import json
from pathlib import Path
import time
import torch
import SimpleITK as sitk

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--profile',default='pilot',choices=['pilot']);a=ap.parse_args()
    protocol=json.loads((PROJECT/'experiments/protocol.json').read_text())
    run=json.loads((PROJECT/'experiments'/f'training-{a.profile}.json').read_text())
    folder=Path(run['model_folder']);checkpoint_path=folder/'fold_all/checkpoint_final.pth'
    plan=json.loads((folder/'plans.json').read_text());dataset=json.loads((folder/'dataset.json').read_text())
    assert sorted(plan['tumseg_experiment']['training_cases'])==sorted(protocol['training_cases'])
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    from nnunetv2.utilities.get_network_via_name import get_network_from_name
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    torch.set_num_threads(4);pm=PlansManager(plan);cm=pm.get_configuration('3d_fullres')
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
    network=get_network_from_name('ResEncL',1,2,input_patchsize=cm.patch_size,deep_supervision=False)
    network.load_state_dict(checkpoint['network_weights'],strict=True)
    predictor=nnUNetPredictor(tile_step_size=.5,use_gaussian=True,use_mirroring=False,
        perform_everything_on_device=True,device=torch.device('cuda'),verbose=False,verbose_preprocessing=False,allow_tqdm=False)
    predictor.manual_initialization(network,pm,cm,[checkpoint['network_weights']],dataset,'TumSegPretrainedTrainer',None)
    out=PROJECT/'results'/a.profile/'predictions';out.mkdir(parents=True,exist_ok=True)
    cases=[]
    for case in protocol['test_cases']:
        image=PROJECT/'data/nnUNet_raw'/DATASET/'imagesTs'/(case+'_0000.nii.gz')
        target=out/(case+'.nii.gz')
        if target.exists():raise FileExistsError('Preserve original independent predictions: '+str(target))
        start=time.perf_counter()
        predictor.predict_from_files([[str(image)]],[str(out/case)],save_probabilities=False,overwrite=False,
            num_processes_preprocessing=1,num_processes_segmentation_export=1)
        img=sitk.ReadImage(str(target))
        for key,value in {'Segment0_ID':'Tumor','Segment0_Name':case+' predicted tumors','Segment0_Color':'0.95 0.4 0.15',
            'Segment0_LabelValue':'1','Segment0_Layer':'0','Segmentation_ContainedRepresentationNames':'Binary labelmap|',
            'Segmentation_MasterRepresentation':'Binary labelmap'}.items():img.SetMetaData(key,value)
        sitk.WriteImage(img,str(out/(case+'.seg.nrrd')),True)
        record=dict(case=case,seconds=time.perf_counter()-start,image=str(image),prediction=str(target),
            prediction_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),uses_reference_labels=False,postprocessing='none')
        cases.append(record);print('PREDICTED',json.dumps(record),flush=True)
    summary=dict(checkpoint=str(checkpoint_path),checkpoint_sha256=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
        cases=cases,seconds=sum(c['seconds'] for c in cases),device='cuda',peak_gpu_allocated_gb=torch.cuda.max_memory_allocated()/1024**3,
        test_time_augmentation=False,postprocessing='none; preserve possible bilateral/multiple tumors')
    (PROJECT/'experiments/inference-pilot.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('INFERENCE_COMPLETE',json.dumps(summary),flush=True)

if __name__=='__main__':main()
