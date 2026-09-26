"""Diagnostic full-volume resubstitution only: fixed checkpoint, two training individuals."""
from project_env import configure,PROJECT,DATASET
configure()
import hashlib
import json
from pathlib import Path
import time
import torch
import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi
import cc3d
from evaluate import overlap_metrics,surface_metrics
from render_assets import save_plane,save_projection

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    protocol=json.loads((PROJECT/'experiments/protocol.json').read_text())
    test_metrics_path=PROJECT/'results/pilot/metrics.json';test_metrics_hash=sha(test_metrics_path)
    original=json.loads((PROJECT/'experiments/inference-pilot.json').read_text())
    checkpoint_path=Path(original['checkpoint']);checkpoint_hash=sha(checkpoint_path)
    assert checkpoint_hash==original['checkpoint_sha256']
    model=checkpoint_path.parent.parent
    plan=json.loads((model/'plans.json').read_text());dataset=json.loads((model/'dataset.json').read_text())
    assert protocol['training_cases']==plan['tumseg_experiment']['training_cases']
    out=PROJECT/'results/diagnostics/train_replay'
    if out.exists():raise FileExistsError('Preserve earlier diagnostic results')
    out.mkdir(parents=True)
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    from nnunetv2.utilities.get_network_via_name import get_network_from_name
    torch.set_num_threads(4);pm=PlansManager(plan);cm=pm.get_configuration('3d_fullres')
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
    network=get_network_from_name('ResEncL',1,2,input_patchsize=cm.patch_size,deep_supervision=False)
    network.load_state_dict(checkpoint['network_weights'],strict=True)
    predictor=nnUNetPredictor(tile_step_size=.5,use_gaussian=True,use_mirroring=False,
        perform_everything_on_device=True,device=torch.device('cuda'),verbose=False,verbose_preprocessing=False,allow_tqdm=False)
    predictor.manual_initialization(network,pm,cm,[checkpoint['network_weights']],dataset,'TumSegPretrainedTrainer',None)
    records=[]
    for case in protocol['training_cases']:
        caseout=out/case;caseout.mkdir()
        image=PROJECT/'data/nnUNet_raw'/DATASET/'imagesTr'/(case+'_0000.nii.gz')
        reference=PROJECT/'data/nnUNet_raw'/DATASET/'labelsTr'/(case+'.nii.gz')
        started=time.perf_counter()
        predictor.predict_from_files([[str(image)]],[str(caseout/'prediction')],save_probabilities=False,overwrite=False,
                                    num_processes_preprocessing=1,num_processes_segmentation_export=1)
        seconds=time.perf_counter()-started
        ci,ri,pi=[sitk.ReadImage(str(p)) for p in [image,reference,caseout/'prediction.nii.gz']]
        for getter in ['GetSize','GetSpacing','GetDirection','GetOrigin']:
            assert np.allclose(getattr(pi,getter)(),getattr(ri,getter)())
        ct,ref,pred=[sitk.GetArrayFromImage(i) for i in [ci,ri,pi]];ref=ref>0;pred=pred>0;spacing=ci.GetSpacing()
        metrics=overlap_metrics(pred,ref,spacing);metrics.update(surface_metrics(pred,ref,spacing))
        overlap=sitk.LabelOverlapMeasuresImageFilter();overlap.Execute(ri,pi)
        assert abs(overlap.GetDiceCoefficient()-metrics['dice'])<1e-12
        labels,n=cc3d.connected_components(pred,connectivity=26,return_N=True)
        counts=np.bincount(labels.ravel());hits=np.bincount(labels[ref],minlength=len(counts))
        ids=np.flatnonzero((counts>0)&(hits==0));ids=ids[ids!=0]
        unmatched=int(counts[ids].sum());fp=pred&~ref
        dist=ndi.distance_transform_edt(~ref,sampling=spacing[::-1]);far=int(np.count_nonzero(fp&(dist>10)));del dist
        diagnostic=dict(prediction_components=int(n),unmatched_components=len(ids),unmatched_foreground_voxels=unmatched,
                        fp_beyond_10mm=far,far10_fraction_of_all_fp=far/int(fp.sum()) if fp.any() else None)
        z=int(np.argmax(ref.sum(axis=(1,2))));fp_z=int(np.argmax(fp.sum(axis=(1,2))))
        extent=[0,ct.shape[2]*spacing[0],0,ct.shape[1]*spacing[1]]
        save_projection(caseout/'full_volume.png',ct,ref,pred,spacing,case+' | TRAIN REPLAY','both',axis=2)
        save_plane(caseout/'target_slice.png',ct[z],ref[z],pred[z],'both',case+' | TRAIN REPLAY | target slice',extent)
        save_plane(caseout/'max_fp_error.png',ct[fp_z],ref[fp_z],pred[fp_z],'error',case+' | TRAIN REPLAY | max-FP',extent)
        record=dict(case=case,role='training resubstitution; not independent test',metrics=metrics,diagnostic=diagnostic,
                    inference_seconds=seconds,postprocessing='none',prediction_sha256=sha(caseout/'prediction.nii.gz'))
        (caseout/'metrics.json').write_text(json.dumps(record,indent=2),encoding='utf-8');records.append(record)
        print('TRAIN_REPLAY_COMPLETE',json.dumps(record),flush=True)
    summary=dict(checkpoint=str(checkpoint_path),checkpoint_sha256=checkpoint_hash,cases=records,
        macro={k:float(np.mean([r['metrics'][k] for r in records])) for k in ['dice','iou','precision','recall']},
        role='Training cases only; do not pool these with the five frozen independent test cases',
        postprocessing='none',training_performed=False,hyperparameter_changes=False,
        test_metrics_sha256_before=test_metrics_hash,test_metrics_unchanged=sha(test_metrics_path)==test_metrics_hash,
        source_sha256=sha(Path(__file__)))
    assert summary['test_metrics_unchanged'] and sha(checkpoint_path)==checkpoint_hash
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('ALL_TRAIN_REPLAY_COMPLETE',json.dumps(summary['macro']),flush=True)

if __name__=='__main__':main()
