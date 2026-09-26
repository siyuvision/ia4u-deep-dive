"""Fixed-final four-shot inference, frozen guard, and paired development evaluation."""
from fourshot_config import *
configure()
import hashlib
import json
import time
from pathlib import Path
import cc3d
import numpy as np
import torch
import SimpleITK as sitk
from scipy import ndimage as ndi
from evaluate import overlap_metrics,surface_metrics
from spatial_guard import apply_file

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def export_seg(image,path,name):
    image=sitk.Cast(image,sitk.sitkUInt8);size=image.GetSize()
    for key,value in {'Segment0_ID':'Tumor','Segment0_Name':name,'Segment0_Color':'1 0.6 0.25','Segment0_LabelValue':'1','Segment0_Layer':'0',
        'Segment0_Extent':f'0 {size[0]-1} 0 {size[1]-1} 0 {size[2]-1}',
        'Segmentation_ReferenceImageExtentOffset':'0 0 0','Segmentation_ContainedRepresentationNames':'Binary labelmap|',
        'Segmentation_MasterRepresentation':'Binary labelmap'}.items():image.SetMetaData(key,value)
    sitk.WriteImage(image,str(path),True)

def main():
    protocol=json.loads((EXP/'protocol.json').read_text());prior=EXP/'spatial-prior.json'
    assert sha(prior)==protocol['postprocessing_prior_sha256']
    assert sha(EXP/'protocol.json')==(EXP/'protocol.sha256').read_text().strip()
    train=json.loads((EXP/'training.json').read_text());model=Path(train['model_folder'])
    checkpoint_path=model/'fold_all/checkpoint_final.pth';checkpoint_hash=sha(checkpoint_path)
    original_metrics_path=PROJECT/'results/pilot/metrics.json';original_hash=sha(original_metrics_path)
    original=json.loads(original_metrics_path.read_text());old={r['case']:r['references']['STAPLE'] for r in original['cases']}
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
    from nnunetv2.utilities.get_network_via_name import get_network_from_name
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    plan=json.loads((model/'plans.json').read_text());dataset=json.loads((model/'dataset.json').read_text());pm=PlansManager(plan);cm=pm.get_configuration('3d_fullres')
    torch.set_num_threads(4)
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=False)
    assert checkpoint['current_epoch']==30
    net=get_network_from_name('ResEncL',1,2,input_patchsize=cm.patch_size,deep_supervision=False)
    net.load_state_dict(checkpoint['network_weights'],strict=True)
    predictor=nnUNetPredictor(tile_step_size=.5,use_gaussian=True,use_mirroring=False,perform_everything_on_device=True,
        device=torch.device('cuda'),verbose=False,verbose_preprocessing=False,allow_tqdm=False)
    predictor.manual_initialization(net,pm,cm,[checkpoint['network_weights']],dataset,'TumSegPretrainedTrainer',None)
    all_records=[]
    for case in TEST:
        image=PROJECT/'data/nnUNet_raw'/DATASET/'imagesTs'/(case+'_0000.nii.gz');_,reference=sources(case)
        out=RESULTS/case;out.mkdir(parents=True,exist_ok=True)
        raw_path=out/'raw.nii.gz';clean_path=out/'guarded.nii.gz'
        if raw_path.exists() or clean_path.exists():raise FileExistsError('Preserve completed predictions')
        start=time.perf_counter()
        predictor.predict_from_files([[str(image)]],[str(out/'raw')],save_probabilities=False,overwrite=False,
                                    num_processes_preprocessing=1,num_processes_segmentation_export=1)
        inference_seconds=time.perf_counter()-start;start=time.perf_counter()
        info=apply_file(image,raw_path,prior,clean_path);guard_seconds=time.perf_counter()-start
        # Reference is opened only after both raw and guarded outputs are saved.
        ci,ri,rawi,cleani=[sitk.ReadImage(str(p)) for p in [image,reference,raw_path,clean_path]]
        for candidate in [ri,rawi,cleani]:
            for field in ['GetSize','GetSpacing','GetOrigin','GetDirection']:
                assert np.allclose(getattr(candidate,field)(),getattr(ci,field)(),atol=1e-5)
        ref=sitk.GetArrayFromImage(ri)>0;raw=sitk.GetArrayFromImage(rawi)>0;clean=sitk.GetArrayFromImage(cleani)>0
        assert not np.any(clean&~raw)
        metrics={}
        distance=ndi.distance_transform_edt(~ref,sampling=ci.GetSpacing()[::-1])
        for name,pred,img in [('raw',raw,rawi),('guarded',clean,cleani)]:
            m=overlap_metrics(pred,ref,ci.GetSpacing());m.update(surface_metrics(pred,ref,ci.GetSpacing()))
            fp=pred&~ref;m['fp_beyond_10mm']=int(np.count_nonzero(fp&(distance>10)))
            m['fp_beyond_5mm']=int(np.count_nonzero(fp&(distance>5)))
            m['prediction_components_26']=int(cc3d.connected_components(pred,connectivity=26,return_N=True)[1])
            overlap=sitk.LabelOverlapMeasuresImageFilter();overlap.Execute(ri,img)
            assert abs(overlap.GetDiceCoefficient()-m['dice'])<1e-12
            assert abs(overlap.GetJaccardCoefficient()-m['iou'])<1e-12
            export_seg(img,out/(name+'.seg.nrrd'),case+' '+name)
            reread=sitk.ReadImage(str(out/(name+'.seg.nrrd')))
            assert np.array_equal(sitk.GetArrayFromImage(reread)>0,pred)
            metrics[name]=m
        del distance
        record=dict(case=case,role='development comparison, previously examined animal',old_two_case_raw=old[case],
            four_case=metrics,inference_seconds=inference_seconds,guard_seconds=guard_seconds,guard=info,
            raw_sha256=sha(raw_path),guarded_sha256=sha(clean_path),reference_sha256=sha(reference),
            tp_removed=metrics['raw']['tp']-metrics['guarded']['tp'],fp_removed=metrics['raw']['fp']-metrics['guarded']['fp'],
            far10_removed=metrics['raw']['fp_beyond_10mm']-metrics['guarded']['fp_beyond_10mm'],
            geometry_verified=True,nifti_nrrd_identical=True)
        (out/'metrics.json').write_text(json.dumps(record,indent=2),encoding='utf-8');all_records.append(record)
        print('FOURSHOT_CASE_EVALUATED',case,json.dumps({k:v['dice'] for k,v in metrics.items()}),flush=True)
    aggregate={}
    stages={'two_case_raw':[r['old_two_case_raw'] for r in all_records],
            'four_case_raw':[r['four_case']['raw'] for r in all_records],
            'four_case_guarded':[r['four_case']['guarded'] for r in all_records]}
    for stage,rows in stages.items():
        aggregate[stage]={}
        for key in ['dice','iou','precision','recall','absolute_volume_error_percent']:
            values=[r[key] for r in rows if r[key] is not None]
            aggregate[stage][key]=float(np.mean(values)) if values else None
    summary=dict(run=RUN,training_cases=TRAIN,evaluation_cases=TEST,cases=all_records,macro=aggregate,
        checkpoint=str(checkpoint_path),checkpoint_sha256=checkpoint_hash,spatial_prior_sha256=sha(prior),
        original_test_metrics_unchanged=sha(original_metrics_path)==original_hash,
        interpretation='Development comparison on three reused individuals; not independent new test performance. Same 30x50 training budget; spatial prior fits four training cases only.',
        warning='Position prior is only validated for the standardized flank-tumor cohort; ectopic lesions may be removed',
        prediction_device='cuda',peak_gpu_allocated_gb=torch.cuda.max_memory_allocated()/1024**3)
    assert summary['original_test_metrics_unchanged']
    (RESULTS/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('FOURSHOT_EVALUATION_COMPLETE',json.dumps(aggregate),flush=True)

if __name__=='__main__':main()
