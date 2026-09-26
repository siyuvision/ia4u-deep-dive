"""Evaluate all frozen unseen cases, preserving empty-target and multi-tumor semantics."""
import json
from pathlib import Path
import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi
import cc3d
from project_env import PROJECT,DATASET

def overlap_metrics(pred,ref,spacing):
    p,r=pred.astype(bool),ref.astype(bool)
    tp=int(np.count_nonzero(p&r));fp=int(np.count_nonzero(p&~r));fn=int(np.count_nonzero(~p&r))
    pn=tp+fp;rn=tp+fn;v=float(np.prod(spacing))
    both_empty=(pn+rn)==0
    return dict(dice=1. if both_empty else 2*tp/(pn+rn),iou=1. if both_empty else tp/(tp+fp+fn),
        precision=tp/pn if pn else None,recall=tp/rn if rn else None,
        reference_empty=rn==0,prediction_empty=pn==0,
        reference_voxels=rn,predicted_voxels=pn,tp=tp,fp=fp,fn=fn,
        reference_volume_mm3=rn*v,predicted_volume_mm3=pn*v,absolute_volume_error_mm3=abs(pn-rn)*v,
        signed_volume_error_percent=100*(pn-rn)/rn if rn else None,
        absolute_volume_error_percent=100*abs(pn-rn)/rn if rn else None)

def surface_metrics(pred,ref,spacing):
    if not np.any(pred) or not np.any(ref):
        return dict(hd95_pooled_boundary_voxels_mm=None,mean_symmetric_boundary_voxel_distance_mm=None,
                    status='undefined: one or both masks empty')
    p=pred.astype(bool);r=ref.astype(bool);structure=ndi.generate_binary_structure(3,1)
    pb=p&~ndi.binary_erosion(p,structure=structure,border_value=0)
    rb=r&~ndi.binary_erosion(r,structure=structure,border_value=0)
    distance=ndi.distance_transform_edt(~rb,sampling=spacing[::-1]);a=distance[pb];del distance
    distance=ndi.distance_transform_edt(~pb,sampling=spacing[::-1]);b=distance[rb];del distance
    pooled=np.concatenate([a,b])
    return dict(hd95_pooled_boundary_voxels_mm=float(np.percentile(pooled,95)),
        mean_symmetric_boundary_voxel_distance_mm=float(pooled.mean()),status='defined; voxel-boundary, not area-weighted NSD')

def main():
    protocol=json.loads((PROJECT/'experiments/protocol.json').read_text());out=PROJECT/'results/pilot';reports=[]
    inference=json.loads((PROJECT/'experiments/inference-pilot.json').read_text())
    for case in protocol['test_cases']:
        path=out/'predictions'/(case+'.nii.gz');pi=sitk.ReadImage(str(path));pred=sitk.GetArrayFromImage(pi)
        exported=sitk.ReadImage(str(out/'predictions'/(case+'.seg.nrrd')))
        assert np.array_equal(pred,sitk.GetArrayFromImage(exported))
        assert set(np.unique(pred)).issubset({0,1})
        refs={}
        for source in ['STAPLE','Annotator_A','Annotator_B','Annotator_C']:
            reference=PROJECT/'data/test_reference'/source/(case+'.nii.gz');ri=sitk.ReadImage(str(reference));ref=sitk.GetArrayFromImage(ri)
            for attr in ['GetSize','GetSpacing','GetOrigin','GetDirection']:
                assert np.allclose(getattr(pi,attr)(),getattr(ri,attr)(),atol=1e-6),(case,source,attr)
            metrics=overlap_metrics(pred,ref,pi.GetSpacing())
            if source=='STAPLE':
                metrics.update(surface_metrics(pred,ref,pi.GetSpacing()))
                if np.any(pred) and np.any(ref):
                    check=sitk.LabelOverlapMeasuresImageFilter();check.Execute(ri,pi)
                    assert abs(check.GetDiceCoefficient()-metrics['dice'])<1e-12
                    assert abs(check.GetJaccardCoefficient()-metrics['iou'])<1e-12
                metrics['reference_components_26']=int(cc3d.connected_components(ref>0,connectivity=26,return_N=True)[1])
            refs[source]=metrics
        record=dict(case=case,individual=case.rsplit('_',1)[0],references=refs,
                    prediction_components_26=int(cc3d.connected_components(pred>0,connectivity=26,return_N=True)[1]),
                    prediction=str(path),geometry_verified=True,nifti_nrrd_identical=True,
                    inference_seconds=next(r['seconds'] for r in inference['cases'] if r['case']==case))
        reports.append(record);print('CASE_METRICS',case,json.dumps(refs['STAPLE']),flush=True)
    aggregate={}
    for source in ['STAPLE','Annotator_A','Annotator_B','Annotator_C']:
        aggregate[source]={}
        for key in ['dice','iou','precision','recall','absolute_volume_error_percent']:
            values=[r['references'][source][key] for r in reports if r['references'][source][key] is not None]
            aggregate[source][key]=dict(mean=float(np.mean(values)) if values else None,
                median=float(np.median(values)) if values else None,min=float(np.min(values)) if values else None,
                max=float(np.max(values)) if values else None,defined_cases=len(values))
    worst=min(reports,key=lambda r:r['references']['STAPLE']['dice'])['case']
    summary=dict(protocol=str(PROJECT/'experiments/protocol.json'),training_individuals=2,training_scans=2,
        test_individuals=5,test_scans=5,cases=reports,macro_by_reference=aggregate,
        representative_test_case=protocol['representative_test_case'],worst_dice_case=worst,
        evaluation='single frozen subset in Dataset 10; all five previously unseen individuals; no post-test changes',
        postprocessing='none',empty_policy='Dice/IoU both empty=1, one empty=0; undefined precision/recall/relative volume/boundary values are null',
        limits='Two training examples and five tests in one cohort cannot establish generalization or pretraining improvement; no scratch baseline')
    (out/'metrics.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('EVALUATION_COMPLETE',json.dumps(aggregate['STAPLE']),flush=True)

if __name__=='__main__':main()
