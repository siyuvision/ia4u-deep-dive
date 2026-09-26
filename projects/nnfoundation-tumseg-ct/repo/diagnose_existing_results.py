"""Read-only prediction diagnosis. No training, inference, or prediction edits."""
import hashlib
import itertools
import json
from pathlib import Path
import cc3d
import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi
from project_env import PROJECT,DATASET

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dice(a,b):
    den=int(a.sum())+int(b.sum())
    return 2*int(np.count_nonzero(a&b))/den if den else 1.

def main():
    protocol=json.loads((PROJECT/'experiments/protocol.json').read_text())
    saved=json.loads((PROJECT/'results/pilot/metrics.json').read_text())
    rows=[];train=[]
    for case in protocol['training_cases']:
        im=sitk.ReadImage(str(PROJECT/'data/nnUNet_raw'/DATASET/'labelsTr'/(case+'.nii.gz')));mask=sitk.GetArrayFromImage(im)>0
        labels,n=cc3d.connected_components(mask,connectivity=26,return_N=True);counts=np.bincount(labels.ravel())[1:]
        train.append(dict(case=case,total_volume_mm3=float(mask.sum()*np.prod(im.GetSpacing())),
                          component_volumes_mm3=sorted((counts*np.prod(im.GetSpacing())).tolist(),reverse=True),
                          foreground_fraction=float(mask.mean())))
    for case in protocol['test_cases']:
        refpath=PROJECT/'data/test_reference/STAPLE'/(case+'.nii.gz')
        predpath=PROJECT/'results/pilot/predictions'/(case+'.nii.gz')
        predsha=sha(predpath)
        im=sitk.ReadImage(str(refpath));ref=sitk.GetArrayFromImage(im)>0;pred=sitk.GetArrayFromImage(sitk.ReadImage(str(predpath)))>0
        spacing=im.GetSpacing();voxel=float(np.prod(spacing))
        experts={}
        for name in ['Annotator_A','Annotator_B','Annotator_C']:
            experts[name]=sitk.GetArrayFromImage(sitk.ReadImage(str(PROJECT/'data/test_reference'/name/(case+'.nii.gz'))))>0
        pairs={a+'_vs_'+b:dice(experts[a],experts[b]) for a,b in itertools.combinations(experts,2)}
        pred_comp,n=cc3d.connected_components(pred,connectivity=26,return_N=True)
        counts=np.bincount(pred_comp.ravel());hits=np.bincount(pred_comp[ref],minlength=len(counts))
        unmatched_ids=np.flatnonzero((counts>0)&(hits==0));unmatched_ids=unmatched_ids[unmatched_ids!=0]
        unmatched=int(counts[unmatched_ids].sum())
        fp=pred&~ref;tp=int(np.count_nonzero(pred&ref));fn=int(np.count_nonzero(~pred&ref));fpn=int(fp.sum())
        distance=ndi.distance_transform_edt(~ref,sampling=spacing[::-1])
        far5=int(np.count_nonzero(fp&(distance>5)));far10=int(np.count_nonzero(fp&(distance>10)));del distance
        z=int(np.argmax(ref.sum(axis=(1,2))));slice_dice=dice(pred[z],ref[z])
        row=dict(case=case,whole_volume_dice=dice(pred,ref),reference_volume_mm3=float(ref.sum()*voxel),
            reference_fraction=float(ref.mean()),predicted_volume_mm3=float(pred.sum()*voxel),
            tp=tp,fp=fpn,fn=fn,fp_to_fn_ratio=fpn/fn if fn else None,
            tp_volume_mm3=tp*voxel,fp_volume_mm3=fpn*voxel,fn_volume_mm3=fn*voxel,
            precision=tp/int(pred.sum()) if pred.any() else None,recall=tp/int(ref.sum()),
            max_reference_axial_index=z,max_reference_axial_dice=slice_dice,
            annotator_pairwise_dice=pairs,mean_annotator_pairwise_dice=float(np.mean(list(pairs.values()))),
            annotator_volumes_mm3={k:float(v.sum()*voxel) for k,v in experts.items()},
            prediction_dice_vs_each_annotator={k:dice(pred,v) for k,v in experts.items()},
            prediction_components=int(n),unmatched_prediction_components=len(unmatched_ids),
            fp_in_components_with_no_reference_overlap=unmatched,
            unmatched_component_fraction_of_all_fp=unmatched/fpn if fpn else None,
            fp_more_than_5mm_from_reference=far5,fp_more_than_10mm_from_reference=far10,
            far5_fraction_of_all_fp=far5/fpn if fpn else None,far10_fraction_of_all_fp=far10/fpn if fpn else None,
            prediction_sha256=predsha,prediction_unchanged=sha(predpath)==predsha)
        expected=next(r['references']['STAPLE']['dice'] for r in saved['cases'] if r['case']==case)
        assert abs(row['whole_volume_dice']-expected)<1e-12 and row['prediction_unchanged']
        rows.append(row);print(json.dumps(row),flush=True)
    report=dict(brand='思语视觉',analysis='read-only diagnostic of already-evaluated predictions',
        training_cases=train,test_cases=rows,
        distance_definition='Euclidean physical distance to any reference foreground voxel, used only to describe errors; 5/10mm are diagnostic cutoffs, not postprocessing rules',
        component_definition='A component with zero reference overlap is diagnostic unmatched foreground; no mask is filtered or exported',
        limitations='Pairwise annotator agreement does not establish pathological truth; target slice was selected using reference and is not a blinded detectability study; causal effects of sample count, training budget or pretraining need controlled comparisons')
    (PROJECT/'experiments/visual-diagnosis.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('TRAINING_TARGETS',json.dumps(train),flush=True)

if __name__=='__main__':main()
