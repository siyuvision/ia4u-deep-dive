"""Real training CT/reference illustrations; never presented as model predictions."""
import json
import numpy as np
import SimpleITK as sitk
from project_env import PROJECT,DATASET
from render_assets import save_plane,save_projection

if __name__=='__main__':
    p=json.loads((PROJECT/'experiments/protocol.json').read_text());out=PROJECT/'results/training_inputs';out.mkdir(parents=True,exist_ok=True)
    records=[]
    for case in p['training_cases']:
        ci=sitk.ReadImage(str(PROJECT/'data/nnUNet_raw'/DATASET/'imagesTr'/(case+'_0000.nii.gz')))
        ri=sitk.ReadImage(str(PROJECT/'data/nnUNet_raw'/DATASET/'labelsTr'/(case+'.nii.gz')))
        ct=sitk.GetArrayFromImage(ci);ref=sitk.GetArrayFromImage(ri)>0;z=int(np.argmax(ref.sum(axis=(1,2))));spacing=ci.GetSpacing()
        extent=[0,ct.shape[2]*spacing[0],0,ct.shape[1]*spacing[1]]
        for mode in ['ct','reference']:
            save_plane(out/(case+'_axial_'+mode+'.png'),ct[z],ref[z],np.zeros_like(ref[z]),mode,case+' | TRAINING '+mode,extent)
        save_projection(out/(case+'_reference_portrait.png'),ct,ref,np.zeros_like(ref),spacing,case+' | TRAINING','reference')
        records.append(dict(case=case,axial_index=z,volume_mm3=float(ref.sum()*np.prod(spacing)),assets=[str(x.resolve()) for x in out.glob(case+'*.png')]))
    (out/'manifest.json').write_text(json.dumps(records,indent=2),encoding='utf-8');print(json.dumps(records),flush=True)
