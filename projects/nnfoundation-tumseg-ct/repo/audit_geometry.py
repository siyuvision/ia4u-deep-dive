"""Cross-check qform/sform/pixdim in every supplied image and annotation."""
import json
from pathlib import Path
import nibabel as nib
import numpy as np
from project_env import PROJECT

def main():
    rows=[];errors=[]
    for path in sorted((PROJECT/'data/input').rglob('*.nii.gz')):
        img=nib.load(path);q=img.get_qform();s=img.get_sform();spacing=img.header.get_zooms()
        qspacing=np.linalg.norm(q[:3,:3],axis=0);sspacing=np.linalg.norm(s[:3,:3],axis=0)
        verified_q=bool(np.allclose(qspacing,spacing,atol=1e-6) and np.allclose(spacing,[.21]*3,atol=1e-6))
        mismatch=not np.allclose(q,s,atol=1e-6)
        if not verified_q:errors.append(str(path))
        rows.append(dict(path=str(path.relative_to(PROJECT)),qform_code=int(img.header['qform_code']),sform_code=int(img.header['sform_code']),
             qform=q.tolist(),sform=s.tolist(),pixdim_xyz_mm=[float(x) for x in spacing],qform_matches_spacing_and_paper=verified_q,sform_qform_disagree=mismatch))
    summary=dict(files=len(rows),qform_valid_count=sum(r['qform_matches_spacing_and_paper'] for r in rows),
        sform_qform_mismatch_count=sum(r['sform_qform_disagree'] for r in rows),errors=errors,
        resolution='For experiment copies, use the documented 0.21mm qform for both qform and sform; no resampling or voxel edits')
    (PROJECT/'experiments/geometry-audit.json').write_text(json.dumps(dict(summary=summary,files=rows),indent=2),encoding='utf-8')
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
