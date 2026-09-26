"""Safe extraction and full-image audit of every scan and annotation in the supplied ZIP."""
from __future__ import annotations
import argparse
import os
import collections
import datetime as dt
import hashlib
import json
import re
import stat
import zipfile
from pathlib import Path,PurePosixPath
import numpy as np
import nibabel as nib
import cc3d

PROJECT=Path(__file__).resolve().parents[1]
ARCHIVE_SHA='1c0567358ec81b9e085434a0362c1c2981b791dc2733931aeb022174ecb80399'

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--archive', default=os.environ.get('TUMSEG_ARCHIVE'))
    args=ap.parse_args()
    if not args.archive:ap.error('Provide --archive PATH or TUMSEG_ARCHIVE')
    ARCHIVE=Path(args.archive).expanduser().resolve()
    if sha(ARCHIVE)!=ARCHIVE_SHA:raise ValueError('Archive differs from the published experiment input; do not silently reuse this protocol')
    (PROJECT/'experiments').mkdir(parents=True,exist_ok=True)
    dest=PROJECT/'data/input';dest.mkdir(parents=True,exist_ok=True)
    records=[]
    with zipfile.ZipFile(ARCHIVE) as z:
        names=set()
        for entry in z.infolist():
            name=entry.filename
            p=PurePosixPath(name)
            if p.is_absolute() or '..' in p.parts or '\\' in name or ':' in name or '\x00' in name:
                raise ValueError(f'Unsafe ZIP member: {name}')
            mode=entry.external_attr>>16
            if stat.S_ISLNK(mode):raise ValueError(f'Symlink ZIP member: {name}')
            target=(dest/Path(*p.parts)).resolve()
            if not target.is_relative_to(dest.resolve()):raise ValueError(name)
            if name in names:raise ValueError(f'Duplicate ZIP member {name}')
            names.add(name)
            if entry.is_dir():target.mkdir(parents=True,exist_ok=True);continue
            target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():
                raise FileExistsError('Audit extraction is immutable; choose a new destination if rerunning')
            data=z.read(entry)  # zipfile validates CRC while reading each complete member.
            assert len(data)==entry.file_size
            target.write_bytes(data)
            records.append(dict(member=name,bytes=entry.file_size,crc32=f'{entry.CRC:08x}',sha256=hashlib.sha256(data).hexdigest()))
    (PROJECT/'experiments/archive-manifest.json').write_text(json.dumps(dict(archive=str(ARCHIVE),bytes=ARCHIVE.stat().st_size,
        sha256=sha(ARCHIVE),checked_crc_all_members=True,entries=records),indent=2),encoding='utf-8')
    print('EXTRACTED',len(records),'files',flush=True)
    image_records=[];errors=[]
    files=sorted(dest.rglob('CT_*.nii.gz'))
    for index,path in enumerate(files):
        rel=path.relative_to(dest);dataset=next(x for x in rel.parts if x.startswith('Dataset '))
        subject_time=path.name.removeprefix('CT_').removesuffix('.nii.gz')
        match=re.fullmatch(r'(M\d+)_(.+)',subject_time)
        if not match:raise ValueError(subject_time)
        animal,time=match.groups();subject=f'D{int(dataset.split()[1]):02d}_{animal}'
        case=f'{subject}_{time}'
        ct=nib.load(path);arr=np.asanyarray(ct.dataobj)
        annotations={}
        for source in ['Annotator_A','Annotator_B','Annotator_C','STAPLE']:
            ap=path.with_name(source+'_'+subject_time+'.nii.gz')
            if not ap.exists():errors.append(f'Missing {ap}');continue
            img=nib.load(ap);mask=np.asanyarray(img.dataobj)
            values,counts=np.unique(mask,return_counts=True)
            same_geometry=ct.shape==img.shape and np.allclose(ct.affine,img.affine,atol=1e-4,rtol=0)
            if not same_geometry:errors.append(f'Geometry mismatch: {case} {source}')
            binary=mask>0
            labels,n=cc3d.connected_components(binary,connectivity=26,return_N=True)
            sizes=np.bincount(labels.ravel());sizes[0]=0
            annotations[source]=dict(path=str(ap.relative_to(PROJECT)),dtype=str(mask.dtype),
                values=values.tolist() if len(values)<20 else dict(count=len(values),min=float(values[0]),max=float(values[-1])),
                value_counts=counts.tolist() if len(values)<20 else None,
                foreground_voxels=int(binary.sum()),volume_mm3=float(binary.sum()*abs(np.linalg.det(img.affine[:3,:3]))),
                components_26=int(n),largest_components_voxels=sorted(sizes[1:].tolist(),reverse=True)[:10],
                geometry_matches_ct=bool(same_geometry),finite=bool(np.isfinite(mask).all()))
        sampling=arr.ravel()[::max(1,arr.size//100000)]
        image_records.append(dict(case=case,dataset=dataset,individual_id=subject,animal_id_within_dataset=animal,
            timepoint=time,image=str(path.relative_to(PROJECT)),shape_xyz=list(ct.shape),spacing_xyz_mm=[float(x) for x in ct.header.get_zooms()],
            affine=ct.affine.tolist(),dtype=str(arr.dtype),finite=bool(np.isfinite(arr).all()),
            intensity_min=float(arr.min()),intensity_max=float(arr.max()),intensity_sample_percentiles=np.percentile(sampling,[0,1,50,99,100]).tolist(),
            annotations=annotations))
        if index%25==0:print('AUDIT',index+1,'/',len(files),case,flush=True)
    counts=collections.Counter(r['dataset'] for r in image_records)
    subjects=collections.defaultdict(list)
    for r in image_records:subjects[r['individual_id']].append(r['case'])
    summary=dict(scans=len(image_records),individuals=len(subjects),dataset_scan_counts=dict(counts),
        dataset_individual_counts={d:len({r['individual_id'] for r in image_records if r['dataset']==d}) for d in counts},
        individual_grouping='dataset + animal identifier; all timepoints of an individual must stay in one split',
        timepoint_counts=dict(collections.Counter(r['timepoint'] for r in image_records)),
        annotator_files_per_scan=4,errors=errors,
        annotation_values={s:sorted({str(r['annotations'][s]['values']) for r in image_records}) for s in ['Annotator_A','Annotator_B','Annotator_C','STAPLE']},
        empty_staple_cases=[r['case'] for r in image_records if r['annotations']['STAPLE']['foreground_voxels']==0],
        geometry_all_match=not any('Geometry' in e for e in errors))
    result=dict(created=dt.datetime.now().astimezone().isoformat(),summary=summary,individuals=dict(subjects),scans=image_records)
    (PROJECT/'experiments/data-audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('AUDIT_COMPLETE',json.dumps(summary),flush=True)

if __name__=='__main__':main()
