"""Freeze scan/animal/co-acquisition disjoint split and normalize NIfTI metadata only."""
import datetime as dt
import collections
import hashlib
import json
from pathlib import Path
import nibabel as nib
import numpy as np
from project_env import PROJECT,DATASET,WEIGHTS,WEIGHTS_SHA

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def write_consistent_image(source,dest,is_mask=False):
    img=nib.load(source);arr=np.asanyarray(img.dataobj)
    if is_mask:
        assert set(np.unique(arr)).issubset({0,1})
        arr=arr.astype(np.uint8)
    qform=img.get_qform()
    assert np.allclose(np.linalg.norm(qform[:3,:3],axis=0),[.21]*3,atol=1e-6)
    header=img.header.copy();header.set_xyzt_units('mm')
    fixed=nib.Nifti1Image(arr,qform,header=header)
    fixed.set_qform(qform,code=1);fixed.set_sform(qform,code=1)
    fixed.set_data_dtype(np.uint8 if is_mask else arr.dtype)
    dest.parent.mkdir(parents=True,exist_ok=True);nib.save(fixed,dest)
    reread=nib.load(dest)
    assert np.array_equal(np.asanyarray(reread.dataobj),arr)
    assert np.allclose(reread.get_qform(),reread.get_sform(),atol=1e-7)
    return dict(source=str(source.relative_to(PROJECT)),source_sha256=sha(source),
                output=str(dest.relative_to(PROJECT)),output_sha256=sha(dest),
                voxels_unchanged=True,shape=list(arr.shape),spacing_mm=list(map(float,reread.header.get_zooms())))

def main():
    audit=json.loads((PROJECT/'experiments/data-audit.json').read_text())
    geom=json.loads((PROJECT/'experiments/geometry-audit.json').read_text())
    assert not audit['summary']['errors'] and not geom['summary']['errors']
    assert audit['summary']['scans']==452 and audit['summary']['individuals']==223
    assert sha(WEIGHTS)==WEIGHTS_SHA
    # The initial audit explicitly inspected both encodings; volume must follow qform/pixdim,
    # not nibabel's preferred but inconsistent unit-scale sform.
    for row in audit['scans']:
        # Dataset 1 filenames shorten 22.5h to '22'; parent directory carries the full timepoint.
        row['filename_timepoint']=row['timepoint']
        folder_time=Path(row['image']).parent.name.split('_',1)[1]
        row['timepoint']=folder_time
        row['case']=row['individual_id']+'_'+folder_time.replace('.','p')
        for annotation in row['annotations'].values():
            annotation['volume_from_inconsistent_sform_mm3']=annotation.pop('volume_mm3')
            annotation['volume_mm3']=annotation['foreground_voxels']*float(np.prod(row['spacing_xyz_mm']))
    audit['summary']['physical_volume_basis']='documented 0.21mm pixdim/qform; sform conflict separately audited'
    audit['summary']['timepoint_counts']=dict(collections.Counter(r['timepoint'] for r in audit['scans']))
    audit['summary']['filename_timepoint_discrepancies']=sum(r['filename_timepoint']!=r['timepoint'] for r in audit['scans'])
    audit['individuals']={key:[r['case'] for r in audit['scans'] if r['individual_id']==key] for key in audit['individuals']}
    (PROJECT/'experiments/data-audit-physical.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    eligible=[r for r in audit['scans'] if r['dataset']=='Dataset 10' and r['timepoint']=='0d']
    groups={}
    for row in eligible:
        group=Path(row['image']).parent.parent.name
        groups.setdefault(group,[]).append(row)
    ordered=sorted(groups,key=lambda g:hashlib.sha256(('20260926|'+g).encode()).hexdigest())
    chosen=[min(groups[g],key=lambda r:int(r['animal_id_within_dataset'][1:])) for g in ordered[:7]]
    train=chosen[:2];test=chosen[2:7]
    assert len({r['individual_id'] for r in chosen})==7
    assert len({Path(r['image']).parent.parent.name for r in chosen})==7
    protocol=dict(frozen_at=dt.datetime.now().astimezone().isoformat(),dataset_id=DATASET,
        input='single-channel cropped whole-mouse micro-CT; tumor only is foreground',labels={'background':0,'all_subcutaneous_tumors':1},
        reference='supplied binary STAPLE consensus; per-annotator A/B/C overlap is secondary evaluation',
        cohort='Dataset 10 baseline (0d) only: holds acquisition domain fixed for a two-example feasibility experiment',
        selection_rule='Sort 0d co-acquisition groups by SHA256(20260926|group); choose lowest numeric mouse ID in each group; first 2 train, next 5 test',
        archive_scans=452,archive_individuals=223,eligible_scans=len(eligible),eligible_co_acquisition_groups=len(groups),
        training_cases=[r['case'] for r in train],test_cases=[r['case'] for r in test],
        training_individuals=[r['individual_id'] for r in train],test_individuals=[r['individual_id'] for r in test],
        all_other_timepoints_excluded=True,co_acquisition_groups_disjoint=True,
        representative_test_case=test[0]['case'],representative_policy='first test case in frozen selection order, not selected by model performance',
        initialization=str(WEIGHTS),initialization_sha256=WEIGHTS_SHA,
        epochs=30,iterations_per_epoch=50,total_optimizer_steps=1500,batch_size=2,patch_size=[128]*3,
        warmup_epochs=3,initial_learning_rate=.001,optimizer='SGD momentum=.99 nesterov',
        preprocessing='official nnssl adaptation with default_nnunet spacing, ZScoreNormalization, fingerprint from two train scans only',
        geometry_resolution='qform and sform aligned to original qform, 0.21mm isotropic; voxel arrays unchanged',
        checkpoint_selection='fixed final checkpoint after 30 epochs; monitoring only on training cases',
        postprocessing='none; no largest-component selection because Dataset 10 has bilateral tumors',
        evaluation='all five predetermined individual scans; original voxel grid/physical qform coordinates; empty-target policy explicit',
        scans=chosen)
    pp=PROJECT/'experiments/protocol.json'
    if pp.exists():raise FileExistsError('Protocol already frozen')
    pp.write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    (PROJECT/'experiments/protocol.sha256').write_text(sha(pp)+'\n')
    manifest=[];raw=PROJECT/'data/nnUNet_raw'/DATASET
    for row in chosen:
        case=row['case'];is_train=case in protocol['training_cases']
        imageout=raw/('imagesTr' if is_train else 'imagesTs')/(case+'_0000.nii.gz')
        manifest.append(write_consistent_image(PROJECT/row['image'],imageout))
        refs=raw/'labelsTr' if is_train else PROJECT/'data/test_reference/STAPLE'
        manifest.append(write_consistent_image(PROJECT/row['annotations']['STAPLE']['path'],refs/(case+'.nii.gz'),True))
        if not is_train:
            for source in ['Annotator_A','Annotator_B','Annotator_C']:
                manifest.append(write_consistent_image(PROJECT/row['annotations'][source]['path'],PROJECT/'data/test_reference'/source/(case+'.nii.gz'),True))
    dataset=dict(channel_names={'0':'CT'},labels={'background':0,'tumor':1},numTraining=2,file_ending='.nii.gz',name=DATASET)
    (raw/'dataset.json').write_text(json.dumps(dataset,indent=2),encoding='utf-8')
    (PROJECT/'experiments/conversion-manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('PROTOCOL_FROZEN',json.dumps({k:protocol[k] for k in ['training_cases','test_cases','representative_test_case','eligible_scans','eligible_co_acquisition_groups']}),flush=True)

if __name__=='__main__':main()
