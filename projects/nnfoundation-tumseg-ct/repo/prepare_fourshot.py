"""Freeze the user-selected four/three development split and train-only spatial prior."""
import datetime as dt
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
import SimpleITK as sitk
from fourshot_config import *
from spatial_guard import fit_prior,apply_prior

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    configure();EXP.mkdir(parents=True,exist_ok=True);RESULTS.mkdir(parents=True,exist_ok=True)
    if (EXP/'protocol.json').exists():raise FileExistsError('Protocol is already frozen')
    old=json.loads((PROJECT/'experiments/protocol.json').read_text())
    assert set(TRAIN+TEST)==set(old['training_cases']+old['test_cases']) and set(TRAIN).isdisjoint(TEST)
    assert sha(WEIGHTS)==WEIGHTS_SHA
    pairs=[(case,*sources(case)) for case in TRAIN]
    prior=fit_prior(pairs,margin_mm=5.,min_volume_fraction=.1)
    # Validate only on the four training annotations, leaving each animal out of prior fitting in turn.
    checks=[]
    for case,image_path,label_path in pairs:
        loo=fit_prior([p for p in pairs if p[0]!=case],margin_mm=5.,min_volume_fraction=.1)
        image=sitk.ReadImage(str(image_path));mask=sitk.ReadImage(str(label_path))
        arr=sitk.GetArrayFromImage(mask)>0
        clean,info=apply_prior(sitk.GetArrayFromImage(image),arr.astype(np.uint8),image.GetSpacing(),image.GetDirection(),loo)
        retained=float(clean.sum()/arr.sum());checks.append(dict(case=case,target_retention=retained,kept_components=info['kept_components'],body=info['body']))
        assert retained>=.995,(case,retained)
    prior.update(training_annotation_leave_one_out=checks,validation='prior preserves at least 99.5% of each excluded training annotation; not a validation of segmentation generalization')
    (EXP/'spatial-prior.json').write_text(json.dumps(prior,indent=2),encoding='utf-8')
    raw=PROJECT/'data/nnUNet_raw'/DATASET;manifest=[]
    for case in TRAIN+TEST:
        image,label=sources(case);im_out=raw/('imagesTr' if case in TRAIN else 'imagesTs')/(case+'_0000.nii.gz')
        im_out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(image,im_out)
        assert sha(image)==sha(im_out)
        record=dict(case=case,image=str(im_out.relative_to(PROJECT)),image_sha256=sha(image),reference=str(label.relative_to(PROJECT)),reference_sha256=sha(label))
        if case in TRAIN:
            label_out=raw/'labelsTr'/(case+'.nii.gz');label_out.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(label,label_out)
            assert sha(label_out)==sha(label)
        manifest.append(record)
    (raw/'dataset.json').write_text(json.dumps(dict(channel_names={'0':'CT'},labels={'background':0,'tumor':1},numTraining=4,file_ending='.nii.gz',name=DATASET),indent=2),encoding='utf-8')
    protocol=dict(run=RUN,frozen_at=dt.datetime.now().astimezone().isoformat(),dataset=DATASET,training_cases=TRAIN,evaluation_cases=TEST,
        evaluation_role='development comparison: all seven individuals and previous predictions were already inspected',
        split_choice='user explicitly chose reuse of the current seven mice, four train / three evaluate; first four in original seeded order become training',
        initialization=str(WEIGHTS),initialization_sha256=WEIGHTS_SHA,seed=20260926,
        epochs=30,iterations_per_epoch=50,optimizer_steps=1500,batch_size=2,patch_size=[128]*3,initial_learning_rate=.001,warmup_epochs=3,
        checkpoint_selection='fixed final checkpoint; no evaluation-driven checkpoint choice',
        reference='unchanged STAPLE consensus',postprocessing_prior=str((EXP/'spatial-prior.json').relative_to(PROJECT)),
        postprocessing_prior_sha256=sha(EXP/'spatial-prior.json'),postprocessing_fitted_only_on=TRAIN,
        fixed_guard_constants=dict(soft_tissue_hu=[-250,300],opening_radius_mm=.42,position_margin_mm=5.,minimum_volume_fraction=.1),
        comparisons=['original two-case model raw predictions on the same three evaluation individuals','new four-case raw predictions','new four-case plus fixed spatial guard'],
        no_test_label_at_postprocess_application=True,guard_limit='cohort-specific position prior; possible loss of ectopic or unusually tiny tumors; not a universal tumor postprocessor',
        data=manifest)
    (EXP/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8');(EXP/'protocol.sha256').write_text(sha(EXP/'protocol.json')+'\n')
    print('FOURSHOT_PROTOCOL_FROZEN',json.dumps(dict(train=TRAIN,evaluate=TEST,z_prior=[prior['z_min_relative'],prior['z_max_relative']],margin_mm=prior['margin_mm'],minimum_volume_mm3=prior['min_component_volume_mm3'],training_loo=checks)),flush=True)

if __name__=='__main__':main()
