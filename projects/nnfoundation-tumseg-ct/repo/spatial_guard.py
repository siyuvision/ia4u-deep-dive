"""Train-fitted longitudinal support for this standardized single-mouse flank-tumor cohort.

This is not a general tumor remover: unseen locations, orientations and cohorts need validation.
Fitting reads training annotations. Application reads only CT, prediction and the frozen prior.
"""
import json
from pathlib import Path
import cc3d
import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi

METHOD='ct_body_relative_longitudinal_component_guard_v1'

def body_extent(ct,spacing):
    # Fixed soft-tissue window, followed by a modest opening to remove thin support structures.
    spacing=np.asarray(spacing,dtype=float)
    grid=np.stack(np.meshgrid(*[np.arange(-2,3)]*3,indexing='ij'),axis=-1)*spacing[::-1]
    structure=np.sum(grid**2,axis=-1)<=.42**2+1e-9
    tissue=(ct>-250)&(ct<300)
    opened=ndi.binary_opening(tissue,structure=structure)
    labels,n=cc3d.connected_components(opened,connectivity=26,return_N=True)
    if not n:raise ValueError('Cannot identify a soft-tissue body component')
    sizes=np.bincount(labels.ravel());sizes[0]=0;body_id=int(np.argmax(sizes))
    bbox=cc3d.statistics(labels)['bounding_boxes'][body_id]
    lower=int(bbox[0].start);upper=int(bbox[0].stop-1)
    if (upper-lower+1)*spacing[2]<30 or sizes[body_id]*float(np.prod(spacing))<1000:
        raise ValueError('Body localization outside validated mouse-size range; review CT orientation/FOV')
    return dict(z_min=lower,z_max=upper,z_length_mm=(upper-lower)*float(spacing[2]),
                body_voxels=int(sizes[body_id]),body_volume_mm3=float(sizes[body_id]*np.prod(spacing)),
                body_bbox_zyx=[[s.start,s.stop] for s in bbox])

def fit_prior(training_pairs,margin_mm=5.0,min_volume_fraction=.1):
    records=[];directions=[]
    for case,image_path,label_path in training_pairs:
        image=sitk.ReadImage(str(image_path));label=sitk.ReadImage(str(label_path))
        for field in ['GetSize','GetSpacing','GetDirection','GetOrigin']:
            if not np.allclose(getattr(image,field)(),getattr(label,field)(),atol=1e-5):raise ValueError(case+' geometry')
        ct=sitk.GetArrayFromImage(image);mask=sitk.GetArrayFromImage(label)>0
        if not mask.any():raise ValueError('Positive training target required: '+case)
        extent=body_extent(ct,image.GetSpacing());span=extent['z_max']-extent['z_min']
        occupied=np.flatnonzero(mask.any(axis=(1,2)))
        lo=(int(occupied[0])-extent['z_min'])/span;hi=(int(occupied[-1])-extent['z_min'])/span
        if not (0<=lo<=hi<=1):raise ValueError('Training target outside estimated body: '+case)
        components,n=cc3d.connected_components(mask,connectivity=26,return_N=True)
        volumes=np.bincount(components.ravel())[1:]*np.prod(image.GetSpacing())
        records.append(dict(case=case,body=extent,target_z_min=int(occupied[0]),target_z_max=int(occupied[-1]),
                            target_relative_z_min=lo,target_relative_z_max=hi,component_volumes_mm3=volumes.tolist()))
        directions.append(list(image.GetDirection()))
    assert all(np.allclose(d,directions[0]) for d in directions)
    return dict(method=METHOD,trained_on=[r['case'] for r in records],
                z_min_relative=min(r['target_relative_z_min'] for r in records),
                z_max_relative=max(r['target_relative_z_max'] for r in records),
                margin_mm=margin_mm,min_component_volume_mm3=min(v for r in records for v in r['component_volumes_mm3'])*min_volume_fraction,
                min_volume_fraction=min_volume_fraction,expected_direction=directions[0],training_records=records,
                applicability='Dataset 10 baseline, standardized single-mouse orientation and subcutaneous flank tumors only',
                decision='Keep whole predicted components whose centroid is in the train-derived body-relative longitudinal interval and volume exceeds the conservative training-derived floor; no top-K restriction',
                uses_test_labels=False)

def apply_prior(ct,pred,spacing,direction,prior):
    if ct.shape!=pred.shape or ct.ndim!=3:raise ValueError('CT and mask must have the same 3D grid')
    if not np.allclose(direction,prior['expected_direction'],atol=1e-6):raise ValueError('Orientation is outside the fitted protocol')
    if not np.isin(pred,[0,1]).all():raise ValueError('Binary mask required')
    extent=body_extent(ct,spacing);span=extent['z_max']-extent['z_min'];voxel_mm3=float(np.prod(spacing))
    lower=extent['z_min']+prior['z_min_relative']*span-prior['margin_mm']/spacing[2]
    upper=extent['z_min']+prior['z_max_relative']*span+prior['margin_mm']/spacing[2]
    labels,n=cc3d.connected_components(pred>0,connectivity=26,return_N=True)
    stats=cc3d.statistics(labels);keep=np.zeros(n+1,dtype=bool);components=[]
    for i in range(1,n+1):
        volume=float(stats['voxel_counts'][i]*voxel_mm3);center=float(stats['centroids'][i][0])
        inside=lower<=center<=upper;large=volume>=prior['min_component_volume_mm3']
        keep[i]=inside and large
        components.append(dict(id=i,volume_mm3=volume,centroid_z=center,within_position_interval=inside,above_volume_floor=large,kept=bool(keep[i])))
    cleaned=keep[labels].astype(np.uint8)
    return cleaned,dict(method=METHOD,body=extent,allowed_z_centroid_interval=[lower,upper],
        components=components,raw_components=int(n),kept_components=int(keep.sum()),
        raw_foreground_voxels=int(np.count_nonzero(pred)),retained_foreground_voxels=int(cleaned.sum()),
        prediction_empty_after=not bool(cleaned.any()),uses_reference_label_at_application=False,
        warning='This prior can remove true ectopic tumors; never claim applicability to other anatomical locations or scan orientations')

def apply_file(image_path,prediction_path,prior_path,output_path):
    prior=json.loads(Path(prior_path).read_text());image=sitk.ReadImage(str(image_path));pred=sitk.ReadImage(str(prediction_path))
    for field in ['GetSize','GetSpacing','GetDirection','GetOrigin']:
        if not np.allclose(getattr(image,field)(),getattr(pred,field)(),atol=1e-5):raise ValueError('Prediction geometry mismatch')
    clean,info=apply_prior(sitk.GetArrayFromImage(image),sitk.GetArrayFromImage(pred),image.GetSpacing(),image.GetDirection(),prior)
    output=sitk.GetImageFromArray(clean);output.CopyInformation(pred)
    path=Path(output_path)
    if path.exists():raise FileExistsError(path)
    path.parent.mkdir(parents=True,exist_ok=True);sitk.WriteImage(output,str(path),True)
    path.with_suffix('.guard.json').write_text(json.dumps(info,indent=2),encoding='utf-8')
    return info
