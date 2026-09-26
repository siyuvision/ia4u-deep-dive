"""Generate source-backed scientific PNG assets, with fixed windows and honest failures."""
import json
from pathlib import Path
import numpy as np
import SimpleITK as sitk
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes
from project_env import PROJECT,DATASET

BG='#080d16';FG='#e7edf4';REF='#22d3d8';PRED='#ff9b43';FP='#ff604f';FN='#6094ff';TP='#39c692'
plt.rcParams.update({'font.family':'DejaVu Sans','figure.facecolor':BG,'axes.facecolor':BG,
                     'text.color':FG,'axes.labelcolor':FG,'xtick.color':FG,'ytick.color':FG,'axes.edgecolor':'#40516a'})

def save_plane(path,ct,ref,pred,mode,title,extent):
    fig=plt.figure(figsize=(10.8,10.8),dpi=100);ax=fig.add_axes([.065,.13,.87,.78])
    ax.imshow(ct,cmap='gray',vmin=-300,vmax=300,origin='lower',extent=extent,interpolation='nearest')
    handles=[]
    if mode in ['reference','prediction','both']:
        for mask,color,label in [(ref,REF,'STAPLE reference'),(pred,PRED,'Prediction')]:
            if (mode=='reference' and label=='Prediction') or (mode=='prediction' and label=='STAPLE reference'):continue
            if mask.any():
                overlay=np.zeros((*mask.shape,4));from matplotlib.colors import to_rgb
                overlay[mask]=(*to_rgb(color),.30)
                ax.imshow(overlay,origin='lower',extent=extent,interpolation='nearest')
                ax.contour(mask,levels=[.5],colors=[color],linewidths=1.4,origin='lower',extent=extent)
            handles.append(Patch(color=color,label=label))
    if mode=='error':
        rgba=np.zeros((*ref.shape,4))
        from matplotlib.colors import to_rgb
        for mask,color,label in [(ref&pred,TP,'Overlap (TP)'),(pred&~ref,FP,'False positive'),(ref&~pred,FN,'Missed target')]:
            rgba[mask]=(*to_rgb(color),.6);handles.append(Patch(color=color,label=label))
        ax.imshow(rgba,origin='lower',extent=extent,interpolation='nearest')
    ax.set_aspect('equal');ax.axis('off')
    # 10mm bar, fixed physical units independent of image resolution.
    x=extent[0]+.07*(extent[1]-extent[0]);y=extent[2]+.07*(extent[3]-extent[2])
    ax.plot([x,x+10],[y,y],color='white',linewidth=3);ax.text(x+5,y+1.0,'10 mm',ha='center',fontsize=14,color='white')
    fig.text(.5,.95,title,ha='center',va='top',fontsize=24,weight='bold')
    if handles:fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.045),ncol=len(handles),frameon=False,fontsize=14,labelcolor=FG)
    fig.text(.5,.018,'CT window: -300 to 300 HU | Full image plane',ha='center',fontsize=13,color='#9aadc5')
    fig.savefig(path,dpi=100,facecolor=BG);plt.close(fig)

def save_projection(path,ct,ref,pred,spacing,case,mode,axis=1):
    fig=plt.figure(figsize=(10.8,19.2),dpi=100);ax=fig.add_axes([.07,.10,.86,.79])
    plane=ct.max(axis=axis);ps=np.delete(np.array(spacing[::-1]),axis)
    extent=[0,plane.shape[1]*ps[1],0,plane.shape[0]*ps[0]]
    ax.imshow(plane,cmap='gray',vmin=-400,vmax=1000,origin='lower',extent=extent)
    if mode!='ct':
        masks=[(ref,REF,'STAPLE')] if mode=='reference' else [(pred,PRED,'Prediction')]
        if mode=='both':masks=[(ref,REF,'STAPLE'),(pred,PRED,'Prediction')]
        for mask,color,label in masks:
            plane=mask.any(axis=axis)
            if plane.any():ax.contour(plane,levels=[.5],colors=[color],linewidths=1.7,origin='lower',extent=extent)
        fig.legend(handles=[Patch(color=c,label=l) for _,c,l in masks],loc='lower center',bbox_to_anchor=(.5,.05),ncol=len(masks),frameon=False,fontsize=19,labelcolor=FG)
    ax.set_aspect('equal');ax.axis('off')
    fig.text(.5,.955,case,ha='center',fontsize=27,weight='bold')
    fig.text(.5,.926,f'Whole mouse CT | axis {axis} | '+mode,ha='center',fontsize=20,color='#b8c9df')
    fig.text(.5,.02,'Full-volume projection | CT window -400 to 1000 HU',ha='center',fontsize=14,color='#9aadc5')
    fig.savefig(path,dpi=100,facecolor=BG);plt.close(fig)

def save_surface(path,ref,pred,spacing,case):
    masks=[(ref,REF,'STAPLE reference'),(pred,PRED,'Raw prediction')]
    meshes=[]
    for mask,color,label in masks:
        if mask.any():
            # Zero padding closes surfaces at original-volume edges without changing saved masks.
            padded=np.pad(mask.astype(np.uint8),1)
            vertices,faces,_,_=marching_cubes(padded,.5,spacing=spacing[::-1],step_size=1,allow_degenerate=False)
            vertices-=np.array(spacing[::-1])
            meshes.append((vertices[:,::-1],faces,color,label))
        else:meshes.append((None,None,color,label))
    valid=[m[0] for m in meshes if m[0] is not None]
    if valid:
        points=np.concatenate(valid);lo=points.min(axis=0)-2;hi=points.max(axis=0)+2
    else:lo=np.zeros(3);hi=np.ones(3)*10
    fig=plt.figure(figsize=(16,9),dpi=100)
    for i,(vertices,faces,color,label) in enumerate(meshes,1):
        ax=fig.add_subplot(1,2,i,projection='3d');ax.set_facecolor(BG)
        if vertices is not None:
            # Lighting follows actual triangle normals; no smoothing, trimming, or component filtering.
            tris=vertices[faces];normals=np.cross(tris[:,1]-tris[:,0],tris[:,2]-tris[:,0]);length=np.linalg.norm(normals,axis=1);normals/=np.maximum(length[:,None],1e-10)
            light=np.array([.3,-.5,.8]);light/=np.linalg.norm(light);shades=.35+.65*np.abs(normals@light)
            from matplotlib.colors import to_rgb
            colors=np.array(to_rgb(color))[None,:]*shades[:,None]
            ax.add_collection3d(Poly3DCollection(tris,facecolors=colors,edgecolor='none',linewidth=0))
        else:ax.text2D(.5,.5,'No foreground predicted' if i==2 else 'Empty reference',transform=ax.transAxes,ha='center',color=FG,fontsize=22)
        ax.set_xlim(lo[0],hi[0]);ax.set_ylim(lo[1],hi[1]);ax.set_zlim(lo[2],hi[2]);ax.set_box_aspect(hi-lo)
        ax.view_init(elev=20,azim=-60);ax.set_axis_off();ax.set_title(label,color=color,fontsize=23)
    fig.suptitle(case+' | Tumor surfaces at the same scale',fontsize=26,color=FG)
    fig.text(.5,.025,'All foreground components retained | Surface from saved masks | Physical spacing 0.21 mm',ha='center',fontsize=14,color='#9aadc5')
    fig.tight_layout(rect=[0,.05,1,.93]);fig.savefig(path,dpi=100,facecolor=BG);plt.close(fig)

def main():
    summary=json.loads((PROJECT/'results/pilot/metrics.json').read_text());out=PROJECT/'results/pilot/assets';out.mkdir(parents=True,exist_ok=True)
    manifests=[]
    for report in summary['cases']:
        case=report['case'];folder=out/case;folder.mkdir(parents=True,exist_ok=True)
        paths=[PROJECT/'data/nnUNet_raw'/DATASET/'imagesTs'/(case+'_0000.nii.gz'),PROJECT/'data/test_reference/STAPLE'/(case+'.nii.gz'),Path(report['prediction'])]
        ci,ri,pi=[sitk.ReadImage(str(p)) for p in paths];ct,ref,pred=[sitk.GetArrayFromImage(i) for i in [ci,ri,pi]];ref=ref>0;pred=pred>0;spacing=ci.GetSpacing()
        selected={}
        for axis,name in [(0,'axial'),(1,'coronal'),(2,'sagittal')]:
            indices=tuple(i for i in range(3) if i!=axis);areas=ref.sum(axis=indices);idx=int(np.argmax(areas)) if ref.any() else ct.shape[axis]//2
            selected[name]=idx
            c,r,p=[np.take(a,idx,axis=axis) for a in [ct,ref,pred]]
            dims=np.delete(np.array(spacing[::-1]),axis);extent=[0,c.shape[1]*dims[1],0,c.shape[0]*dims[0]]
            for mode in ['ct','reference','prediction','both','error']:
                save_plane(folder/(name+'_'+mode+'.png'),c,r,p,mode,f'{case} | {name} | {mode}',extent)
        for mode in ['ct','reference','prediction','both']:save_projection(folder/('projection_'+mode+'.png'),ct,ref,pred,spacing,case,mode)
        for mode in ['ct','reference','prediction','both']:save_projection(folder/('projection_axis2_'+mode+'.png'),ct,ref,pred,spacing,case,mode,axis=2)
        # Explicit failure illustration after evaluation, distinct from prespecified reference-area slices.
        fp_counts=(pred&~ref).sum(axis=(1,2));fp_z=int(np.argmax(fp_counts))
        extent=[0,ct.shape[2]*spacing[0],0,ct.shape[1]*spacing[1]]
        for mode in ['ct','prediction','error']:
            save_plane(folder/('axial_max_fp_'+mode+'.png'),ct[fp_z],ref[fp_z],pred[fp_z],mode,
                       f'{case} | max-FP plane | {mode}',extent)
        save_surface(folder/'tumor_surfaces.png',ref,pred,spacing,case)
        manifest=dict(case=case,representative=case==summary['representative_test_case'],worst_dice=case==summary['worst_dice_case'],
            slice_selection='maximum STAPLE foreground area on each axis, not optimized for prediction agreement',slice_indices_zyx=selected,
            max_false_positive_axial_index=fp_z,max_fp_slice_policy='post-evaluation diagnostic: plane with most FP voxels; not used for training or case selection',
            ct_window_hu=[-300,300],projection_window_hu=[-400,1000],assets=[str(p.resolve()) for p in sorted(folder.glob('*.png'))],metrics=report['references']['STAPLE'])
        manifests.append(manifest);print('ASSETS_READY',case,flush=True)
    fig,ax=plt.subplots(figsize=(14,7),dpi=100);x=np.arange(len(summary['cases']));w=.35
    for key,off,color in [('dice',-w/2,REF),('iou',w/2,PRED)]:
        values=[r['references']['STAPLE'][key]*100 for r in summary['cases']]
        ax.bar(x+off,values,w,color=color,label=key.upper())
        for xx,v in zip(x+off,values):ax.text(xx,v+1,f'{v:.1f}',ha='center',fontsize=13)
    ax.set_xticks(x,[r['case'].replace('D10_','') for r in summary['cases']]);ax.set_ylim(0,108);ax.set_ylabel('Overlap with STAPLE (%)');ax.set_xlabel('Five prespecified unseen mice (one scan each)')
    ax.set_title('Two-example fine-tuning | All five frozen test cases',fontsize=24,pad=18);ax.legend(frameon=False,labelcolor=FG)
    fig.text(.5,.015,'Raw model predictions; no postprocessing. One cohort, five mice: not a generalization guarantee.',ha='center',fontsize=13,color='#9aadc5')
    fig.tight_layout(rect=[0,.05,1,1]);fig.savefig(out/'all_test_metrics.png',dpi=100,facecolor=BG);plt.close(fig)
    (PROJECT/'experiments/assets-manifest.json').write_text(json.dumps(dict(cases=manifests,aggregate_chart=str((out/'all_test_metrics.png').resolve())),indent=2),encoding='utf-8')
    print('ALL_ASSETS_READY',flush=True)

if __name__=='__main__':main()
