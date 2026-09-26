"""Paired raw/guarded plots and a report, without replacing earlier independent results."""
import json
from pathlib import Path
import numpy as np
import SimpleITK as sitk
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from fourshot_config import *
from render_assets import save_plane,save_projection,BG,FG,REF,PRED

def main():
    summary=json.loads((RESULTS/'summary.json').read_text());train=json.loads((EXP/'training.json').read_text())
    for row in summary['cases']:
        case=row['case'];out=RESULTS/case;image,reference=sources(case)
        ci,ri,ai,bi=[sitk.ReadImage(str(p)) for p in [image,reference,out/'raw.nii.gz',out/'guarded.nii.gz']]
        ct,ref,raw,clean=[sitk.GetArrayFromImage(i) for i in [ci,ri,ai,bi]];ref=ref>0;raw=raw>0;clean=clean>0;spacing=ci.GetSpacing()
        z=int(np.argmax(ref.sum(axis=(1,2))));extent=[0,ct.shape[2]*spacing[0],0,ct.shape[1]*spacing[1]]
        save_plane(out/'target_guarded.png',ct[z],ref[z],clean[z],'both',case+' | 4 train + guard',extent)
        save_plane(out/'target_error.png',ct[z],ref[z],clean[z],'error',case+' | 4 train + guard | errors',extent)
        save_projection(out/'raw_full_volume.png',ct,ref,raw,spacing,case+' | RAW 4-shot','both',axis=2)
        save_projection(out/'guarded_full_volume.png',ct,ref,clean,spacing,case+' | SPATIAL GUARD','both',axis=2)
        fig,axes=plt.subplots(1,2,figsize=(12,12),dpi=130)
        view=ct.max(axis=2);physical=[0,ct.shape[1]*spacing[1],0,ct.shape[0]*spacing[2]]
        for ax,pred,kind in zip(axes,[raw,clean],['raw','guarded']):
            ax.imshow(view,cmap='gray',vmin=-400,vmax=1000,origin='lower',extent=physical)
            if ref.any():ax.contour(ref.any(axis=2),levels=[.5],colors=[REF],linewidths=.8,origin='lower',extent=physical)
            if pred.any():ax.contour(pred.any(axis=2),levels=[.5],colors=[PRED],linewidths=.8,origin='lower',extent=physical)
            lower,upper=row['guard']['allowed_z_centroid_interval']
            for bound in [lower,upper]:ax.axhline(bound*spacing[2],color='#d6be70',linestyle='--',linewidth=.7)
            ax.set_aspect('equal');ax.axis('off');m=row['four_case'][kind]
            ax.set_title(f'{kind} | Dice {m["dice"]*100:.2f}%\nIoU {m["iou"]*100:.2f}%',fontsize=19)
        fig.suptitle(case+' | same model, before / after fixed guard',fontsize=22,color=FG)
        fig.legend(handles=[Patch(color=REF,label='Reference'),Patch(color=PRED,label='Prediction'),Patch(color='#d6be70',label='Allowed component-centroid Z')],loc='lower center',ncol=3,frameon=False,labelcolor=FG)
        fig.tight_layout(rect=[0,.05,1,.94]);fig.savefig(out/'comparison.png',facecolor=BG);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(14,6),dpi=130);x=np.arange(3);w=.25
    for stage,shift,color,label in [('two_case_raw',-w,'#718096','2 train, raw'),('four_case_raw',0,PRED,'4 train, raw'),('four_case_guarded',w,REF,'4 train + guard')]:
        for ax,key in zip(axes,['dice','iou']):
            values=[]
            for r in summary['cases']:
                m=r['old_two_case_raw'] if stage=='two_case_raw' else r['four_case']['raw' if stage=='four_case_raw' else 'guarded']
                values.append(m[key]*100)
            ax.bar(x+shift,values,w,color=color,label=label)
            for xx,val in zip(x+shift,values):ax.text(xx,val+1,f'{val:.1f}',ha='center',fontsize=10)
            ax.set_xticks(x,[r['case'].replace('D10_','') for r in summary['cases']]);ax.set_ylim(0,108);ax.set_ylabel(key.upper()+' (%)');ax.set_xlabel('Same three development cases')
            ax.set_title(key.upper()+' vs unchanged STAPLE')
    axes[0].legend(frameon=False,labelcolor=FG,fontsize=10)
    fig.suptitle('Four train / three evaluate | Development comparison',fontsize=21,color=FG)
    fig.text(.5,.015,'No new independent test claim. Guard fits only four training animals; all raw predictions are preserved.',ha='center',fontsize=11,color='#9aadc5')
    fig.tight_layout(rect=[0,.06,1,.94]);fig.savefig(RESULTS/'metrics_comparison.png',facecolor=BG);plt.close(fig)
    lines=['# 四例训练、三例开发性评估','',
        '品牌：思语视觉。沿用已有七只小鼠，旧结果已被检查，因此本轮是开发性对比，不是新的独立测试。','',
        '- 训练：'+', '.join(TRAIN)+'。评估：'+', '.join(TEST)+'。',
        f'- 重新从官方原始nnFoundation-CNN初始化；30 epoch×50次迭代，训练{train["seconds"]/60:.2f}分钟，峰值allocated显存{train["peak_gpu_allocated_gb"]:.2f}GiB。',
        '- 新旧原始模型使用相同训练迭代预算；增加数据与后处理的效果分别列出。','',
        '## 自动清理方法','',
        'CT软组织窗口及轻度开运算定位主体，在身体长轴的相对坐标上拟合四例训练肿瘤的范围，外加5mm安全边界。保留中心位于范围内的完整预测连通块，并排除小于最小训练连通目标体积10%的极小区域。本轮体积下限约5.24mm³；不限制只能有一个或两个目标，不逐像素截断大块边缘。',
        '阈值和先验只来自四例训练数据；逐次留出一个训练标注检查时保留100%前景。这是先验覆盖性检查，不代表模型泛化已验证。应用阶段只读CT、预测、固定先验，不读评估标注。','',
        '## 同三例宏平均','']
    for stage,m in summary['macro'].items():lines.append(f'- {stage}：Dice {m["dice"]*100:.2f}%，IoU {m["iou"]*100:.2f}%。')
    before=sum(r['four_case']['raw']['fp_beyond_10mm'] for r in summary['cases'])
    after=sum(r['four_case']['guarded']['fp_beyond_10mm'] for r in summary['cases'])
    lines += ['',f'距参考目标>10mm的远端FP合计由{before}降为{after}体素；清理共移除{sum(r["fp_removed"] for r in summary["cases"])}个FP，损失{sum(r["tp_removed"] for r in summary["cases"])}个原本正确重合的体素。10mm是评价时的误差统计阈值，清理算法并不读取评估参考或使用到真瘤的距离。',
        f'原始/清理后的宏平均recall分别为{summary["macro"]["four_case_raw"]["recall"]*100:.2f}% / {summary["macro"]["four_case_guarded"]["recall"]*100:.2f}%。最终Dice为{summary["macro"]["four_case_guarded"]["dice"]*100:.2f}%，IoU为{summary["macro"]["four_case_guarded"]["iou"]*100:.2f}%；原有局部漏分仍需结合recall和切片复核。']
    for row in summary['cases']:
        lines += ['',f'## {row["case"]}','']
        for kind,m in row['four_case'].items():
            lines.append(f'- {kind}：Dice {m["dice"]*100:.2f}%，IoU {m["iou"]*100:.2f}%，体积偏差 {m["signed_volume_error_percent"]:+.2f}%，距参考>10mm的FP {m["fp_beyond_10mm"]}体素。')
        lines.append(f'- 清理删除FP {row["fp_removed"]}体素、重合真阳性 {row["tp_removed"]}体素；原图和清理后均保留，便于核查收益与损失。')
        lines.append(f'- [前后对照]({row["case"]}/comparison.png) · [清理后Slicer掩膜]({row["case"]}/guarded.seg.nrrd)')
        if (RESULTS/row['case']/'comparison.mrb').exists():lines.append(f'- [Slicer完整对比场景]({row["case"]}/comparison.mrb)，已实际载入原始/清理掩膜并逐体素核验。')
    lines += ['', '## 使用范围与限制','',
        '这是一种针对标准化Dataset10基线、皮下侧腹目标的位置先验。使用本队列已与动物长轴大致对齐的CT z轴，没有独立识别头尾或定位腹部器官。异位肿瘤、扫描姿态改变、不同裁剪范围或极小目标可能被错误排除；不应当作任意肿瘤分割的通用后处理。身体定位或方向检查失败时需人工复核。先验不能找回模型完全漏掉的目标，也不保证解决允许区域内的局部误分。',
        '评估文件只在原始和清理后预测都保存之后打开。三个病例以前已经用于诊断，因而即使本轮规则未直接读取它们的标签，结果仍应称为开发性复测。']
    (RESULTS/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('FOURSHOT_REPORT_READY',RESULTS/'REPORT.md',flush=True)

if __name__=='__main__':main()
