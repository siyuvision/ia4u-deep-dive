from project_env import PROJECT,WEIGHTS,WEIGHTS_SHA,configure
RUN='four_train_three_eval_20260927_v1'
DATASET='Dataset903_TumSegFourShot'
DATASET_ID=903
PLAN='TumSegFourShot_30ep_128'
EXP=PROJECT/'experiments'/RUN
RESULTS=PROJECT/'results'/RUN
TRAIN=['D10_M16_0d','D10_M33_0d','D10_M07_0d','D10_M19_0d']
TEST=['D10_M02_0d','D10_M22_0d','D10_M11_0d']

def sources(case):
    old=PROJECT/'data/nnUNet_raw/Dataset902_TumSegFewShot'
    if case in ['D10_M16_0d','D10_M33_0d']:
        return old/'imagesTr'/(case+'_0000.nii.gz'),old/'labelsTr'/(case+'.nii.gz')
    return old/'imagesTs'/(case+'_0000.nii.gz'),PROJECT/'data/test_reference/STAPLE'/(case+'.nii.gz')
