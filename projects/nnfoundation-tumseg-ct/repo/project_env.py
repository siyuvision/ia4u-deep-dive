import os
from pathlib import Path
PROJECT=Path(__file__).resolve().parents[1]
DATASET='Dataset902_TumSegFewShot'
WEIGHTS=Path(os.environ.get('NNFOUNDATION_WEIGHTS', Path.home()/'.cache/siyu-vision-models/nnfoundation/nnFoundationCNN.pth')).expanduser().resolve()
WEIGHTS_SHA='ac262d3e8c226c79f9567fddc34d38e011284730ddf2294b5c6c319ce2f22bbd'

def configure():
    for key,rel in {'nnUNet_raw':'data/nnUNet_raw','nnUNet_preprocessed':'data/nnUNet_preprocessed','nnUNet_results':'results/models'}.items():
        p=PROJECT/rel;p.mkdir(parents=True,exist_ok=True);os.environ[key]=str(p)
    os.environ['nnUNet_compile']='false'
    os.environ['nnUNet_n_proc_DA']='2'
    os.environ['OMP_NUM_THREADS']='4'
    os.environ['MKL_NUM_THREADS']='4'
    os.environ['PYTHONUTF8']='1'
