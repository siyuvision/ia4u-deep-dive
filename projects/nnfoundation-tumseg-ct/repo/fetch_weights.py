"""Fetch the official checkpoint on explicit request, or verify the existing cache."""
import argparse
import hashlib
import urllib.request
from project_env import WEIGHTS, WEIGHTS_SHA

URL='https://huggingface.co/MIC-DKFZ/nnFoundationCNN/resolve/main/checkpoint_final.pth'

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--download',action='store_true',help='Download ~410 MB under the upstream CC BY-SA 4.0 license')
    args=ap.parse_args()
    if not WEIGHTS.exists():
        if not args.download:raise FileNotFoundError('Set NNFOUNDATION_WEIGHTS to the official checkpoint or pass --download')
        WEIGHTS.parent.mkdir(parents=True,exist_ok=True)
        temp=WEIGHTS.with_suffix('.download')
        if temp.exists():raise FileExistsError('Previous partial download exists; inspect it before retrying: '+str(temp))
        print('Downloading official nnFoundation-CNN weights; CC BY-SA 4.0',flush=True)
        with urllib.request.urlopen(URL,timeout=120) as response,temp.open('xb') as f:
            while block:=response.read(8*1024*1024):f.write(block)
        if digest(temp)!=WEIGHTS_SHA:raise ValueError('Checkpoint hash differs from this experiment. Partial file preserved; no cache replacement.')
        temp.rename(WEIGHTS)
    if digest(WEIGHTS)!=WEIGHTS_SHA:raise ValueError('Cached checkpoint hash does not match the published experiment')
    print('Official checkpoint SHA256 verified:',WEIGHTS_SHA)

if __name__=='__main__':main()
