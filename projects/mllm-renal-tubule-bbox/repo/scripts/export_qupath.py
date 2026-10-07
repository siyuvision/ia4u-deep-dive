"""Export stored bounding boxes as QuPath GeoJSON in the clean input image's pixel space."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from PIL import Image

PROJECT=Path(__file__).resolve().parents[2]

def feature_collection(data: dict, target_size: tuple[int,int]) -> dict:
    sx=target_size[0]/data['image']['width']
    sy=target_size[1]/data['image']['height']
    features=[]
    for b in data['predictions']:
        x0=(b['x']-b['width']/2)*sx
        y0=(b['y']-b['height']/2)*sy
        x1=(b['x']+b['width']/2)*sx
        y1=(b['y']+b['height']/2)*sy
        features.append({'type':'Feature','geometry':{'type':'Polygon','coordinates':[[[x0,y0],[x1,y0],[x1,y1],[x0,y1],[x0,y0]]]},'properties':{'objectType':'annotation','classification':{'name':'renal tubule bbox'},'name':'bounding box, not segmentation'}})
    return {'type':'FeatureCollection','features':features}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-id',required=True)
    p.add_argument('--image',type=Path,default=PROJECT/'data/input/kidney-he-case01-1024x791.jpg')
    p.add_argument('--results-dir',type=Path,default=PROJECT/'results')
    a=p.parse_args()
    with Image.open(a.image) as im:size=im.size
    out=a.results_dir/a.run_id/'geojson';out.mkdir(parents=True,exist_ok=True)
    files=list((a.results_dir/a.run_id/'parsed').glob('*.json'))
    files.append(PROJECT/'data/annotations/gt-manual.json')
    for f in files:
        result=feature_collection(json.loads(f.read_text(encoding='utf-8')),size)
        (out/(f.stem+'.geojson')).write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(f'Exported {len(files)} GeoJSON files at {size[0]}x{size[1]} input-image pixel coordinates to {out}')

if __name__=='__main__':main()
