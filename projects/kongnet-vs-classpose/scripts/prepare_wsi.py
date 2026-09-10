"""Create one common, lightweight tissue ROI for both WSI pipelines."""
import json
from pathlib import Path
import cv2
import numpy as np
import openslide
from PIL import Image

PROJECT = Path(__file__).resolve().parents[2]
slide_path = next((PROJECT/'data/input').glob('*.svs'))
slide=openslide.OpenSlide(str(slide_path))
thumb=np.asarray(slide.get_thumbnail((1800,1800)).convert('RGB'))
hsv=cv2.cvtColor(thumb,cv2.COLOR_RGB2HSV)
mask=((hsv[:,:,1]>25)&(hsv[:,:,2]<245)).astype(np.uint8)*255
mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((9,9),np.uint8))
contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
contours=[c for c in contours if cv2.contourArea(c)>500]
mask[:]=0
cv2.drawContours(mask,contours,-1,255,thickness=cv2.FILLED)
out=PROJECT/'data/derived';out.mkdir(parents=True,exist_ok=True)
masks=out/'masks';masks.mkdir(exist_ok=True)
Image.fromarray(mask).save(masks/f'{slide_path.stem}.png')
sx=slide.dimensions[0]/mask.shape[1];sy=slide.dimensions[1]/mask.shape[0]
features=[]
for i,c in enumerate(contours):
    points=c.reshape(-1,2)*[sx,sy]
    points=np.concatenate([points,points[:1]])
    features.append({'type':'Feature','properties':{'classification':{'name':'shared-tissue'}},'geometry':{'type':'Polygon','coordinates':[points.tolist()]}})
(out/'shared-tissue.geojson').write_text(json.dumps({'type':'FeatureCollection','features':features}),encoding='utf-8')
(out/'wsi-input.json').write_text(json.dumps({'slide':str(slide_path),'dimensions':slide.dimensions,'mpp':slide.properties.get('openslide.mpp-x'),'mask_shape':mask.shape,'tissue_fraction':float((mask>0).mean()),'method':'HSV saturation>25 and value<245; close9; external contours area>500 thumbnail px; common approximate ROI, not a trained tissue model'},indent=2),encoding='utf-8')
print('Prepared common ROI:',len(features),'regions', 'tissue fraction',float((mask>0).mean()))
slide.close()
