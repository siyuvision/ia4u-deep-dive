"""Read actual Parquet bytes, never dataset viewer thumbnails, for labels."""
import io
import json
from pathlib import Path
import fastparquet
import numpy as np
from PIL import Image

P=Path(__file__).resolve().parents[2]
df=fastparquet.ParquetFile(P/'data/input/conic-train-shard0.parquet').to_pandas()
row=df[df.patch_id==1001].iloc[0]
out=P/'data/derived/conic-1001';out.mkdir(parents=True,exist_ok=True)
for field in ['image','inst_map','class_map']:
    img=Image.open(io.BytesIO(row[field+'.bytes']))
    img.save(out/(field+'.png'))
inst=np.asarray(Image.open(out/'inst_map.png'));cls=np.asarray(Image.open(out/'class_map.png'))
assert cls.ndim==2 and inst.shape==cls.shape
assert set(np.unique(cls)).issubset(set(range(7))),np.unique(cls)
counts=[]
for c in range(1,7):
    counts.append(sum(int(np.bincount(cls[inst==i].astype(int)).argmax())==c for i in np.unique(inst) if i))
expected=[int(row['count_'+n]) for n in ['neutrophil','epithelial','lymphocyte','plasma','eosinophil','connective']]
edge=set(np.r_[inst[0,:],inst[-1,:],inst[:,0],inst[:,-1]])-{0}
report={'patch_id':1001,'patch_info':row.patch_info,'raster_counts':counts,'metadata_counts':expected,'counts_agree':counts==expected,'instances':sum(counts),'edge_touching_instances':len(edge),'validation':'Original Parquet 2D label IDs verified; count metadata differs, so evaluation derives all reference points/counts from raster labels including boundary instances. Exact metadata counting convention is unverified.','scope':'Public training-release sample; not an independent test'}
(out/'validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
