"""Create a tiny 3-modality precomputed dataset for end-to-end pipeline debugging."""
from pathlib import Path
import numpy as np, pandas as pd
root=Path('dummy_data').resolve(); root.mkdir(exist_ok=True)
rng=np.random.default_rng(7); rows=[]
for split,n in [('train',32),('val',12),('test',12)]:
    for i in range(n):
        sid={'train':1000,'val':2000,'test':3000}[split]+i
        image=rng.normal(size=64).astype('float32'); report=rng.normal(size=48).astype('float32'); ehr=rng.normal(size=32).astype('float32')
        # synthetic label with multimodal interaction
        score=image[:4].mean()+0.6*report[:4].mean()+0.8*ehr[:4].mean()+0.3*image[0]*report[0]
        y=float(score>0)
        paths=[]
        for name,x in [('image',image),('report',report),('ehr',ehr)]:
            p=root/f'{split}_{sid}_{name}.npy'; np.save(p,x); paths.append(str(p))
        rows.append(dict(subject_id=sid,study_id=sid*10,split=split,index_time='2026-01-01',image_path=paths[0],report_embedding_path=paths[1],ehr_path=paths[2],y_vent24=y,mask_vent24=1,y_mortality=float(score>0.8),mask_mortality=1))
pd.DataFrame(rows).to_csv(root/'manifest.csv',index=False)
print(root/'manifest.csv')
