"""Offline report encoder. Produces one .npy vector per report and an updated index CSV.

Use a local HuggingFace model directory when compute nodes do not have internet access.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch

ap=argparse.ArgumentParser(); ap.add_argument("--index",required=True); ap.add_argument("--model",required=True); ap.add_argument("--outdir",required=True); ap.add_argument("--batch",type=int,default=32); ap.add_argument("--max_length",type=int,default=256); ap.add_argument("--local_files_only",action="store_true")
a=ap.parse_args()
from transformers import AutoTokenizer, AutoModel

df=pd.read_csv(a.index); tok=AutoTokenizer.from_pretrained(a.model,local_files_only=a.local_files_only); model=AutoModel.from_pretrained(a.model,local_files_only=a.local_files_only).eval(); dev=torch.device("cuda" if torch.cuda.is_available() else "cpu"); model.to(dev)
outdir=Path(a.outdir); outdir.mkdir(parents=True,exist_ok=True); new=[]
for st in range(0,len(df),a.batch):
    part=df.iloc[st:st+a.batch]; texts=[Path(p).read_text(encoding="utf-8") for p in part.report_path]
    t=tok(texts,padding=True,truncation=True,max_length=a.max_length,return_tensors="pt"); t={k:v.to(dev) for k,v in t.items()}
    with torch.no_grad():
        h=model(**t).last_hidden_state; mask=t["attention_mask"].unsqueeze(-1); emb=((h*mask).sum(1)/mask.sum(1).clamp_min(1)).cpu().numpy()
    for (_,r),e in zip(part.iterrows(),emb):
        p=outdir/f"s{int(r.study_id)}.npy"; np.save(p,e.astype(np.float32)); new.append(str(p))
df["report_embedding_path"]=new; df.to_csv(Path(a.index).with_name(Path(a.index).stem+"_embedded.csv"),index=False)
