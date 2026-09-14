from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image


class ManifestDataset(Dataset):
    """Generic multimodal dataset driven by YAML modality_fields.

    Supported kinds:
      image    -> PIL RGB converted later in collate
      vector   -> npy [D]
      sequence -> npy [T,F]
      text     -> UTF-8 text file
    """
    def __init__(self, manifest: str, cfg: Dict[str, Any], split: str):
        self.cfg = cfg
        self.split = str(split)
        fmt = cfg["data"].get("format", "csv")
        self.df = pd.read_parquet(manifest) if fmt == "parquet" else pd.read_csv(manifest)
        if "split" in self.df.columns:
            self.df = self.df[self.df["split"].astype(str) == split].reset_index(drop=True)
        self.modalities = list(cfg["model"]["modalities"])
        self.fields = cfg["data"]["modality_fields"]
        all_labels = cfg["data"]["labels"]
        task_names = cfg.get("experiment", {}).get("task_names", list(all_labels))
        self.labels = {name: all_labels[name] for name in task_names}

    def __len__(self):
        return len(self.df)

    def _load(self, value, kind):
        if value is None or (isinstance(value, float) and np.isnan(value)) or str(value).strip() == "":
            return None
        p = Path(str(value))
        if kind == "image":
            return Image.open(p).convert("RGB")
        if kind in ("vector", "sequence"):
            return np.load(p).astype(np.float32)
        if kind == "text":
            return p.read_text(encoding="utf-8")
        raise ValueError(f"Unsupported modality kind: {kind}")

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        mods = {}
        availability = []
        for m in self.modalities:
            spec = self.fields[m]
            col = spec["column"]
            val = row[col] if col in row.index else None
            x = self._load(val, spec["kind"])
            mods[m] = x
            availability.append(0.0 if x is None else 1.0)

        ys, masks = [], []
        for task, spec in self.labels.items():
            ys.append(float(row.get(spec["column"], 0.0)))
            masks.append(float(row.get(spec["mask_column"], 1.0)))

        meta = {k: row[k] for k in ["subject_id", "hadm_id", "stay_id", "study_id", "index_time"] if k in row.index}
        return {
            "modalities": mods,
            "modality_mask": np.asarray(availability, dtype=np.float32),
            "labels": np.asarray(ys, dtype=np.float32),
            "label_mask": np.asarray(masks, dtype=np.float32),
            "meta": meta,
        }


def _image_to_tensor(img: Image.Image, image_size: int):
    import torchvision.transforms.functional as TF
    img = TF.resize(img, [image_size, image_size])
    x = TF.to_tensor(img)
    return x


def collate_manifest(batch: List[Dict[str, Any]], cfg: Dict[str, Any]):
    modalities = list(cfg["model"]["modalities"])
    fields = cfg["data"]["modality_fields"]
    out_mods = {}

    for m in modalities:
        kind = fields[m]["kind"]
        items = [b["modalities"][m] for b in batch]
        if kind == "image":
            size = int(cfg["data"].get("image_size", 224))
            template = torch.zeros(3, size, size)
            out_mods[m] = torch.stack([template if x is None else _image_to_tensor(x, size) for x in items])
        elif kind == "vector":
            first = next((x for x in items if x is not None), None)
            if first is None:
                in_dim = int(cfg["model"]["encoders"][m]["input_dim"])
                first = np.zeros((in_dim,), np.float32)
            out_mods[m] = torch.from_numpy(np.stack([np.zeros_like(first) if x is None else x for x in items])).float()
        elif kind == "sequence":
            first = next((x for x in items if x is not None), None)
            if first is None:
                f = int(cfg["model"]["encoders"][m]["input_dim"])
                first = np.zeros((1, f), np.float32)
            max_len = min(max((1 if x is None else x.shape[0]) for x in items), int(cfg["data"].get("ehr_max_len", 48)))
            f = first.shape[-1]
            arr = np.zeros((len(items), max_len, f), np.float32)
            lengths = np.ones((len(items),), np.int64)
            for i, x in enumerate(items):
                if x is None:
                    continue
                xx = x[-max_len:]
                arr[i, :len(xx)] = xx
                lengths[i] = len(xx)
            out_mods[m] = {"values": torch.from_numpy(arr).float(), "lengths": torch.from_numpy(lengths).long()}
        elif kind == "text":
            out_mods[m] = ["" if x is None else x for x in items]
        else:
            raise ValueError(kind)

    return {
        "modalities": out_mods,
        "modality_mask": torch.from_numpy(np.stack([b["modality_mask"] for b in batch])).float(),
        "labels": torch.from_numpy(np.stack([b["labels"] for b in batch])).float(),
        "label_mask": torch.from_numpy(np.stack([b["label_mask"] for b in batch])).float(),
        "meta": [b["meta"] for b in batch],
    }
