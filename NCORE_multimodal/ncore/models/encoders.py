from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
import importlib.util
import math
import sys
import torch
import torch.nn as nn


class IdentityPrecomputedEncoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int | None = None):
        super().__init__()
        output_dim = output_dim or input_dim
        self.output_dim = output_dim
        self.proj = nn.Identity() if input_dim == output_dim else nn.Linear(input_dim, output_dim)
    def forward(self, x):
        return self.proj(x.float())


class NativeEHREncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, layers: int = 1, dropout: float = 0.0):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.output_dim = hidden_dim
    def forward(self, x):
        values, lengths = x["values"], x["lengths"]
        packed = nn.utils.rnn.pack_padded_sequence(values, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(packed)
        return h[-1]


class NativeEHRTransformer(nn.Module):
    """Mask-aware temporal Transformer that returns one pooled EHR vector."""
    def __init__(self, input_dim: int, hidden_dim: int = 128, layers: int = 2,
                 num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=layers)
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_dim = hidden_dim

    def forward(self, x):
        values, lengths = x["values"], x["lengths"]
        hidden = self.input_proj(values.float())
        position = torch.arange(hidden.size(1), device=hidden.device, dtype=torch.float32)
        frequency = torch.exp(
            torch.arange(0, hidden.size(2), 2, device=hidden.device, dtype=torch.float32)
            * (-math.log(10000.0) / hidden.size(2))
        )
        positional = torch.zeros(
            hidden.size(1), hidden.size(2), device=hidden.device, dtype=torch.float32
        )
        positional[:, 0::2] = torch.sin(position.unsqueeze(1) * frequency)
        positional[:, 1::2] = torch.cos(
            position.unsqueeze(1) * frequency[:positional[:, 1::2].size(1)]
        )
        hidden = hidden + positional.to(hidden.dtype).unsqueeze(0)
        time = torch.arange(hidden.size(1), device=hidden.device).unsqueeze(0)
        padding_mask = time >= lengths.to(hidden.device).unsqueeze(1)
        hidden = self.transformer(hidden, src_key_padding_mask=padding_mask)
        valid = (~padding_mask).unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * valid).sum(1) / valid.sum(1).clamp_min(1.0)
        return self.output_norm(pooled)


class TorchvisionImageEncoder(nn.Module):
    def __init__(self, architecture="densenet121", pretrained=False):
        super().__init__()
        import torchvision.models as tvm
        weights = None
        model = getattr(tvm, architecture)(weights=weights)
        if hasattr(model, "classifier"):
            d = model.classifier.in_features
            model.classifier = nn.Identity()
        elif hasattr(model, "fc"):
            d = model.fc.in_features
            model.fc = nn.Identity()
        else:
            raise ValueError(f"Unsupported torchvision architecture {architecture}")
        self.backbone = model
        self.output_dim = d
    def forward(self, x):
        return self.backbone(x.float())


class HFTextEncoder(nn.Module):
    def __init__(self, model_name_or_path: str, max_length=256, local_files_only=False):
        super().__init__()
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as e:
            raise ImportError("Install transformers for backend=hf_text") from e
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, local_files_only=local_files_only)
        self.model = AutoModel.from_pretrained(model_name_or_path, local_files_only=local_files_only)
        self.output_dim = int(self.model.config.hidden_size)
        self.max_length = max_length
    def forward(self, texts):
        dev = next(self.model.parameters()).device
        tok = self.tokenizer(texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
        tok = {k:v.to(dev) for k,v in tok.items()}
        out = self.model(**tok).last_hidden_state
        mask = tok["attention_mask"].unsqueeze(-1)
        return (out * mask).sum(1) / mask.sum(1).clamp_min(1)


class MedFuseEHRAdapter(nn.Module):
    """Reuse upstream MedFuse LSTM feature API without copying its source."""
    def __init__(self, repo: str, checkpoint: str | None, input_dim=76, hidden_dim=128, num_classes=1):
        super().__init__()
        repo = str(Path(repo).resolve())
        if repo not in sys.path: sys.path.insert(0, repo)
        from models.ehr_models import LSTM
        self.model = LSTM(input_dim=input_dim, num_classes=num_classes, hidden_dim=hidden_dim)
        self.output_dim = self.model.feats_dim
        if checkpoint:
            state = torch.load(checkpoint, map_location="cpu")
            sd = state.get("state_dict", state.get("model", state))
            # Tolerate upstream trainer prefixes.
            clean = {k.replace("ehr_model.", "").replace("model.ehr_model.", ""):v for k,v in sd.items() if "ehr_model" in k or k in self.model.state_dict()}
            self.model.load_state_dict(clean, strict=False)
    def forward(self, x):
        values, lengths = x["values"], x["lengths"].cpu().numpy().tolist()
        _, feat = self.model(values, lengths)
        return feat if feat.ndim == 2 else feat.unsqueeze(0)


class MedFuseCXRAdapter(nn.Module):
    def __init__(self, repo: str, checkpoint: str | None, architecture="densenet121"):
        super().__init__()
        repo = str(Path(repo).resolve())
        if repo not in sys.path: sys.path.insert(0, repo)
        # Build a tiny args object matching upstream CXRModels requirements.
        class A: pass
        a = A(); a.vision_backbone = architecture; a.pretrained = False; a.vision_num_classes = 14
        from models.cxr_models import CXRModels
        self.model = CXRModels(a, device="cpu")
        self.output_dim = self.model.feats_dim
        if checkpoint:
            state = torch.load(checkpoint, map_location="cpu")
            sd = state.get("state_dict", state.get("model", state))
            clean = {k.replace("cxr_model.", "").replace("model.cxr_model.", ""):v for k,v in sd.items() if "cxr_model" in k or k in self.model.state_dict()}
            self.model.load_state_dict(clean, strict=False)
    def forward(self, x):
        _, _, feat = self.model(x)
        return feat


def build_encoder(name: str, spec: Dict[str, Any], cfg: Dict[str, Any]) -> nn.Module:
    b = spec["backend"]
    if b == "precomputed":
        enc = IdentityPrecomputedEncoder(int(spec["input_dim"]), int(spec.get("output_dim", spec["input_dim"])))
    elif b == "native_lstm":
        enc = NativeEHREncoder(int(spec["input_dim"]), int(spec.get("hidden_dim", 128)), int(spec.get("layers", 1)), float(spec.get("dropout", 0.0)))
    elif b == "native_transformer":
        enc = NativeEHRTransformer(
            int(spec["input_dim"]),
            int(spec.get("hidden_dim", 128)),
            int(spec.get("layers", 2)),
            int(spec.get("num_heads", 4)),
            float(spec.get("dropout", 0.1)),
        )
    elif b == "torchvision":
        enc = TorchvisionImageEncoder(spec.get("architecture", "densenet121"), bool(spec.get("pretrained", False)))
    elif b == "hf_text":
        model_path = spec.get("model_name_or_path") or cfg["paths"]["report_model_name_or_path"]
        enc = HFTextEncoder(model_path, int(cfg["data"].get("report_max_length", 256)), bool(spec.get("local_files_only", False)))
    elif b == "medfuse_ehr":
        enc = MedFuseEHRAdapter(cfg["paths"]["medfuse_repo"], cfg["paths"].get("medfuse_ehr_checkpoint"),
                                int(spec.get("input_dim", 76)), int(spec.get("hidden_dim", 128)))
    elif b == "medfuse_cxr":
        enc = MedFuseCXRAdapter(cfg["paths"]["medfuse_repo"], cfg["paths"].get("medfuse_cxr_checkpoint"), spec.get("architecture", "densenet121"))
    else:
        raise ValueError(f"Unknown encoder backend {b} for modality {name}")
    if bool(spec.get("freeze", False)):
        for p in enc.parameters(): p.requires_grad = False
    return enc
