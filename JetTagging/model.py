import argparse
import glob
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

MAXC, NFEAT, HEADER = 100, 6, 6
RECORD = HEADER + MAXC * NFEAT
N_CONT = 7
N_IN = 13

CONFIGS = {
    "pcnn": dict(amp=True, chunk=20000, no_ptmatch=False, width=96, dropout=0.1,
                 batch=32, lr=1e-3, epochs=50, eval_batch=128, num_workers=2,
                 out="pcnn_best.pt"),
    "part": dict(amp=True, chunk=20000, no_ptmatch=False, width=96, heads=8, layers=4,
                 dropout=0.2, batch=32, lr=1e-4, epochs=25, eval_batch=128,
                 num_workers=2, out="part_best.pt"),
}

def expand(patterns):
    out = []
    for p in patterns:
        hits = sorted(glob.glob(p))
        if not hits:
            raise FileNotFoundError(p)
        out += hits
    return out

def iter_records(paths, chunk):
    g = 0
    for p in paths:
        size = os.path.getsize(p)
        if size % (RECORD * 4):
            raise ValueError(f"{p}: size is not a multiple of the record length")
        n = size // (RECORD * 4)
        with open(p, "rb") as fh:
            for s in range(0, n, chunk):
                m = min(chunk, n - s)
                raw = np.fromfile(fh, dtype=np.float32, count=m * RECORD).reshape(m, RECORD)
                yield g, raw
                g += m

def read_headers(paths, chunk):
    lab, pt, nc = [], [], []
    for _, raw in iter_records(paths, chunk):
        lab.append(raw[:, 0].astype(np.int64))
        pt.append(np.array(raw[:, 1]))
        nc.append(raw[:, 5].astype(np.int32))
    return np.concatenate(lab), np.concatenate(pt), np.concatenate(nc)

def wrap_phi(x):
    return (x + np.pi) % (2 * np.pi) - np.pi

def build_features(raw, L):
    jpt, jeta, jphi, jm = raw[:, 1], raw[:, 2], raw[:, 3], raw[:, 4]
    P = raw[:, HEADER:].reshape(-1, MAXC, NFEAT)[:, :L]
    pt, eta, phi, E, pid, q = (P[..., i] for i in range(NFEAT))
    mask = pt > 0
    jE = np.sqrt((jpt * np.cosh(jeta)) ** 2 + jm ** 2)
    deta = eta - jeta[:, None]
    dphi = wrap_phi(phi - jphi[:, None])
    safe_pt = np.where(mask, pt, 1.0)
    safe_E = np.where(mask, E, 1.0)
    apid = np.abs(pid)
    is_e, is_mu, is_gam = apid == 11, apid == 13, apid == 22
    is_chh = (q != 0) & ~is_e & ~is_mu
    is_neh = (q == 0) & ~is_gam
    out = np.empty((len(raw), N_IN, L), np.float32)
    out[:, 0] = deta
    out[:, 1] = dphi
    out[:, 2] = np.log(safe_pt)
    out[:, 3] = np.log(safe_E)
    out[:, 4] = np.log(safe_pt / jpt[:, None])
    out[:, 5] = np.log(safe_E / jE[:, None])
    out[:, 6] = np.hypot(deta, dphi)
    out[:, 7] = q
    out[:, 8] = is_e
    out[:, 9] = is_mu
    out[:, 10] = is_gam
    out[:, 11] = is_chh
    out[:, 12] = is_neh
    out *= mask[:, None, :]
    return out, mask

def match_pt_spectrum(jpt, label, nbins=20, seed=0):
    rng = np.random.default_rng(seed)
    classes = np.unique(label)
    edges = np.quantile(jpt, np.linspace(0, 1, nbins + 1))
    edges[-1] += 1e-3
    b = np.digitize(jpt, edges) - 1
    keep = []
    for i in range(nbins):
        groups = [np.where((b == i) & (label == c))[0] for c in classes]
        n = min(len(g) for g in groups)
        if n == 0:
            continue
        keep += [rng.choice(g, n, replace=False) for g in groups]
    return np.sort(np.concatenate(keep))

class JetDataset(torch.utils.data.Dataset):
    def __init__(self, paths, keep, L, mu, sd):
        self.paths, self.keep, self.L, self.mu, self.sd = paths, keep, L, mu, sd
        self.file_ranges = []
        offset = 0
        for p in paths:
            n = os.path.getsize(p) // (RECORD * 4)
            self.file_ranges.append((offset, offset + n))
            offset += n

    def __len__(self):
        return len(self.keep)

    def __getitem__(self, idx):
        g = self.keep[idx]
        for path, (lo, hi) in zip(self.paths, self.file_ranges):
            if lo <= g < hi:
                with open(path, "rb") as fh:
                    fh.seek((g - lo) * RECORD * 4)
                    raw = np.fromfile(fh, dtype=np.float32, count=RECORD).reshape(1, RECORD)
                feats, mask = build_features(raw, self.L)
                feats[0, :N_CONT] = (feats[0, :N_CONT] - self.mu[:, None]) / self.sd[:, None]
                return torch.from_numpy(feats[0]), torch.from_numpy(mask[0]), int(raw[0, 0])
        raise IndexError(idx)

def compute_normalization(paths, keep, L, chunk):
    s, s2, cnt = np.zeros(N_CONT), np.zeros(N_CONT), 0
    for g, raw in iter_records(paths, chunk):
        lo, hi = np.searchsorted(keep, [g, g + len(raw)])
        if hi == lo:
            continue
        f, mk = build_features(raw[keep[lo:hi] - g], L)
        m = mk[:, None, :]
        x = f[:, :N_CONT] * m
        s += x.sum((0, 2))
        s2 += (x * x).sum((0, 2))
        cnt += m.sum()
    mu = s / cnt
    sd = np.sqrt(np.maximum(s2 / cnt - mu * mu, 0)) + 1e-6
    return mu, sd

def collate_fn(batch):
    feats, masks, labels = zip(*batch)
    return torch.stack(feats), torch.stack(masks), torch.tensor(labels, dtype=torch.long)

def split_indices(label):
    all_idx = np.arange(len(label))
    train_idx, hold_idx = train_test_split(all_idx, test_size=0.3, stratify=label, random_state=1)
    val_idx, test_idx = train_test_split(hold_idx, test_size=0.5, stratify=label[hold_idx], random_state=1)
    return train_idx, val_idx, test_idx

def make_loader(dataset, idx, batch, cfg, shuffle):
    return torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, idx), batch_size=batch, shuffle=shuffle,
        num_workers=cfg["num_workers"], pin_memory=True, collate_fn=collate_fn,
        drop_last=shuffle)

class ResBlock1d(nn.Module):
    def __init__(self, c_in, c_out, dilation=1, dropout=0.0):
        super().__init__()
        pad = dilation
        self.conv1 = nn.Conv1d(c_in, c_out, 3, padding=pad, dilation=dilation, bias=False)
        self.bn1 = nn.BatchNorm1d(c_out)
        self.conv2 = nn.Conv1d(c_out, c_out, 3, padding=pad, dilation=dilation, bias=False)
        self.bn2 = nn.BatchNorm1d(c_out)
        self.short = (nn.Identity() if c_in == c_out else
                      nn.Sequential(nn.Conv1d(c_in, c_out, 1, bias=False), nn.BatchNorm1d(c_out)))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, m):
        h = F.relu(self.bn1(self.conv1(x)))
        h = self.drop(h) * m
        h = self.bn2(self.conv2(h))
        return F.relu(h + self.short(x)) * m

class PCNN1D(nn.Module):
    def __init__(self, n_in=N_IN, width=64, n_classes=2, dropout=0.1):
        super().__init__()
        w = width
        self.stem = nn.Sequential(nn.Conv1d(n_in, w, 1, bias=False), nn.BatchNorm1d(w), nn.ReLU())
        self.blocks = nn.ModuleList([
            ResBlock1d(w, w, dilation=1),
            ResBlock1d(w, 2 * w, dilation=2),
            ResBlock1d(2 * w, 2 * w, dilation=4),
            ResBlock1d(2 * w, 4 * w, dilation=8, dropout=dropout),
        ])
        self.head = nn.Sequential(nn.Linear(8 * w, 4 * w), nn.ReLU(), nn.Dropout(dropout),
                                  nn.Linear(4 * w, n_classes))

    def forward(self, feats, mask):
        x = self.stem(feats)
        m = mask[:, None, :].to(x.dtype)
        x = x * m
        for blk in self.blocks:
            x = blk(x, m)
        x = x.float()
        mf = m.float()
        mean = x.sum(-1) / mf.sum(-1).clamp(min=1)
        mx = x.masked_fill(mf == 0, -1e9).max(-1).values
        mx = torch.where(mask.any(-1, keepdim=True), mx, torch.zeros_like(mx))
        return self.head(torch.cat([mean, mx], dim=-1))

class PairAttention(nn.Module):
    def __init__(self, width, heads, dropout):
        super().__init__()
        self.width, self.heads, self.head_dim = width, heads, width // heads
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.proj = nn.Linear(width, width)
        self.pair_bias = nn.Sequential(
            nn.Linear(3, width // 2), nn.GELU(), nn.Linear(width // 2, heads, bias=False))
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, tokens, mask, pair_features):
        batch, n_tokens, _ = tokens.shape
        qkv = self.qkv(tokens).reshape(batch, n_tokens, 3, self.heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        scores = (q * (self.head_dim ** -0.5)) @ k.transpose(-2, -1)
        pb = self.pair_bias(pair_features).permute(0, 3, 1, 2)
        bias = scores.new_zeros((batch, self.heads, n_tokens, n_tokens))
        bias[:, :, 1:, 1:] = pb.to(dtype=scores.dtype)
        scores = scores + bias
        key_mask = torch.cat((torch.ones((batch, 1), dtype=torch.bool, device=mask.device), mask), dim=1)
        scores = scores.masked_fill(~key_mask[:, None, None, :], torch.finfo(scores.dtype).min)
        attn = self.attn_drop(scores.softmax(dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(batch, n_tokens, self.width)
        out = self.proj_drop(self.proj(out))
        return torch.cat((out[:, :1], out[:, 1:] * mask.unsqueeze(-1).to(out.dtype)), dim=1)

class TransformerBlock(nn.Module):
    def __init__(self, width, heads, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attn = PairAttention(width, heads, dropout)
        self.norm2 = nn.LayerNorm(width)
        self.ffn = nn.Sequential(
            nn.Linear(width, 4 * width), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(4 * width, width), nn.Dropout(dropout))

    def forward(self, tokens, mask, pair_features):
        tokens = tokens + self.attn(self.norm1(tokens), mask, pair_features)
        tokens = tokens + self.ffn(self.norm2(tokens))
        return torch.cat((tokens[:, :1], tokens[:, 1:] * mask.unsqueeze(-1).to(tokens.dtype)), dim=1)

class ParticleTransformer(nn.Module):
    def __init__(self, n_in=N_IN, width=128, heads=8, layers=6, n_classes=2, dropout=0.1):
        super().__init__()
        self.embed = nn.Sequential(nn.Linear(n_in, width), nn.LayerNorm(width), nn.GELU())
        self.cls_token = nn.Parameter(torch.zeros(1, 1, width))
        self.blocks = nn.ModuleList(TransformerBlock(width, heads, dropout) for _ in range(layers))
        self.norm = nn.LayerNorm(width)
        self.head = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Dropout(dropout),
                                  nn.Linear(width, n_classes))
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, feats, mask):
        x = feats.transpose(1, 2).float()
        mask = mask.bool()
        particles = self.embed(x) * mask.unsqueeze(-1).to(x.dtype)
        cls = self.cls_token.expand(feats.shape[0], -1, -1).to(particles.dtype)
        tokens = torch.cat((cls, particles), dim=1)

        eta, phi, log_pt_frac = x[..., 0], x[..., 1], x[..., 4]
        deta = eta.unsqueeze(2) - eta.unsqueeze(1)
        dphi = phi.unsqueeze(2) - phi.unsqueeze(1)
        dlogpt = log_pt_frac.unsqueeze(2) - log_pt_frac.unsqueeze(1)
        pair = torch.stack((deta, dphi, dlogpt), dim=-1)
        pair = pair * (mask[:, :, None] & mask[:, None, :]).unsqueeze(-1)

        for block in self.blocks:
            tokens = block(tokens, mask, pair)
        return self.head(self.norm(tokens[:, 0]).float())

def build_model(name, cfg, n_classes):
    if name == "pcnn":
        return PCNN1D(n_in=N_IN, width=cfg["width"], n_classes=n_classes, dropout=cfg["dropout"])
    return ParticleTransformer(n_in=N_IN, width=cfg["width"], heads=cfg["heads"],
                               layers=cfg["layers"], n_classes=n_classes, dropout=cfg["dropout"])

def prepare_data(cfg, paths):
    label, jet_pt, n_const = read_headers(paths, cfg["chunk"])
    n_classes = int(label.max()) + 1
    if n_classes != 2:
        raise ValueError("expected two classes (0 = QCD, 1 = W)")
    keep = np.arange(len(label)) if cfg["no_ptmatch"] else match_pt_spectrum(jet_pt, label)
    label = label[keep]
    length = int(np.clip(n_const[keep].max(), 1, MAXC))
    mu, sd = compute_normalization(paths, keep, length, cfg["chunk"])
    dataset = JetDataset(paths, keep, length, mu, sd)
    return dataset, label, n_classes, length

def predict(model, loader, device, use_amp):
    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for feats, masks, y in loader:
            feats, masks = feats.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(feats, masks)
            probs.append(F.softmax(logits.float(), dim=-1).cpu())
            labels.append(y)
    return torch.cat(probs).numpy().astype(np.float64), torch.cat(labels).numpy()

def train(name, cfg, paths):
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = cfg["amp"] and device.type == "cuda"

    dataset, label, n_classes, _ = prepare_data(cfg, paths)
    train_idx, val_idx, _ = split_indices(label)
    train_loader = make_loader(dataset, train_idx, cfg["batch"], cfg, shuffle=True)
    val_loader = make_loader(dataset, val_idx, cfg["eval_batch"], cfg, shuffle=False)

    model = build_model(name, cfg, n_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=cfg["lr"], epochs=cfg["epochs"], steps_per_epoch=len(train_loader))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_auc = -1.0
    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        total_loss, n_seen = torch.zeros((), device=device), 0
        for feats, masks, y in train_loader:
            feats = feats.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                loss = F.cross_entropy(model(feats, masks), y)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += loss.detach().float() * len(y)
            n_seen += len(y)

        p, y_val = predict(model, val_loader, device, use_amp)
        val_auc = roc_auc_score(y_val, p[:, 1])
        print(f"Epoch {epoch:3d} | Train Loss: {total_loss.item() / n_seen:.4f} | Val AUC: {val_auc:.4f}")
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save({"model_state": model.state_dict(), "mu": dataset.mu,
                        "sd": dataset.sd, "length": dataset.L}, cfg["out"])
            print(f"Saved best model with AUC {best_auc:.4f}")
    return model

def evaluate(name, cfg, paths, roc_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = cfg["amp"] and device.type == "cuda"
    ckpt = torch.load(cfg["out"], map_location=device, weights_only=False)

    label, jet_pt, _ = read_headers(paths, cfg["chunk"])
    keep = np.arange(len(label)) if cfg["no_ptmatch"] else match_pt_spectrum(jet_pt, label)
    label = label[keep]
    _, _, test_idx = split_indices(label)

    dataset = JetDataset(paths, keep, ckpt["length"], ckpt["mu"], ckpt["sd"])
    loader = make_loader(dataset, test_idx, cfg["eval_batch"], cfg, shuffle=False)

    model = build_model(name, cfg, int(label.max()) + 1).to(device)
    model.load_state_dict(ckpt["model_state"])
    p, y = predict(model, loader, device, use_amp)

    fpr, tpr, _ = roc_curve(y, p[:, 1])
    auc = roc_auc_score(y, p[:, 1])
    acc = float(np.mean(np.argmax(p, axis=1) == y))
    print(f"Test accuracy: {acc * 100:.2f}%")
    print(f"Test ROC AUC : {auc:.4f}")

    plt.figure(figsize=(5, 5))
    plt.plot(tpr, 1.0 / np.maximum(fpr, 1e-6), drawstyle="steps-post", label=f"AUC = {auc:.4f}")
    plt.yscale("log")
    plt.xlabel(r"$\epsilon_S$")
    plt.ylabel(r"$1/\epsilon_B$")
    plt.title("W-jet vs QCD-jet ROC")
    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(roc_path, dpi=200)
    print(f"Saved ROC curve to {roc_path}")

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["train", "eval"])
    ap.add_argument("--model", choices=["pcnn", "part"], required=True)
    ap.add_argument("--data", nargs="+", default=["jets_qcd.bin", "jets_signal.bin"],
                    help="binary jet files (globs allowed)")
    ap.add_argument("--roc", default=None, help="ROC plot path for eval (default: <model>_roc.png)")
    for key, typ in [("epochs", int), ("batch", int), ("lr", float), ("width", int),
                     ("layers", int), ("heads", int), ("dropout", float), ("out", str)]:
        ap.add_argument(f"--{key}", type=typ, default=None)
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    cfg = dict(CONFIGS[args.model])
    for key in ("epochs", "batch", "lr", "width", "layers", "heads", "dropout", "out"):
        value = getattr(args, key)
        if value is not None:
            cfg[key] = value
    paths = expand(args.data)

    if args.action == "train":
        train(args.model, cfg, paths)
    else:
        evaluate(args.model, cfg, paths, args.roc or f"{args.model}_roc.png")

if __name__ == "__main__":
    main()
