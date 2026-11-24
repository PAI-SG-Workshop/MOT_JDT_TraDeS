#!/usr/bin/env python3
"""
MOT stage (2026-10-24): train TraDeS (common/mot/trades.py) on KITTI-MOTS train, detector protocol.

    UNIFIED_IMAGE_H=384 python tools/mot/train_trades.py                  # 30 epochs
    UNIFIED_IMAGE_H=384 python tools/mot/train_trades.py --check          # 30 iterations, shapes + loss

From scratch (common init, weights_init_normal), Adam 1e-4, micro batch 2 x accum 4 = 8, seed 777, input 1216 x 384,
common augmentation (profile image_aug, h-flip 0.5), PreciseBN (1600 images) before the checkpoint is saved.
Output: <study>/runs/trades/trades_kittimots_e030.pth. Clip of 3 frames, previous frames with gradient. Log lines "Epoch NN done loss=... time=...s".
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import config as C  # noqa: E402
from common.darknet import weights_init_normal  # noqa: E402
from common.mot.trades import CLIP_LEN, ClipDataset, TraDeSNet  # noqa: E402
from common.mot.evaluate import KITTI_CLASSES, load_kittimots  # noqa: E402


def to_dev(t, d):
    return {k: v.to(d, non_blocking=True) for k, v in t.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--micro", type=int, default=2)
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--img-size", type=int, default=1216)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--out", default=str(C.RUNS_ROOT / "trades" / "trades_kittimots_e030.pth"))
    a = ap.parse_args()
    assert os.environ.get("UNIFIED_IMAGE_H") == "384", "set UNIFIED_IMAGE_H=384 (wide input)"
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = torch.device("cuda")
    root = C.DATASETS_ROOT / "KITTI_MOTS_VOC"
    gt, ids = load_kittimots(root, "train")
    paths = {s: [str(root / "JPEGImages" / f"{i}.jpg") for i in v] for s, v in ids.items()}
    ds = ClipDataset(gt, paths, a.img_size, len(KITTI_CLASSES), train=True, aug=C.PROFILE["image_aug"],
                     hflip_prob=C.PROFILE["hflip_prob"])
    n_obj = sum(len(fr["ids"]) for fr in sum(gt.values(), []))
    print(f"train: {len(gt)} sequences, {len(ds)} frames, {n_obj} objects, classes {KITTI_CLASSES}", flush=True)
    model = TraDeSNet(len(KITTI_CLASSES)).to(device)
    model.apply(weights_init_normal)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    g = torch.Generator()
    g.manual_seed(a.seed)
    dl = torch.utils.data.DataLoader(ds, batch_size=a.micro, shuffle=True, num_workers=a.workers, drop_last=True,
                                     generator=g, pin_memory=True,
                                     worker_init_fn=lambda w: np.random.seed(a.seed * 100 + w + int(time.time()) % 1000))
    t_all = time.time()
    for ep in range(1, a.epochs + 1):
        model.train()
        t0, tot, n, parts_sum = time.time(), 0.0, 0, {}
        opt.zero_grad(set_to_none=True)
        for bi, (img, pimgs, phms, t) in enumerate(dl, 1):
            pimgs, phms = pimgs.to(device), phms.to(device)
            prev = [model.feats(pimgs[:, k]) for k in range(CLIP_LEN - 1)]   # previous frames with gradient
            out = model(img.to(device), prev, [phms[:, k] for k in range(CLIP_LEN - 1)])
            loss, parts = model.loss(out, to_dev(t, device))
            (loss / a.accum).backward()
            if bi % a.accum == 0:
                opt.step()
                opt.zero_grad(set_to_none=True)
            tot, n = tot + float(loss), n + 1
            for k, v in parts.items():
                parts_sum[k] = parts_sum.get(k, 0.0) + v
            if a.check and bi == 30:
                print("check: shapes", {k: tuple(v.shape) for k, v in out.items() if torch.is_tensor(v)},
                      "volumes", [tuple(c.shape) for c in out["volumes"][0]],
                      f"max memory {torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GB", flush=True)
                break
        p = "  ".join(f"{k}={v / max(n, 1):.4f}" for k, v in parts_sum.items())
        print(f"Epoch {ep:02d} done  loss={tot / max(n, 1):.4f}  {p}  time={time.time() - t0:.1f}s", flush=True)
        if a.check:
            break
    # PreciseBN (as CenterNet): cumulative BN statistics over 1600 training pairs, no gradient
    bns = [m for m in model.modules() if isinstance(m, torch.nn.modules.batchnorm._BatchNorm)]
    for bn in bns:
        bn.reset_running_stats()
        bn.momentum = None
    model.train()
    seen = 0
    with torch.no_grad():
        for img, pimgs, phms, _ in torch.utils.data.DataLoader(ds, batch_size=4, shuffle=True, num_workers=a.workers):
            pimgs, phms = pimgs.to(device), phms.to(device)
            model(img.to(device), [model.feats(pimgs[:, k]) for k in range(CLIP_LEN - 1)],
                  [phms[:, k] for k in range(CLIP_LEN - 1)])
            seen += img.shape[0]
            if seen >= (64 if a.check else 1600):
                break
    for bn in bns:
        bn.momentum = 0.1
    print(f"PreciseBN    : {len(bns)} BN layers re-estimated on {seen} training clips")
    out = Path(a.out if not a.check else str(Path(a.out).with_name("trades_check.pth")))
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(), "epochs": a.epochs, "img_size": a.img_size, "img_h": 384,
                "classes": KITTI_CLASSES, "seed": a.seed, "check": a.check}, out)
    print(f"-> {out}  ({(time.time() - t_all) / 60:.1f} min)")


if __name__ == "__main__":
    main()
