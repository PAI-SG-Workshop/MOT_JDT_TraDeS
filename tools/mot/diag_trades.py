#!/usr/bin/env python3
"""
MOT stage (2026-10-28): why is TraDeS low (val HOTA 0.293, DetA 0.238) next to CenterTrack (0.396) on the same
ResNet-18-DCN?

    UNIFIED_IMAGE_H=384 python tools/mot/diag_trades.py          # ~10 min

A. detection only: AP50 (per class, VOC all-point) of the raw TraDeS detections (online, tracker heat maps, score
   >= 0.05, before the tracker) on KITTI-MOTS val, and of the saved CenterNet detections, same AP code.
   -> TraDeS << CenterNet: the detection branch (MFW fusion / training) is the problem, not the tracker.
B. the same TraDeS detections with the previous heat maps drawn from the GT of the previous frames.
   -> B >> A: the heat-map feedback (tracker output -> next frame) hurts; B ~ A: it does not.
C. motion: on consecutive val frames, displacement predicted at the GT centres (previous minus current centre,
   input pixels) by TraDeS (CVA tracking offset, GT previous heat maps) and by CenterTrack (tracking head, GT
   previous heat map); median error in pixels and relative to the box height.
   -> TraDeS error >> CenterTrack: the cost-volume association did not learn.
Detections on ignore regions (KITTI-MOTS id 10000) are neither true nor false positives (>= 50 % inside).
Output: stdout + ~/mot_logs/trades_diag.json
"""
import json
import sys
from collections import deque
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from common import config as C  # noqa: E402
from common.datasets.voc import letterbox, to_tensor  # noqa: E402
from common.mot.centertrack import CenterTrackNet, render_pre_hm  # noqa: E402
from common.mot.core import iou_matrix  # noqa: E402
from common.mot.evaluate import KITTI_CLASSES, _inside, load_kittimots  # noqa: E402
from common.mot.trades import CLIP_LEN, TraDeSNet, decode  # noqa: E402
from detectors.centernet.build import DOWN_RATIO  # noqa: E402

ROOT = C.DATASETS_ROOT / "KITTI_MOTS_VOC"
DEV = torch.device("cuda")


def ap50(gt, dets_per_seq):
    """dets_per_seq[seq]: rows [frame, x1, y1, x2, y2, score, cls] (original pixels)."""
    res = {}
    for c, name in enumerate(KITTI_CLASSES):
        scores, tp, n_gt = [], [], 0
        for seq, frames in gt.items():
            d = dets_per_seq[seq]
            d = d[d[:, 6] == c]
            for f, g in enumerate(frames):
                gb = g["boxes"][g["cls"] == c]
                ign = g["ign"][(g["ign"][:, 4] == c) | (g["ign"][:, 4] < 0)][:, :4] if len(g["ign"]) else np.zeros((0, 4))
                n_gt += len(gb)
                df = d[d[:, 0] == f]
                df = df[np.argsort(-df[:, 5])]
                used = np.zeros(len(gb), bool)
                iou = iou_matrix(df[:, 1:5], gb)
                inside = _inside(df[:, 1:5], ign) if len(ign) else np.zeros(len(df))
                for k in range(len(df)):
                    j = int(np.argmax(iou[k])) if len(gb) else -1
                    if j >= 0 and iou[k, j] >= 0.5 and not used[j]:
                        used[j] = True
                        scores.append(df[k, 5]); tp.append(1)
                    elif inside[k] >= 0.5:
                        continue
                    else:
                        scores.append(df[k, 5]); tp.append(0)
        if not scores or n_gt == 0:
            res[name] = 0.0
            continue
        o = np.argsort(-np.array(scores))
        t = np.array(tp)[o]
        ctp, cfp = np.cumsum(t), np.cumsum(1 - t)
        rec, prec = ctp / n_gt, ctp / np.maximum(ctp + cfp, 1e-9)
        mrec, mpre = np.r_[0, rec, 1], np.r_[0, prec, 0]
        for i in range(len(mpre) - 2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i + 1])
        idx = np.where(mrec[1:] != mrec[:-1])[0]
        res[name] = float(((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]).sum())
    res["mean"] = float(np.mean([res[n] for n in KITTI_CLASSES]))
    return res


def gt_tracks_for(frame):
    return [{"bbox": b, "score": 1.0} for b in frame["boxes"]]


@torch.no_grad()
def trades_dets(model, gt, paths, img_size, use_gt_hm, offsets=None):
    """online TraDeS detections; previous heat maps from the GT (use_gt_hm) or from the detections (> 0.4) of the
    previous frame (the tracker draws its tracks, i.e. the detections it kept). offsets: dict to collect C."""
    out_rows = {}
    for seq, frames in gt.items():
        feats, hms, rows_all, prev_kept = deque(maxlen=CLIP_LEN - 1), deque(maxlen=CLIP_LEN - 1), [], []
        for f, p in enumerate(paths[seq]):
            im = Image.open(p).convert("RGB")
            lb, sc, px, py = letterbox(im, img_size)
            x = to_tensor(lb).unsqueeze(0).to(DEV)
            H, W = x.shape[2:]
            if use_gt_hm:
                prev = [{"bbox": b * sc + np.array([px, py, px, py]), "score": 1.0} for b in frames[f - 1]["boxes"]] if f else []
            else:
                prev = prev_kept
            hms.appendleft(torch.from_numpy(render_pre_hm(prev, H, W, 0.4)).to(DEV))
            cur = model.feats(x)
            fl = list(feats) or [cur]
            while len(fl) < CLIP_LEN - 1:
                fl.append(fl[-1])
            hl = list(hms)
            while len(hl) < CLIP_LEN - 1:
                hl.append(hl[-1])
            out = model.forward_with_feats(x, cur, fl, hl)
            rows, _ = decode(out, 100)
            if offsets is not None and f and use_gt_hm:
                collect_offsets(out["tracking"], frames[f - 1], frames[f], sc, px, py, offsets, "trades")
            prev_kept = [{"bbox": r[:4], "score": float(r[4])} for r in rows if r[4] > 0.4]
            feats.appendleft(cur)
            r = rows[rows[:, 4] >= 0.05]
            b = r[:, :4].copy()
            b[:, [0, 2]] = (b[:, [0, 2]] - px) / sc
            b[:, [1, 3]] = (b[:, [1, 3]] - py) / sc
            rows_all.append(np.c_[np.full(len(r), f), b, r[:, 4], r[:, 5]])
        out_rows[seq] = np.concatenate(rows_all)
    return out_rows


def collect_offsets(trk, g_prev, g_cur, sc, px, py, acc, key):
    """trk (1, 2, h, w) displacement map in output pixels; GT displacement prev - cur in input pixels."""
    prev = {int(i): b for i, b in zip(g_prev["ids"], g_prev["boxes"])}
    h, w = trk.shape[2:]
    for i, b in zip(g_cur["ids"], g_cur["boxes"]):
        if int(i) not in prev:
            continue
        bi, pb = b * sc + np.array([px, py, px, py]), prev[int(i)] * sc + np.array([px, py, px, py])
        c = np.array([(bi[0] + bi[2]) / 2, (bi[1] + bi[3]) / 2])
        pc = np.array([(pb[0] + pb[2]) / 2, (pb[1] + pb[3]) / 2])
        ci = (c / DOWN_RATIO).astype(int)
        ci = np.clip(ci, 0, [w - 1, h - 1])
        pred = trk[0, :, ci[1], ci[0]].float().cpu().numpy() * DOWN_RATIO
        err = float(np.linalg.norm(pred - (pc - c)))
        acc.setdefault(key, []).append((err, float(bi[3] - bi[1]), float(np.linalg.norm(pc - c))))


@torch.no_grad()
def centertrack_offsets(gt, paths, img_size, acc):
    ck = torch.load(C.RUNS_ROOT / "centertrack" / "centertrack_kittimots_e030.pth", map_location=DEV, weights_only=False)
    m = CenterTrackNet(len(KITTI_CLASSES)).to(DEV)
    m.load_state_dict(ck["model_state_dict"])
    m.eval()
    for seq, frames in gt.items():
        prev_x = None
        for f, p in enumerate(paths[seq]):
            lb, sc, px, py = letterbox(Image.open(p).convert("RGB"), img_size)
            x = to_tensor(lb).unsqueeze(0).to(DEV)
            if f:
                H, W = x.shape[2:]
                prev = [{"bbox": b * sc + np.array([px, py, px, py]), "score": 1.0} for b in frames[f - 1]["boxes"]]
                hm = torch.from_numpy(render_pre_hm(prev, H, W, 0.4)).to(DEV)
                out = m(x, prev_x, hm)
                collect_offsets(out["tracking"], frames[f - 1], frames[f], sc, px, py, acc, "centertrack")
            prev_x = x


def summarize(v):
    a = np.array(v)
    return {"pairs": len(a), "median_err_px": float(np.median(a[:, 0])), "median_err_rel_h": float(np.median(a[:, 0] / np.maximum(a[:, 1], 1))),
            "median_true_motion_px": float(np.median(a[:, 2]))}


def main():
    gt, ids = load_kittimots(ROOT, "val")
    paths = {s: [str(ROOT / "JPEGImages" / f"{i}.jpg") for i in v] for s, v in ids.items()}
    ck = torch.load(C.RUNS_ROOT / "trades" / "trades_kittimots_e030.pth", map_location=DEV, weights_only=False)
    model = TraDeSNet(len(KITTI_CLASSES)).to(DEV)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    res, offs = {}, {}
    cn = np.load(Path("~/mot_logs/dets/kittimots/centernet.npz").expanduser())
    res["A_centernet_AP50"] = ap50(gt, {s: cn[s] for s in gt})
    res["A_trades_AP50"] = ap50(gt, trades_dets(model, gt, paths, ck["img_size"], use_gt_hm=False))
    res["B_trades_gt_prev_hm_AP50"] = ap50(gt, trades_dets(model, gt, paths, ck["img_size"], use_gt_hm=True, offsets=offs))
    centertrack_offsets(gt, paths, ck["img_size"], offs)
    res["C_offsets"] = {k: summarize(v) for k, v in offs.items()}
    f = lambda d: f"car {d['car']:.3f}  pedestrian {d['pedestrian']:.3f}  mean {d['mean']:.3f}"
    print("A. AP50 CenterNet (saved dets)         :", f(res["A_centernet_AP50"]))
    print("A. AP50 TraDeS (tracker heat maps)     :", f(res["A_trades_AP50"]))
    print("B. AP50 TraDeS (GT previous heat maps) :", f(res["B_trades_gt_prev_hm_AP50"]))
    for k, v in res["C_offsets"].items():
        print(f"C. offset {k:<11}: median error {v['median_err_px']:.1f} px ({v['median_err_rel_h']:.2f} x box height), "
              f"true motion {v['median_true_motion_px']:.1f} px, {v['pairs']} pairs")
    Path("~/mot_logs/trades_diag.json").expanduser().write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
