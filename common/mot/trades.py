"""
MOT stage (2026-10-27): TraDeS (Wu et al., "Track to Detect and Segment", CVPR 2021), JDT. Box part only (the
segmentation head belongs to the video-instance stage).
Official code JialianW/TraDeS (master), checked:
  model/networks/base_model.py  current features + features of the previous clip_len-1 frames (same backbone);
                                CVA: 128-d embedding (1x1-3x3-1x1 conv) on the current and previous features, max-
                                pooled by 2, cost volume = matmul, max over width / height -> 1D volumes, softmax(x5),
                                expected displacement -> tracking offset (x2, up-sampled); MFW: previous features x
                                previous heat map (avg-pooled by 4), deformable 3x3 conv with offsets from
                                [tracking offset, chain of 4 basic blocks on (cur - prev)], attention 3x3 conv per
                                frame, softmax over frames, weighted sum -> enhanced features -> hm / reg / wh heads
  model/losses.py               CostVolumeLoss1D: softmax(out * (1 - 1D gaussian target)), focal-style on the
                                previous-centre position of each current object, averaged over the previous frames;
                                total = hm + 0.1 wh + reg + 0.5 cost_volume (no tracking head with --trades)
  dataset/generic_dataset.py    clip of clip_len frames (previous frames |dt| < max_frame_dist), same augmentation,
                                previous heat maps with hm_disturb 0.05, lost_disturb 0.4, fp_disturb 0.1
  utils/tracker.py, detector.py greedy matching of (centre + offset) to previous centres within both box areas,
                                same class and box IoU >= 0.05; unmatched detections > new_thresh are re-identified
                                with the embedding bank (cosine >= 0.3, track unseen for < window_size 20 frames,
                                EMA alpha 0.1), else new ids; max_age -1
[Unified] (in addition to the CenterTrack / FairMOT ones: our ResNet-18-DCN, detector protocol, 1216 x 384):
  * clip_len 3 (official MOT17 setting), max_frame_dist 3 and thresholds track / new / pre 0.4 from CenterTrack's
    KITTI tracking setting (TraDeS has no KITTI configuration); no amodal ltrb head (KITTI-MOTS boxes are modal).
  * the cost-volume target is stored per object (rows of the official per-position target), same loss values.
  * deformable conv = torchvision deform_conv2d with an all-ones mask (official DCN_TraDeS, same offset layout).
"""

import math
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.ops import deform_conv2d

from common.datasets.voc import letterbox, to_tensor
from common.mot.centertrack import MAX_OBJS, PairDataset, render_pre_hm
from detectors.centernet.build import (BN_MOMENTUM, DOWN_RATIO, BasicBlock, PoseResNet, _neg_loss, _sigmoid,
                                       draw_umich_gaussian, gaussian_radius, reg_l1_loss)

CLIP_LEN, EMB = 3, 128


class TraDeSNet(nn.Module):
    def __init__(self, num_classes, out_hw=(96, 304)):
        super().__init__()
        self.num_classes = num_classes
        self.net = PoseResNet(BasicBlock, [2, 2, 2, 2], {"hm": num_classes, "wh": 2, "reg": 2}, 64)
        self.embedconv = nn.Sequential(
            nn.Conv2d(64, EMB, 1), nn.BatchNorm2d(EMB, momentum=BN_MOMENTUM), nn.ReLU(inplace=True),
            nn.Conv2d(EMB, EMB, 3, padding=1), nn.BatchNorm2d(EMB, momentum=BN_MOMENTUM), nn.ReLU(inplace=True),
            nn.Conv2d(EMB, EMB, 1))
        down = nn.Sequential(nn.Conv2d(64, 128, 1, bias=False), nn.BatchNorm2d(128, momentum=BN_MOMENTUM))
        self.offset_feats = nn.Sequential(BasicBlock(64, 128, 1, down), *[BasicBlock(128, 128) for _ in range(3)])
        self.attention_cur = nn.Conv2d(64, 1, 3, padding=1)
        self.attention_prev = nn.Conv2d(64, 1, 3, padding=1)
        self.conv_offset_w = nn.Conv2d(129, 9, 3, padding=1)
        self.conv_offset_h = nn.Conv2d(129, 9, 3, padding=1)
        self.dcn_weight = nn.Parameter(torch.empty(64, 64, 3, 3))
        self.dcn_bias = nn.Parameter(torch.zeros(64))
        nn.init.kaiming_uniform_(self.dcn_weight, a=math.sqrt(5))
        h, w = out_hw[0] // 2, out_hw[1] // 2
        ii = torch.arange(h)[:, None].expand(h, w).reshape(-1)
        jj = torch.arange(w)[None, :].expand(h, w).reshape(-1)
        self.register_buffer("tm", ((torch.arange(w)[None, :] - jj[:, None]) * 2.0)[None], persistent=False)
        self.register_buffer("tv", ((torch.arange(h)[None, :] - ii[:, None]) * 2.0)[None], persistent=False)
        self.precise_bn, self.precise_bn_images, self.precise_bn_batch_size = True, 1600, 16

    def feats(self, x):
        n = self.net
        y = n.maxpool(n.relu(n.bn1(n.conv1(x))))
        return n.deconv_layers(n.layer4(n.layer3(n.layer2(n.layer1(y)))))

    def cva(self, emb_prime, feat_prev):
        B = feat_prev.shape[0]
        e = F.max_pool2d(self.embedconv(feat_prev), 2)
        hc, wc = e.shape[2:]
        c = torch.matmul(emb_prime, e.view(B, EMB, -1)).view(B, hc * wc, hc, wc)
        c_h, c_w = c.max(dim=3)[0], c.max(dim=2)[0]
        off_h = (F.softmax(c_h * 5, dim=2) * self.tv).sum(2).view(B, 1, hc, wc)
        off_w = (F.softmax(c_w * 5, dim=2) * self.tm).sum(2).view(B, 1, hc, wc)
        off = torch.cat([F.interpolate(off_w, scale_factor=2), F.interpolate(off_h, scale_factor=2)], 1)
        return c_h, c_w, off

    def mfw(self, support, off, feat_cur, feat_prev):
        B, _, H, W = feat_cur.shape
        fo = self.offset_feats(feat_cur - feat_prev)
        oh = self.conv_offset_h(torch.cat([off[:, 1:2], fo], 1))[:, :, None]
        ow = self.conv_offset_w(torch.cat([off[:, 0:1], fo], 1))[:, :, None]
        offset = torch.cat([oh, ow], 2).view(B, 18, H, W)
        mask = torch.ones(B, 9, H, W, device=support.device, dtype=support.dtype)
        return deform_conv2d(support, offset, self.dcn_weight, self.dcn_bias, padding=1, mask=mask)

    def forward(self, x, prev_feats, prev_hms):
        """prev_feats: list of backbone features of the previous frames (newest first), prev_hms: list of previous
        heat maps at input resolution (B, 1, H, W). -> outputs dict (+ volumes for the loss)."""
        return self.forward_with_feats(x, self.feats(x), prev_feats, prev_hms)

    def forward_with_feats(self, x, cur, prev_feats, prev_hms):
        B = cur.shape[0]
        emb = self.embedconv(cur)
        emb_prime = F.max_pool2d(emb, 2).view(B, EMB, -1).permute(0, 2, 1)
        props, atts, vols, trk = [cur], [self.attention_cur(cur)], [], None
        for k, (fp, hm) in enumerate(zip(prev_feats, prev_hms)):
            c_h, c_w, off = self.cva(emb_prime, fp)
            vols.append((c_h, c_w))
            if k == 0:
                trk = off
            p = self.mfw(fp * F.avg_pool2d(hm, 4), off, cur, fp)
            props.append(p)
            atts.append(self.attention_prev(p))
        w = F.softmax(torch.cat(atts, 1), dim=1)
        enhanced = sum(w[:, i:i + 1] * props[i] for i in range(len(props)))
        out = {h: getattr(self.net, h)(enhanced) for h in self.net.heads}
        out.update({"embedding": emb, "tracking": trk, "volumes": vols, "cur_feat": cur})
        return out

    def loss(self, out, t):
        hm = _neg_loss(_sigmoid(out["hm"].clone()), t["hm"])
        wh = reg_l1_loss(out["wh"], t["mask"], t["ind"], t["wh"])
        reg = reg_l1_loss(out["reg"], t["mask"], t["ind"], t["reg"])
        cv = 0.0
        for k, (c_h, c_w) in enumerate(out["volumes"]):
            cv = cv + cost_volume_loss(c_h, t[f"cv_h{k}"], t[f"cv_hi{k}"], t[f"cv_cat{k}"], t[f"cv_m{k}"])
            cv = cv + cost_volume_loss(c_w, t[f"cv_w{k}"], t[f"cv_wi{k}"], t[f"cv_cat{k}"], t[f"cv_m{k}"])
        cv = cv / len(out["volumes"])
        parts = {"hm": hm, "wh": 0.1 * wh, "reg": reg, "cost_volume": 0.5 * cv}
        return sum(parts.values()), {k: float(v) for k, v in parts.items()}


def cost_volume_loss(vol, target_rows, ind, cat, mask):
    """CostVolumeLoss1D per object: vol (B, HW, L); target_rows (B, M, L); ind (B, M) previous position;
    cat (B, M) current position; mask (B, M)."""
    rows = vol.gather(1, cat[..., None].expand(-1, -1, vol.shape[2]))        # (B, M, L)
    p = F.softmax(rows * (1 - target_rows), dim=2).gather(2, ind[..., None])[..., 0]
    m = mask.float()
    pos = (torch.log(p.clamp(min=1e-12)) * (1 - p) ** 2 * m).sum()
    n = m.sum()
    return -pos / n if n > 0 else -pos


def _gauss1d(n, c, radius):
    x = np.arange(n) - c
    sigma = (2 * radius + 1) / 6
    g = np.exp(-(x * x) / (2 * sigma * sigma))
    g[np.abs(x) > radius] = 0
    return g.astype(np.float32)


class ClipDataset(PairDataset):
    """Current frame + 2 previous frames (|dt| < 3, as CenterTrack's sampler), same augmentation; targets: hm / wh /
    reg (+ cost-volume rows for every previous frame)."""

    def __getitem__(self, i):
        seq, f = self.items[i]
        n = len(self.gt[seq])
        cand = [k for k in range(n) if abs(k - f) < self.mfd]
        prevs = [int(np.random.choice(cand)) for _ in range(CLIP_LEN - 1)]
        geo, flip = int(np.random.randint(0, 2 ** 31 - 1)), np.random.random() < self.hflip
        img, boxes, ids, cls = self._load(seq, f, geo, flip, True)
        H, W = img.shape[:2]
        oh, ow = H // DOWN_RATIO, W // DOWN_RATIO
        hc, wc = oh // 2, ow // 2
        t = {"hm": np.zeros((self.C, oh, ow), np.float32), "wh": np.zeros((MAX_OBJS, 2), np.float32),
             "reg": np.zeros((MAX_OBJS, 2), np.float32), "ind": np.zeros(MAX_OBJS, np.int64),
             "mask": np.zeros(MAX_OBJS, np.uint8)}
        cur_half = {}
        for k, (b, tid, c) in enumerate(zip(boxes[:MAX_OBJS], ids[:MAX_OBJS], cls[:MAX_OBJS])):
            bo = b / DOWN_RATIO
            bo[[0, 2]] = np.clip(bo[[0, 2]], 0, ow - 1)
            bo[[1, 3]] = np.clip(bo[[1, 3]], 0, oh - 1)
            h, w = bo[3] - bo[1], bo[2] - bo[0]
            if h <= 0 or w <= 0:
                continue
            r = max(0, int(gaussian_radius((math.ceil(h), math.ceil(w)))))
            ct = np.array([(bo[0] + bo[2]) / 2, (bo[1] + bo[3]) / 2], np.float32)
            ci = ct.astype(np.int32)
            draw_umich_gaussian(t["hm"][int(c)], ci, r)
            t["wh"][k] = w, h
            t["ind"][k] = ci[1] * ow + ci[0]
            t["reg"][k] = ct - ci
            t["mask"][k] = 1
            bh = bo / 2
            hh, ww = bh[3] - bh[1], bh[2] - bh[0]
            cth = np.array([(bh[0] + bh[2]) / 2, (bh[1] + bh[3]) / 2]).astype(np.int32)
            cur_half[int(tid)] = (k, int(min(cth[1], hc - 1) * wc + min(cth[0], wc - 1)),
                                  max(0, int(gaussian_radius((math.ceil(hh), math.ceil(ww))))))
        pimgs, phms = [], []
        for pk, fp in enumerate(prevs):
            pimg, pboxes, pids, _ = self._load(seq, fp, geo, flip, True)
            phm = np.zeros((1, H, W), np.float32)
            rows_h = np.ones((MAX_OBJS, hc), np.float32)
            rows_w = np.ones((MAX_OBJS, wc), np.float32)
            hi, wi, cat = (np.zeros(MAX_OBJS, np.int64), np.zeros(MAX_OBJS, np.int64), np.zeros(MAX_OBJS, np.int64))
            m = np.zeros(MAX_OBJS, np.uint8)
            for b, tid in zip(pboxes, pids):
                h, w = b[3] - b[1], b[2] - b[0]
                if h <= 0 or w <= 0:
                    continue
                r = max(0, int(gaussian_radius((math.ceil(h), math.ceil(w)))))
                ct0 = np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2], np.float32)
                ct = ct0 + np.random.randn(2) * self.hd * np.array([w, h])
                conf = 1.0 if np.random.random() > 0.4 else 0.0                    # TraDeS lost_disturb 0.4
                if conf > 0:
                    draw_umich_gaussian(phm[0], ct.astype(np.int32), r, k=conf)
                if np.random.random() < self.fd:
                    ct2 = ct0 + np.random.randn(2) * 0.05 * np.array([w, h])
                    draw_umich_gaussian(phm[0], ct2.astype(np.int32), r, k=conf)
                if int(tid) in cur_half:                                           # cost-volume target
                    k, pos, rad = cur_half[int(tid)]
                    pc = (ct0 / DOWN_RATIO / 2).astype(np.int32)
                    py, px = int(np.clip(pc[1], 0, hc - 1)), int(np.clip(pc[0], 0, wc - 1))
                    gy, gx = _gauss1d(hc, py, rad), _gauss1d(wc, px, rad)
                    rows_h[k], rows_w[k] = gy, gx
                    rows_h[k, py], rows_w[k, px] = 0, 0
                    hi[k], wi[k], cat[k], m[k] = py, px, pos, 1
            t.update({f"cv_h{pk}": rows_h, f"cv_w{pk}": rows_w, f"cv_hi{pk}": hi, f"cv_wi{pk}": wi,
                      f"cv_cat{pk}": cat, f"cv_m{pk}": m})
            pimgs.append(to_tensor(pimg))
            phms.append(torch.from_numpy(phm))
        return to_tensor(img), torch.stack(pimgs), torch.stack(phms), {k: torch.from_numpy(v) for k, v in t.items()}


# ------------------------------------------------------------------------------------------- inference
@torch.no_grad()
def decode(out, K=100):
    heat = out["hm"].sigmoid()
    heat = heat * (F.max_pool2d(heat, 3, stride=1, padding=1) == heat).float()
    B, C, H, W = heat.shape
    scores, idx = heat.view(B, -1).topk(K)
    cls, pix = idx // (H * W), idx % (H * W)
    ys, xs = (pix // W).float(), (pix % W).float()
    g = lambda t: t.view(B, t.shape[1], -1).gather(2, pix[:, None, :].expand(B, t.shape[1], K))
    reg, wh, trk = g(out["reg"]), g(out["wh"]), g(out["tracking"])
    emb = F.normalize(g(out["embedding"]), dim=1)
    cx, cy = xs + reg[:, 0], ys + reg[:, 1]
    rows = torch.stack([cx - wh[:, 0] / 2, cy - wh[:, 1] / 2, cx + wh[:, 0] / 2, cy + wh[:, 1] / 2, scores,
                        cls.float(), cx, cy, trk[:, 0], trk[:, 1]], -1)
    rows[..., [0, 1, 2, 3, 6, 7, 8, 9]] *= DOWN_RATIO
    return rows[0].cpu().numpy(), emb[0].T.cpu().numpy()


def _iou(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ix = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = ix * iy
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + bb[None] - inter, 1e-9)


class TraDeSTracker:
    def __init__(self, new_thresh=0.4, overlap_thresh=0.05, window_size=20, alpha=0.1):
        self.new_thresh, self.ov, self.win, self.alpha = new_thresh, overlap_thresh, window_size, alpha
        self.tracks, self.id_count, self.bank, self.cat_bank, self.ages = [], 0, {}, {}, {}

    def step(self, rows, embs):
        N, M = len(rows), len(self.tracks)
        ret, alive, matched_d = [], set(), set()
        if N and M:
            tb = np.array([t["bbox"] for t in self.tracks])
            dets = rows[:, 6:8] + rows[:, 8:10]
            prev = np.array([t["ct"] for t in self.tracks])
            dist = ((prev[None] - dets[:, None]) ** 2).sum(2)
            tsize = (tb[:, 2] - tb[:, 0]) * (tb[:, 3] - tb[:, 1])
            isize = (rows[:, 2] - rows[:, 0]) * (rows[:, 3] - rows[:, 1])
            tcat = np.array([t["class"] for t in self.tracks])
            invalid = (dist > tsize[None]) | (dist > isize[:, None]) | (rows[:, 5][:, None] != tcat[None]) | \
                      (_iou(rows[:, :4], tb) < self.ov)
            dist = dist + invalid * 1e18
            for i in range(N):
                j = int(dist[i].argmin())
                if dist[i, j] < 1e16:
                    dist[:, j] = 1e18
                    tid = self.tracks[j]["id"]
                    matched_d.add(i)
                    self.bank[tid] = self.alpha * embs[i] + (1 - self.alpha) * self.bank[tid]
                    self.cat_bank[tid] = int(rows[i, 5])
                    alive.add(tid)
                    ret.append(self._item(rows[i], tid))
        for i in range(N):
            if i in matched_d or rows[i, 4] <= self.new_thresh:
                continue
            best, bid = -2.0, None
            for tid, e in self.bank.items():
                if tid in alive or self.cat_bank[tid] != int(rows[i, 5]):
                    continue
                c = float(e @ embs[i] / (np.linalg.norm(e) * np.linalg.norm(embs[i]) + 1e-12))
                if c > best:
                    best, bid = c, tid
            if bid is not None and best >= 0.3 and self.ages.get(bid, 1e9) < self.win:
                tid = bid
                self.bank[tid] = self.alpha * embs[i] + (1 - self.alpha) * self.bank[tid]
            else:
                self.id_count += 1
                tid = self.id_count
                self.bank[tid], self.cat_bank[tid] = embs[i].copy(), int(rows[i, 5])
            alive.add(tid)
            ret.append(self._item(rows[i], tid))
        for tid in self.ages:
            self.ages[tid] += 1
        for r in ret:
            self.ages[r["id"]] = 1
        self.tracks = ret                                                  # max_age -1
        return ret

    @staticmethod
    def _item(r, tid):
        return {"bbox": r[:4].copy(), "score": float(r[4]), "class": int(r[5]), "ct": r[6:8].copy(), "id": tid}


@torch.no_grad()
def track_sequence(model, frame_paths, img_size, device, track_thresh=0.4, K=100, upto=None):
    trk = TraDeSTracker(new_thresh=track_thresh)
    feats, hms, preds, times = deque(maxlen=CLIP_LEN - 1), deque(maxlen=CLIP_LEN - 1), [], []
    for p in (frame_paths[:upto] if upto else frame_paths):
        im = Image.open(p).convert("RGB")
        lb, sc, px, py = letterbox(im, img_size)
        x = to_tensor(lb).unsqueeze(0).to(device)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        H, W = x.shape[2:]
        hm = torch.from_numpy(render_pre_hm(trk.tracks, H, W, track_thresh)).to(device)
        hms.appendleft(hm)
        cur = model.feats(x)
        fl = list(feats) or [cur]
        while len(fl) < CLIP_LEN - 1:
            fl.append(fl[-1])
        hl = list(hms)
        while len(hl) < CLIP_LEN - 1:
            hl.append(hl[-1])
        out = model.forward_with_feats(x, cur, fl, hl)
        rows, embs = decode(out, K)
        keep = rows[:, 4] > track_thresh
        tracks = trk.step(rows[keep], embs[keep])
        feats.appendleft(cur)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
        o = np.array([[*t["bbox"], t["id"], t["class"], t["score"]] for t in tracks]).reshape(-1, 7)
        if len(o):
            o[:, [0, 2]] = (o[:, [0, 2]] - px) / sc
            o[:, [1, 3]] = (o[:, [1, 3]] - py) / sc
        preds.append(o)
    return preds, times
