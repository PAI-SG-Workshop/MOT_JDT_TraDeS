"""
MOT stage (2026-10-24): CenterTrack (Zhou, Koltun, Krahenbuhl, "Tracking Objects as Points", ECCV 2020), JDT method.
Official code: xingyizhou/CenterTrack src/lib (checked line by line):
  model/networks/dla.py      previous image and previous heat map enter through their own 7x7 conv + BN + ReLU
                             stems, added to the stem of the current image (base_layer + pre_img_layer + pre_hm_layer)
  dataset/generic_dataset.py training pairs: previous frame drawn uniformly from frames with |dt| < max_frame_dist
                             (3; dt = 0 and negative dt included); same geometric augmentation (same_aug_pre);
                             previous heat map from the previous frame's GT with hm_disturb 0.05 (centre jitter
                             x w / h), lost_disturb 0.2 (object left out), fp_disturb 0.1 (extra peak nearby);
                             tracking target = previous centre - current integer centre (output stride 4)
  trainer.py / losses.py     focal hm 1, wh L1 0.1, reg L1 1, tracking L1 1
  detector.py / tracker.py   test: first frame uses itself as previous image and an empty heat map; detections with
                             score > out_thresh (= track_thresh 0.4 for KITTI) are matched greedily (score order) to
                             the previous tracks by centre + predicted offset, within the box area of both and the
                             same class; unmatched detections > new_thresh (0.4) start tracks; max_age -1 (a track
                             not matched in this frame ends); previous heat map rendered from tracks > pre_thresh (0.4)
  experiments/kitti_*.sh     --pre_hm --same_aug --hm_disturb 0.05 --lost_disturb 0.2 --fp_disturb 0.1
                             --track_thresh 0.4, input 384 x 1280

[Unified] decisions:
  * backbone = our CenterNet ResNet-18-DCN (detectors/centernet/build.py). The official ResNet-DCN of CenterTrack
    has no tracking inputs (imgpre2feats only in DLA); the DLA way (extra 7x7 stems added after the first stem) is
    applied to it, so the CenterNet detector and CenterTrack share the backbone and differ only in the tracking
    inputs and the offset head.
  * training protocol of the detectors: from scratch, 30 epochs, Adam 1e-4, effective batch 8, seed 777, common
    augmentation (scale / shift 0.2, h-flip 0.5 - same for both frames; colour jitter per frame), input 1216 x 384,
    PreciseBN as CenterNet. (Official: nuScenes pre-training, 70 epochs, batch 16.)
  * KITTI-MOTS boxes (tight boxes of the masks), ignore regions are not drawn (like the detectors' training).
"""

import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from common.datasets.voc import hsv_jitter, letterbox, random_scale_translate, to_tensor
from detectors.centernet.build import (BN_MOMENTUM, DOWN_RATIO, BasicBlock, PoseResNet, _neg_loss, _sigmoid,
                                       draw_umich_gaussian, gaussian_radius, reg_l1_loss)

MAX_OBJS = 128


def _stem(cin):
    return nn.Sequential(nn.Conv2d(cin, 64, kernel_size=7, stride=2, padding=3, bias=False),
                         nn.BatchNorm2d(64, momentum=BN_MOMENTUM), nn.ReLU(inplace=True))


class CenterTrackNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.net = PoseResNet(BasicBlock, [2, 2, 2, 2], {"hm": num_classes, "wh": 2, "reg": 2, "tracking": 2}, 64)
        self.pre_img_layer, self.pre_hm_layer = _stem(3), _stem(1)
        self.precise_bn, self.precise_bn_images, self.precise_bn_batch_size = True, 1600, 16

    def forward(self, x, pre_img, pre_hm):
        n = self.net
        y = n.relu(n.bn1(n.conv1(x))) + self.pre_img_layer(pre_img) + self.pre_hm_layer(pre_hm)
        y = n.layer4(n.layer3(n.layer2(n.layer1(n.maxpool(y)))))
        y = n.deconv_layers(y)
        return {h: getattr(n, h)(y) for h in n.heads}

    def loss(self, out, t):
        hm = _neg_loss(_sigmoid(out["hm"].clone()), t["hm"])
        wh = reg_l1_loss(out["wh"], t["mask"], t["ind"], t["wh"])
        reg = reg_l1_loss(out["reg"], t["mask"], t["ind"], t["reg"])
        trk = reg_l1_loss(out["tracking"], t["tracking_mask"], t["ind"], t["tracking"])
        parts = {"hm": hm, "wh": 0.1 * wh, "reg": reg, "tracking": trk}
        return sum(parts.values()), {k: float(v) for k, v in parts.items()}


# ---------------------------------------------------------------------------------------------------- training data
class PairDataset(torch.utils.data.Dataset):
    """KITTI-MOTS train frames with a previous frame (|dt| < 3) -> current image, previous image, previous heat map,
    CenterNet targets + tracking offsets. gt / paths from common.mot.evaluate.load_kittimots."""

    def __init__(self, gt, paths, img_size, num_classes, train=True, aug=None, hflip_prob=0.5,
                 max_frame_dist=3, hm_disturb=0.05, lost_disturb=0.2, fp_disturb=0.1):
        self.items = [(s, f) for s in gt for f in range(len(gt[s]))]
        self.gt, self.paths, self.size, self.C = gt, paths, img_size, num_classes
        self.train, self.aug, self.hflip = train, aug or {}, hflip_prob
        self.mfd, self.hd, self.ld, self.fd = max_frame_dist, hm_disturb, lost_disturb, fp_disturb

    def __len__(self):
        return len(self.items)

    def _load(self, seq, f, geo_seed, flip, color):
        im, sc, px, py = letterbox(Image.open(self.paths[seq][f]).convert("RGB"), self.size)
        g = self.gt[seq][f]
        boxes = g["boxes"] * sc + np.array([px, py, px, py]) if len(g["boxes"]) else np.zeros((0, 4))
        arr = np.asarray(im).copy()
        keep = np.ones(len(boxes), bool)
        if self.train:
            if color:
                arr = hsv_jitter(arr, self.aug.get("hue", 0.1), self.aug.get("saturation", 1.5),
                                 self.aug.get("exposure", 1.5))
            st = np.random.get_state()
            np.random.seed(geo_seed)                           # identical geometry for both frames
            arr, boxes, keep = random_scale_translate(arr, boxes, self.aug.get("jitter", 0.2))
            np.random.set_state(st)
            if flip:
                arr = arr[:, ::-1].copy()
                W = arr.shape[1]
                boxes = boxes.copy()
                boxes[:, [0, 2]] = W - boxes[:, [2, 0]]
        if len(keep) == 0:
            keep = np.zeros(0, bool)
        return arr, boxes[keep], g["ids"][keep], g["cls"][keep]

    def __getitem__(self, i):
        seq, f = self.items[i]
        n = len(self.gt[seq])
        if self.train:
            cand = [k for k in range(n) if abs(k - f) < self.mfd]
            fp = int(np.random.choice(cand))
        else:
            fp = max(f - 1, 0)
        geo_seed = int(np.random.randint(0, 2 ** 31 - 1))
        flip = self.train and np.random.random() < self.hflip
        img, boxes, ids, cls = self._load(seq, f, geo_seed, flip, True)
        pimg, pboxes, pids, _ = self._load(seq, fp, geo_seed, flip, True)
        H, W = img.shape[:2]
        oh, ow = H // DOWN_RATIO, W // DOWN_RATIO
        pre_hm = np.zeros((1, H, W), np.float32)
        pre_cts = {}
        for b, tid in zip(pboxes, pids):
            h, w = b[3] - b[1], b[2] - b[0]
            if h <= 0 or w <= 0:
                continue
            radius = max(0, int(gaussian_radius((math.ceil(h), math.ceil(w)))))
            ct0 = np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2], np.float32)
            pre_cts[int(tid)] = ct0 / DOWN_RATIO
            ct = ct0.copy()
            conf = 1.0
            if self.train:
                ct[0] += np.random.randn() * self.hd * w
                ct[1] += np.random.randn() * self.hd * h
                conf = 1.0 if np.random.random() > self.ld else 0.0
            ct_int = ct.astype(np.int32)
            if conf > 0:
                draw_umich_gaussian(pre_hm[0], ct_int, radius, k=conf)
            if self.train and np.random.random() < self.fd:
                ct2 = ct0 + np.random.randn(2) * 0.05 * np.array([w, h])
                draw_umich_gaussian(pre_hm[0], ct2.astype(np.int32), radius, k=conf)
        hm = np.zeros((self.C, oh, ow), np.float32)
        t = {k: np.zeros((MAX_OBJS, 2), np.float32) for k in ("wh", "reg", "tracking")}
        ind = np.zeros(MAX_OBJS, np.int64)
        mask = np.zeros(MAX_OBJS, np.uint8)
        tmask = np.zeros(MAX_OBJS, np.uint8)
        for k, (b, tid, c) in enumerate(zip(boxes[:MAX_OBJS], ids[:MAX_OBJS], cls[:MAX_OBJS])):
            bo = b / DOWN_RATIO
            bo[[0, 2]] = np.clip(bo[[0, 2]], 0, ow - 1)
            bo[[1, 3]] = np.clip(bo[[1, 3]], 0, oh - 1)
            h, w = bo[3] - bo[1], bo[2] - bo[0]
            if h <= 0 or w <= 0:
                continue
            radius = max(0, int(gaussian_radius((math.ceil(h), math.ceil(w)))))
            ct = np.array([(bo[0] + bo[2]) / 2, (bo[1] + bo[3]) / 2], np.float32)
            ct_int = ct.astype(np.int32)
            draw_umich_gaussian(hm[int(c)], ct_int, radius)
            t["wh"][k] = w, h
            ind[k] = ct_int[1] * ow + ct_int[0]
            t["reg"][k] = ct - ct_int
            mask[k] = 1
            if int(tid) in pre_cts:
                t["tracking"][k] = pre_cts[int(tid)] - ct_int
                tmask[k] = 1
        target = {"hm": hm, "ind": ind, "mask": mask, "tracking_mask": tmask, **t}
        return (to_tensor(img), to_tensor(pimg), torch.from_numpy(pre_hm),
                {k: torch.from_numpy(v) for k, v in target.items()})


# ---------------------------------------------------------------------------------------------------- inference
@torch.no_grad()
def decode(out, K=100):
    """model/decode.py generic_decode (3x3 peak NMS, top-K over classes) -> per-image numpy rows
    [x1, y1, x2, y2, score, class, ct_x, ct_y, trk_x, trk_y] in input pixels."""
    heat = out["hm"].sigmoid()
    hmax = F.max_pool2d(heat, 3, stride=1, padding=1)
    heat = heat * (hmax == heat).float()
    B, C, H, W = heat.shape
    scores, idx = heat.view(B, -1).topk(K)
    cls = idx // (H * W)
    pix = idx % (H * W)
    ys, xs = (pix // W).float(), (pix % W).float()

    def gather(t):
        return t.view(B, t.shape[1], -1).gather(2, pix[:, None, :].expand(B, t.shape[1], K))

    reg, wh, trk = gather(out["reg"]), gather(out["wh"]), gather(out["tracking"])
    cx, cy = xs + reg[:, 0], ys + reg[:, 1]
    rows = torch.stack([cx - wh[:, 0] / 2, cy - wh[:, 1] / 2, cx + wh[:, 0] / 2, cy + wh[:, 1] / 2,
                        scores, cls.float(), cx, cy, trk[:, 0], trk[:, 1]], -1)
    rows[..., [0, 1, 2, 3, 6, 7, 8, 9]] *= DOWN_RATIO
    return rows.cpu().numpy()


class GreedyTracker:
    """utils/tracker.py (private detections, greedy, max_age -1)."""

    def __init__(self, new_thresh=0.4):
        self.new_thresh, self.tracks, self.id_count = new_thresh, [], 0

    def step(self, dets):
        """dets: rows sorted by score, already > out_thresh, input pixels -> list of track dicts."""
        N, M = len(dets), len(self.tracks)
        ret = []
        if N:
            cur = dets[:, 6:8] + dets[:, 8:10]
            item_size = (dets[:, 2] - dets[:, 0]) * (dets[:, 3] - dets[:, 1])
        if N and M:
            prev = np.array([t["ct"] for t in self.tracks])
            tsize = np.array([(t["bbox"][2] - t["bbox"][0]) * (t["bbox"][3] - t["bbox"][1]) for t in self.tracks])
            tcls = np.array([t["class"] for t in self.tracks])
            dist = ((prev[None] - cur[:, None]) ** 2).sum(2)
            invalid = (dist > tsize[None]) | (dist > item_size[:, None]) | (dets[:, 5][:, None] != tcls[None])
            dist = dist + invalid * 1e18
            matched_t, matched_d = set(), set()
            for i in range(N):                               # greedy_assignment
                j = int(dist[i].argmin())
                if dist[i, j] < 1e16:
                    dist[:, j] = 1e18
                    matched_d.add(i)
                    matched_t.add(j)
                    ret.append({"bbox": dets[i, :4], "score": dets[i, 4], "class": int(dets[i, 5]),
                                "ct": dets[i, 6:8], "id": self.tracks[j]["id"]})
        else:
            matched_d = set()
        for i in range(N):
            if i not in matched_d and dets[i, 4] > self.new_thresh:
                self.id_count += 1
                ret.append({"bbox": dets[i, :4], "score": dets[i, 4], "class": int(dets[i, 5]),
                            "ct": dets[i, 6:8], "id": self.id_count})
        self.tracks = ret
        return ret


def render_pre_hm(tracks, H, W, pre_thresh):
    hm = np.zeros((1, 1, H, W), np.float32)
    for t in tracks:
        if t["score"] < pre_thresh:
            continue
        b = t["bbox"]
        h, w = b[3] - b[1], b[2] - b[0]
        if h > 0 and w > 0:
            radius = max(0, int(gaussian_radius((math.ceil(h), math.ceil(w)))))
            ct = np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2], np.float32)
            draw_umich_gaussian(hm[0, 0], ct.astype(np.int32), radius)
    return hm


@torch.no_grad()
def track_sequence(model, frame_paths, img_size, device, track_thresh=0.4, K=100, upto=None):
    """Online CenterTrack over one sequence. -> per-frame [M, 7] rows in original pixels, per-frame ms
    (letterbox excluded; forward + decode + tracker + heat-map rendering timed with cuda synchronize)."""
    tracker = GreedyTracker(new_thresh=track_thresh)
    pre_img, preds, times = None, [], []
    for f, p in enumerate(frame_paths[:upto] if upto else frame_paths):
        im = Image.open(p).convert("RGB")
        lb, sc, px, py = letterbox(im, img_size)
        x = to_tensor(lb).unsqueeze(0).to(device)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        if pre_img is None:
            pre_img = x
        H, W = x.shape[2:]
        # tracker state is kept in input pixels; render from it
        pre_hm = torch.from_numpy(render_pre_hm(tracker.tracks, H, W, track_thresh)).to(device)
        out = model(x, pre_img, pre_hm)
        rows = decode(out, K)[0]
        rows = rows[rows[:, 4] > track_thresh]
        tracks = tracker.step(rows)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
        pre_img = x
        o = np.array([[*t["bbox"], t["id"], t["class"], t["score"]] for t in tracks]).reshape(-1, 7)
        if len(o):                                            # input pixels -> original pixels
            o[:, [0, 2]] = (o[:, [0, 2]] - px) / sc
            o[:, [1, 3]] = (o[:, [1, 3]] - py) / sc
        preds.append(o)
    return preds, times
