# MOT_JDT_TraDeS

**TraDeS** - joint detection and tracking on KITTI-MOTS, trained from scratch.

- Paper: Wu et al., CVPR 2021; official code: `JialianW/TraDeS`
- Implementation: `common/mot/trades.py` (the module docstring below lists what was checked against the official
  code and every `[Unified]` difference)

## Results (KITTI-MOTS val)

| run | HOTA | DetA | AssA | MOTA | IDF1 | IDSW | ms / frame |
|---|---|---|---|---|---|---|---|
| trades | 0.293 | 0.238 | 0.375 | 0.247 | 0.367 | 65 | 30.4 |


Comparison with tracking-by-detection on the same network: `RESULTS.md` of `MOT_Detectors_KITTIMOTS`, section 4.

## Implementation notes

```
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
```

## Protocol (all MOT_* repositories)

- Online / causal only: every method sees the current frame and its own state; checked by re-running prefixes of
  every sequence (the output of frame t must not change when later frames are removed).
- Input 1216 x 384 (KITTI 1242 x 375 letterboxed), detectors trained from scratch on KITTI-MOTS train (12 sequences),
  30 epochs, Adam 1e-4, effective batch 8, seed 777 - the protocol of the *Physical AI Object Detection* series.
- Metrics: HOTA / DetA / AssA (TrackEval classes, car / pedestrian mean), MOTA, IDF1, IDSW.
- Latency: RTX 4070 Laptop GPU, batch 1, fp32, medians over val; pipeline = detector (forward + post-processing) +
  ReID (appearance trackers) + tracker (CPU); image decoding / letterboxing not included.
- Trackers run with their official default parameters; frame-based values given for 30 fps are scaled to 10 fps.
  A sensitivity table with one common score threshold for every tracker is reported separately (chosen on val, so it
  is analysis, not ranking).

## Data: KITTI-MOTS (box level, val = 9 sequences, 2,981 frames)

KITTI tracking images (21 training sequences) with instance / track annotations; boxes = tight boxes of the masks,
classes car and pedestrian, ignore regions (id 10000) neither rewarded nor penalised. The annotations are taken from
the **KITTI-STEP** panoptic maps (same sequences and split, car / person instances merged from KITTI-MOTS; frames,
track counts and pedestrian masks match the KITTI-MOTS paper, car masks differ by 39 of 26,899).

```bash
bash tools/mot/get_kitti_mots.sh            # KITTI tracking images + annotations (KITTI-STEP fallback)
python tools/mot/kittimots_to_voc.py        # VOC-layout copy with tracks + count report
UNIFIED_IMAGE_DATASET=kittimots python tools/mot/count_loader.py     # counts as the loaders read them
```

**Licence of the data and of every trained weight here: KITTI / KITTI-MOTS / KITTI-STEP are CC BY-NC-SA 3.0
(non-commercial).**

## Usage

```bash
pip install -r requirements.txt
pip install "git+https://github.com/JonathonLuiten/TrackEval.git@12c8791b303e0a0b50f753af204249e622d0281a"
UNIFIED_IMAGE_DATASET=kittimots UNIFIED_IMAGE_H=384 python tools/mot/train_trades.py --check     # smoke test
UNIFIED_IMAGE_DATASET=kittimots UNIFIED_IMAGE_H=384 python tools/mot/train_trades.py             # training
UNIFIED_IMAGE_DATASET=kittimots UNIFIED_IMAGE_H=384 python tools/mot/run_trades.py --causal
```

Trained weights: the GitHub Release of this repository (`trades_kittimots_e030.pth`, put it under `runs/trades/`).

## Global Collaboration

Hills Robotics is currently participating in multiple projects in Korea and internationally.

We are actively interested in building relationships with organizations that can complement our technology with global sales networks, distribution channels, system integration capabilities, manufacturing ecosystems, AI infrastructure, robotics technologies, research capabilities, or strategic investment.

Several technology and business collaborations are already underway. Some partners, projects, system architectures, and commercial programs cannot be publicly disclosed because they are subject to confidentiality and business agreements.

We believe that the emerging Physical AI ecosystem cannot be built by a single company.
It requires a global network connecting robotics companies, AI researchers, semiconductor and computing companies, sensor manufacturers, software developers, universities, system integrators, industrial customers, investors, and global distribution partners.

We have now begun our serious exploration of the Physical AI frontier, and we welcome connections with people and organizations who want to build this ecosystem together.

## Research, Contribution, and Donation

The repositories of the Statera Working Group are intended to contribute to the broader technical discussion around Physical AI.

Researchers, engineers, developers, companies, universities, and independent contributors interested in these subjects are welcome to connect with us.

If you would like to collaborate on Physical AI or autonomous robotics, participate in research or technology development, discuss international business cooperation, propose technology or platform integration, contribute technical knowledge, or support this GitHub knowledge initiative through sponsorship or donation, please contact us.

## Contact

**Richard Minsoo Go**
Head of AI, Hills Robotics
Statera Working Group

Personal / Research Contact:
[gominsoo888@gmail.com](mailto:gominsoo888@gmail.com)

Hills Robotics / Business Contact:
[minsoo.go@hillslogis.com](mailto:minsoo.go@hillslogis.com)

Hills Robotics:
https://hillslogis.com/

For inquiries regarding next-generation outdoor autonomous robotic platforms, Physical AI Spider Grid, Intelligence Cabinet, Edge AI, or On-Premise AI architectures that are not yet publicly presented on the company website, please contact us directly.
