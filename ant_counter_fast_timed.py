#!/usr/bin/env python3
# ant_counter_fast_timed.py
#
# Fast + timed version.
# Fixes:
#  - avoids per-seg np.where in hot loops (1-pass segment stats)
#  - optional downscale via --max-side
#  - optional HV / Hough / diagonal line suppression
#  - timing logs + stage prints
#
# NEW (your request): remove "gray speckles" on side walls by filtering FINAL segments
# by mean blackhat response and/or PCA linearity:
#   --seg-min-meanbh <float>
#   --seg-min-linearity <float>
#
# Tip:
#  - If it "hangs" after "[I] starting mask CC prefilter...", that's connectedComponentsWithStats.
#    Use --max-side 1200..1400 and/or increase --open-ksize to reduce noise.

import argparse
import os
import time
from datetime import datetime

import cv2
import numpy as np


# -----------------------------
# tiny timing helper
# -----------------------------
class Timer:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self.t0 = time.perf_counter()
        self.last = self.t0
        self.rows = []

    def lap(self, name: str):
        if not self.enabled:
            return
        now = time.perf_counter()
        dt = now - self.last
        total = now - self.t0
        self.rows.append((name, dt, total))
        self.last = now
        print(f"[T] {name:<30s} {dt:7.3f}s   (total {total:7.3f}s)")

    def summary(self):
        if not self.enabled or not self.rows:
            return
        print("\n[T] ---- timing summary ----")
        for name, dt, total in self.rows:
            print(f"[T] {name:<30s} {dt:7.3f}s   (total {total:7.3f}s)")


# -----------------------------
# LAB helpers
# -----------------------------
def bgr_to_lab(img_bgr):
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, A, B = cv2.split(lab)
    return L, A, B


def chroma_weight(A_u8, B_u8, gamma=1.2, blur=0, floor=0.05):
    a = A_u8.astype(np.float32) - 128.0
    b = B_u8.astype(np.float32) - 128.0
    chroma = np.sqrt(a * a + b * b)
    c = np.clip(chroma / 181.0, 0.0, 1.0)
    w = 1.0 - c
    if gamma != 1.0:
        w = np.power(np.clip(w, 0.0, 1.0), float(gamma))
    if int(blur) > 0:
        k = int(blur)
        if k < 3:
            k = 3
        if k % 2 == 0:
            k += 1
        w = cv2.GaussianBlur(w, (k, k), 0)
    w = np.clip(w, float(floor), 1.0)
    return w


def clahe_u8(gray_u8, clip=2.0, grid=8):
    clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(int(grid), int(grid)))
    return clahe.apply(gray_u8)


def blackhat_u8(gray_u8, ksize):
    k = int(ksize)
    if k < 3:
        k = 3
    if k % 2 == 0:
        k += 1
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    bh = cv2.morphologyEx(gray_u8, cv2.MORPH_BLACKHAT, ker)
    bh = cv2.normalize(bh, None, 0, 255, cv2.NORM_MINMAX)
    return bh


def preprocess_lab_blackhat(
    img_bgr,
    blur_ksize=5,
    clahe_clip=2.0,
    clahe_grid=8,
    blackhat_ksize=41,
    multi_scale=False,
    small_blackhat_ksize=21,
    use_chroma_suppress=True,
    chroma_gamma=1.2,
    chroma_blur=0,
    chroma_floor=0.05,
):
    L, A, B = bgr_to_lab(img_bgr)

    k = int(blur_ksize)
    if k < 1:
        k = 1
    if k % 2 == 0:
        k += 1
    L_blur = cv2.GaussianBlur(L, (k, k), 0)

    Lc = clahe_u8(L_blur, clip=clahe_clip, grid=clahe_grid)

    bh_big = blackhat_u8(Lc, blackhat_ksize)
    bh_small = None
    bh_used = bh_big

    if multi_scale:
        bh_small = blackhat_u8(Lc, small_blackhat_ksize)
        bh_used = cv2.max(bh_big, bh_small)

    w = None
    bh_w = bh_used.copy()
    if use_chroma_suppress:
        w = chroma_weight(A, B, gamma=chroma_gamma, blur=chroma_blur, floor=chroma_floor)
        bh_w = (bh_used.astype(np.float32) * w).astype(np.uint8)

    return L_blur, Lc, bh_big, bh_small, bh_used, w, bh_w, A, B


# -----------------------------
# Preprocess / Mask
# -----------------------------
def make_initial_mask(
    score_u8,
    adaptive=False,
    block_size=51,
    C=2,
    min_thresh=0,
    open_ksize=3,
    open_iter=1,
    close_iter=1,
):
    if adaptive:
        bs = int(block_size)
        if bs < 3:
            bs = 3
        if bs % 2 == 0:
            bs += 1
        binmask = cv2.adaptiveThreshold(
            score_u8, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, bs, int(C)
        )
    else:
        if int(min_thresh) == 0:
            _, binmask = cv2.threshold(score_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            _, binmask = cv2.threshold(score_u8, int(min_thresh), 255, cv2.THRESH_BINARY)

    ok = int(open_ksize)
    if ok < 1:
        ok = 1
    if ok % 2 == 0:
        ok += 1
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ok, ok))
    clean = cv2.morphologyEx(binmask, cv2.MORPH_OPEN, k_open, iterations=int(open_iter))

    k3 = np.ones((3, 3), np.uint8)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, k3, iterations=int(close_iter))

    return binmask, clean


def suppress_long_lines(mask, line_len=75, dilate=1):
    """Horizontal/vertical only (fast)."""
    L = int(line_len)
    if L < 9:
        L = 9
    if L % 2 == 0:
        L += 1

    hker = cv2.getStructuringElement(cv2.MORPH_RECT, (L, 1))
    vker = cv2.getStructuringElement(cv2.MORPH_RECT, (1, L))

    horiz = cv2.morphologyEx(mask, cv2.MORPH_OPEN, hker, iterations=1)
    vert = cv2.morphologyEx(mask, cv2.MORPH_OPEN, vker, iterations=1)
    line_mask = cv2.bitwise_or(horiz, vert)

    d = int(dilate)
    if d > 0:
        k3 = np.ones((3, 3), np.uint8)
        line_mask = cv2.dilate(line_mask, k3, iterations=d)

    cleaned = cv2.subtract(mask, line_mask)
    return cleaned, line_mask


def suppress_lines_hough(
    mask_u8,
    canny1=50,
    canny2=150,
    hough_threshold=60,
    min_line_length=110,
    max_line_gap=10,
    thickness=5,
    dilate=0,
):
    """Suppress long straight lines at ANY angle using HoughLinesP on edges of the mask."""
    edges = cv2.Canny(mask_u8, int(canny1), int(canny2), apertureSize=3, L2gradient=True)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=int(hough_threshold),
        minLineLength=int(min_line_length),
        maxLineGap=int(max_line_gap),
    )

    line_mask = np.zeros_like(mask_u8)
    num = 0
    if lines is not None:
        num = int(lines.shape[0])
        for x1, y1, x2, y2 in lines[:, 0, :]:
            cv2.line(line_mask, (int(x1), int(y1)), (int(x2), int(y2)), 255, int(thickness), cv2.LINE_AA)

    if int(dilate) > 0:
        k3 = np.ones((3, 3), np.uint8)
        line_mask = cv2.dilate(line_mask, k3, iterations=int(dilate))

    cleaned = cv2.subtract(mask_u8, line_mask)
    return cleaned, line_mask, num


def _rotate_u8(img_u8, angle_deg):
    h, w = img_u8.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), float(angle_deg), 1.0)
    rot = cv2.warpAffine(img_u8, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    return rot, M


def _warp_back_u8(img_u8, M):
    Minv = cv2.invertAffineTransform(M)
    h, w = img_u8.shape[:2]
    back = cv2.warpAffine(img_u8, Minv, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    return back


def suppress_diagonal_strokes(mask_u8, angles_deg, line_len=85, thickness=3, dilate=0):
    """
    Remove long strokes at specified angles from binary mask.
    Rotate -> horizontal open -> rotate back -> accumulate -> subtract.
    """
    L = int(line_len)
    if L < 9:
        L = 9
    if L % 2 == 0:
        L += 1

    t = int(thickness)
    if t < 1:
        t = 1
    if t % 2 == 0:
        t += 1

    ker = cv2.getStructuringElement(cv2.MORPH_RECT, (L, 1))
    line_mask_total = np.zeros_like(mask_u8)

    for ang in angles_deg:
        rot, M = _rotate_u8(mask_u8, ang)
        if t > 1:
            k = cv2.getStructuringElement(cv2.MORPH_RECT, (t, t))
            rot2 = cv2.dilate(rot, k, iterations=1)
        else:
            rot2 = rot

        opened = cv2.morphologyEx(rot2, cv2.MORPH_OPEN, ker, iterations=1)
        back = _warp_back_u8(opened, M)
        line_mask_total = cv2.bitwise_or(line_mask_total, back)

    if int(dilate) > 0:
        k3 = np.ones((3, 3), np.uint8)
        line_mask_total = cv2.dilate(line_mask_total, k3, iterations=int(dilate))

    cleaned = cv2.subtract(mask_u8, line_mask_total)
    return cleaned, line_mask_total


def filter_mask_components(mask, min_area=45, max_area=200000, max_aspect=12.0, min_extent=0.02):
    """
    Filter binary mask components before watershed.
    NOTE: This is the stage that can be slow if mask is noisy & big.
    """
    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = np.zeros_like(mask)

    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if area < int(min_area) or area > int(max_area):
            continue

        aspect = max(w, h) / max(1, min(w, h))
        if aspect > float(max_aspect):
            continue

        extent = area / float(max(1, w * h))
        if extent < float(min_extent):
            continue

        out[labels == i] = 255

    return out


# -----------------------------
# Watershed
# -----------------------------
def watershed_split(img_bgr, mask, dist_thresh_ratio=0.42, sure_bg_dilate=2):
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
    dist_norm = cv2.normalize(dist, None, 0, 1.0, cv2.NORM_MINMAX)

    _, sure_fg = cv2.threshold(dist_norm, float(dist_thresh_ratio), 1.0, cv2.THRESH_BINARY)
    sure_fg = (sure_fg * 255).astype(np.uint8)

    kernel = np.ones((3, 3), np.uint8)
    sure_bg = cv2.dilate(mask, kernel, iterations=int(sure_bg_dilate))
    unknown = cv2.subtract(sure_bg, sure_fg)

    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown > 0] = 0

    markers = cv2.watershed(img_bgr, markers)
    return dist_norm, sure_fg, sure_bg, unknown, markers


# -----------------------------
# Fast segment stats from markers
# -----------------------------
class SegStats:
    __slots__ = (
        "ids", "area",
        "x0", "y0", "x1", "y1",
        "w", "h",
        "aspect", "extent",
        "meanBH", "meanChr",
        "angle", "linearity",
        "start", "end",
        "xs_sorted", "ys_sorted",
    )

    def __init__(
        self,
        ids, area, x0, y0, x1, y1, w, h,
        aspect, extent,
        meanBH, meanChr,
        angle, linearity,
        start, end,
        xs_sorted, ys_sorted,
    ):
        self.ids = ids
        self.area = area
        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1
        self.w = w
        self.h = h
        self.aspect = aspect
        self.extent = extent
        self.meanBH = meanBH
        self.meanChr = meanChr
        self.angle = angle
        self.linearity = linearity
        self.start = start
        self.end = end
        self.xs_sorted = xs_sorted
        self.ys_sorted = ys_sorted

    def slice_coords(self, seg_id: int):
        idx = np.searchsorted(self.ids, seg_id)
        if idx >= len(self.ids) or int(self.ids[idx]) != int(seg_id):
            return None, None, None
        s = int(self.start[idx])
        e = int(self.end[idx])
        return idx, self.xs_sorted[s:e], self.ys_sorted[s:e]


def _pca_from_sums(sx, sy, sxx, syy, sxy, n):
    n = np.maximum(1.0, n.astype(np.float32))
    mx = sx.astype(np.float32) / n
    my = sy.astype(np.float32) / n

    cxx = (sxx.astype(np.float32) / n) - mx * mx
    cyy = (syy.astype(np.float32) / n) - my * my
    cxy = (sxy.astype(np.float32) / n) - mx * my

    tr = cxx + cyy
    det = cxx * cyy - cxy * cxy
    disc = np.maximum(0.0, tr * tr - 4.0 * det)
    root = np.sqrt(disc)

    lam1 = 0.5 * (tr + root)
    lam2 = 0.5 * (tr - root)

    angle = 0.5 * np.degrees(np.arctan2(2.0 * cxy, cxx - cyy))
    angle = np.where(angle <= -90.0, angle + 180.0, angle)
    angle = np.where(angle > 90.0, angle - 180.0, angle)

    linearity = 1.0 - (lam2 / (lam1 + 1e-9))
    linearity = np.clip(linearity, 0.0, 1.0)

    return angle.astype(np.float32), linearity.astype(np.float32)


def compute_segstats_from_markers(markers, bh_w_u8, A_u8=None, B_u8=None):
    ys, xs = np.where(markers > 1)
    if xs.size == 0:
        return SegStats(
            ids=np.array([], np.int32),
            area=np.array([], np.int32),
            x0=np.array([], np.int32),
            y0=np.array([], np.int32),
            x1=np.array([], np.int32),
            y1=np.array([], np.int32),
            w=np.array([], np.int32),
            h=np.array([], np.int32),
            aspect=np.array([], np.float32),
            extent=np.array([], np.float32),
            meanBH=np.array([], np.float32),
            meanChr=np.array([], np.float32),
            angle=np.array([], np.float32),
            linearity=np.array([], np.float32),
            start=np.array([], np.int64),
            end=np.array([], np.int64),
            xs_sorted=np.array([], np.int32),
            ys_sorted=np.array([], np.int32),
        )

    seg = markers[ys, xs].astype(np.int32)

    order = np.argsort(seg, kind="mergesort")
    seg_s = seg[order]
    xs_s = xs[order].astype(np.int32)
    ys_s = ys[order].astype(np.int32)

    cuts = np.flatnonzero(seg_s[1:] != seg_s[:-1]) + 1
    start = np.concatenate(([0], cuts)).astype(np.int64)
    end = np.concatenate((cuts, [seg_s.size])).astype(np.int64)
    ids = seg_s[start].astype(np.int32)
    area = (end - start).astype(np.int32)

    x0 = np.minimum.reduceat(xs_s, start).astype(np.int32)
    x1 = np.maximum.reduceat(xs_s, start).astype(np.int32)
    y0 = np.minimum.reduceat(ys_s, start).astype(np.int32)
    y1 = np.maximum.reduceat(ys_s, start).astype(np.int32)
    w = (x1 - x0 + 1).astype(np.int32)
    h = (y1 - y0 + 1).astype(np.int32)

    aspect = (np.maximum(w, h) / np.maximum(1, np.minimum(w, h))).astype(np.float32)
    extent = (area.astype(np.float32) / np.maximum(1.0, (w * h).astype(np.float32))).astype(np.float32)

    bh_vals = bh_w_u8[ys_s, xs_s].astype(np.float32)
    sum_bh = np.add.reduceat(bh_vals, start).astype(np.float32)
    meanBH = (sum_bh / np.maximum(1.0, area.astype(np.float32))).astype(np.float32)

    meanChr = np.zeros_like(meanBH, dtype=np.float32)
    if A_u8 is not None and B_u8 is not None:
        a = A_u8[ys_s, xs_s].astype(np.float32) - 128.0
        b = B_u8[ys_s, xs_s].astype(np.float32) - 128.0
        chr_vals = np.sqrt(a * a + b * b).astype(np.float32)
        sum_chr = np.add.reduceat(chr_vals, start).astype(np.float32)
        meanChr = (sum_chr / np.maximum(1.0, area.astype(np.float32))).astype(np.float32)

    xs_f = xs_s.astype(np.float32)
    ys_f = ys_s.astype(np.float32)
    sx = np.add.reduceat(xs_f, start)
    sy = np.add.reduceat(ys_f, start)
    sxx = np.add.reduceat(xs_f * xs_f, start)
    syy = np.add.reduceat(ys_f * ys_f, start)
    sxy = np.add.reduceat(xs_f * ys_f, start)

    angle, linearity = _pca_from_sums(sx, sy, sxx, syy, sxy, area)

    return SegStats(
        ids=ids,
        area=area,
        x0=x0, y0=y0, x1=x1, y1=y1,
        w=w, h=h,
        aspect=aspect, extent=extent,
        meanBH=meanBH, meanChr=meanChr,
        angle=angle, linearity=linearity,
        start=start, end=end,
        xs_sorted=xs_s, ys_sorted=ys_s,
    )


# -----------------------------
# Counting / filters
# -----------------------------
def count_from_stats(area, area_min=20, area_single_max=520, area_blob_min=750):
    area = area.astype(np.int32)
    valid = area >= int(area_min)

    single = valid & (area <= int(area_single_max))
    blob = valid & ~single

    if np.any(single):
        median_single = int(np.median(area[single]))
    else:
        median_single = max(1, int(area_single_max) // 2)

    est = np.zeros_like(area, dtype=np.int32)
    est[single] = 1

    if np.any(blob):
        a = area[blob].astype(np.float32)
        e = np.round(a / float(median_single)).astype(np.int32)
        e = np.maximum(2, e)
        e = np.where(area[blob] < int(area_blob_min), 1, e)
        est[blob] = e

    total = int(est.sum())
    return total, median_single, est


def filter_segments_by_shape_from_stats(
    stats: SegStats,
    est_per_seg,
    max_aspect=14.0,
    min_aspect=1.0,
    min_extent=0.04,
    max_extent=0.95,
    max_bbox_area=1e9,
    min_meanbh=0.0,
    min_linearity=0.0,
):
    """
    Post-watershed segment filter:
      - bbox area
      - aspect range
      - extent range
      - NEW: meanBH lower bound  (kills "gray speckles" on glass/walls)
      - NEW: linearity lower bound (optional)
    """
    bbox_area = (stats.w.astype(np.float32) * stats.h.astype(np.float32))
    keep = (
        (bbox_area <= float(max_bbox_area))
        & (stats.aspect >= float(min_aspect))
        & (stats.aspect <= float(max_aspect))
        & (stats.extent >= float(min_extent))
        & (stats.extent <= float(max_extent))
        & (stats.meanBH >= float(min_meanbh))
        & (stats.linearity >= float(min_linearity))
    )

    ids = stats.ids[keep]
    est = est_per_seg[keep]
    area = stats.area[keep]
    parts = [(int(i), int(e), int(a)) for i, e, a in zip(ids, est, area)]
    total = int(est.sum())
    return parts, total


# -----------------------------
# Skeleton orientation using ximgproc
# -----------------------------
def skeleton_orientation_from_roi(segmask_u8):
    if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning"):
        sk = cv2.ximgproc.thinning(segmask_u8)
    else:
        sk = segmask_u8.copy()

    ys, xs = np.where(sk > 0)
    if xs.size < 5:
        ys, xs = np.where(segmask_u8 > 0)
        if xs.size < 5:
            return 0.0, 0.0

    pts = np.column_stack([xs.astype(np.float32), ys.astype(np.float32)])
    mean = pts.mean(axis=0, keepdims=True)
    X = pts - mean
    cov = (X.T @ X) / max(1, (pts.shape[0] - 1))
    w, v = np.linalg.eigh(cov)
    lam2, lam1 = float(w[0]), float(w[1])
    vec = v[:, 1]
    angle = np.degrees(np.arctan2(vec[1], vec[0]))
    while angle <= -90:
        angle += 180
    while angle > 90:
        angle -= 180
    linearity = 1.0 - (lam2 / (lam1 + 1e-9))
    return float(angle), float(np.clip(linearity, 0.0, 1.0))


# -----------------------------
# Obvious scoring (cached)
# -----------------------------
def obvious_from_stats(
    stats: SegStats,
    parts,
    chroma_hi=18.0,
    orient="pca",
    area_lo=55,
    area_hi=420,
    meanbh_lo=30,
    aspect_lo=1.6,
    extent_hi=0.85,
    linearity_lo=0.35,
    time_skeleton=False,
):
    ids = stats.ids
    id_to_idx = {int(i): k for k, i in enumerate(ids)}

    obvious_list = []
    near_rows = []
    sk_calls = 0
    sk_time = 0.0

    for seg_id, est, _a0 in parts:
        idx = id_to_idx.get(int(seg_id), None)
        if idx is None:
            continue

        area = int(stats.area[idx])
        meanBH = float(stats.meanBH[idx])
        asp = float(stats.aspect[idx])
        ext = float(stats.extent[idx])
        meanChr = float(stats.meanChr[idx])

        ang = float(stats.angle[idx])
        lin = float(stats.linearity[idx])

        if orient == "skeleton":
            cheap_ok = True
            if not (int(area_lo) <= area <= int(area_hi)):
                cheap_ok = False
            if meanBH < float(meanbh_lo):
                cheap_ok = False
            if asp < float(aspect_lo):
                cheap_ok = False
            if ext > float(extent_hi):
                cheap_ok = False
            if meanChr > float(chroma_hi):
                cheap_ok = False

            if cheap_ok:
                _idx2, xs, ys = stats.slice_coords(int(seg_id))
                if xs is not None and xs.size > 0:
                    x0, y0, x1, y1 = int(stats.x0[idx]), int(stats.y0[idx]), int(stats.x1[idx]), int(stats.y1[idx])
                    w = x1 - x0 + 1
                    h = y1 - y0 + 1
                    segmask = np.zeros((h, w), np.uint8)
                    segmask[(ys - y0), (xs - x0)] = 255

                    t0 = time.perf_counter()
                    ang, lin = skeleton_orientation_from_roi(segmask)
                    sk_time += (time.perf_counter() - t0)
                    sk_calls += 1

        dbg = dict(
            area=area,
            meanBH=meanBH,
            lin=float(lin),
            ang=float(ang),
            asp=asp,
            ext=ext,
            meanChr=float(meanChr),
            bb=(int(stats.x0[idx]), int(stats.y0[idx]), int(stats.x1[idx]), int(stats.y1[idx])),
        )

        passed = 0
        passed += int(int(area_lo) <= area <= int(area_hi))
        passed += int(meanBH >= float(meanbh_lo))
        passed += int(asp >= float(aspect_lo))
        passed += int(ext <= float(extent_hi))
        passed += int(float(lin) >= float(linearity_lo))
        passed += int(meanChr <= float(chroma_hi))
        near_rows.append((passed, meanBH, int(seg_id), dbg))

        # hard gates for "obvious"
        if area < int(area_lo) or area > int(area_hi):
            continue
        if meanBH < float(meanbh_lo):
            continue
        if asp < float(aspect_lo):
            continue
        if ext > float(extent_hi):
            continue
        if float(lin) < float(linearity_lo):
            continue
        if meanChr > float(chroma_hi):
            continue

        # soft score
        s_area = 1.0 - abs((area - 160.0) / 160.0)
        s_area = float(np.clip(s_area, 0.0, 1.0))
        s_bh = float(np.clip((meanBH - float(meanbh_lo)) / 80.0, 0.0, 1.0))
        s_lin = float(np.clip((float(lin) - float(linearity_lo)) / 0.5, 0.0, 1.0))
        s_asp = float(np.clip((asp - float(aspect_lo)) / 4.0, 0.0, 1.0))
        score01 = 0.35 * s_bh + 0.25 * s_lin + 0.20 * s_asp + 0.20 * s_area
        score01 = float(np.clip(score01, 0.0, 1.0))

        obvious_list.append(
            {"seg_id": int(seg_id), "score": score01, "angle": float(ang), "linearity": float(lin), "bbox": dbg["bb"], "dbg": dbg}
        )

    if time_skeleton and orient == "skeleton":
        avg = sk_time / max(1, sk_calls)
        print(f"[T] skeleton calls={sk_calls}, total={sk_time:.3f}s, avg={avg:.5f}s")

    return obvious_list, near_rows


# -----------------------------
# Visualization
# -----------------------------
def draw_overlay(img_bgr, stats: SegStats, parts, obvious_list=None, show_angles=False, show_ids=False, markers=None):
    out = img_bgr.copy()
    if markers is not None:
        out[markers == -1] = (0, 0, 255)

    id_to_idx = {int(i): k for k, i in enumerate(stats.ids)}

    for seg_id, est, _area in parts:
        idx = id_to_idx.get(int(seg_id), None)
        if idx is None:
            continue
        x0, y0, x1, y1 = int(stats.x0[idx]), int(stats.y0[idx]), int(stats.x1[idx]), int(stats.y1[idx])
        cv2.rectangle(out, (x0, y0), (x1, y1), (0, 255, 0), 1)
        cv2.putText(out, str(est), (x0, max(0, y0 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

    if obvious_list:
        for item in obvious_list:
            x0, y0, x1, y1 = item["bbox"]
            sc = item["score"]
            ang = item["angle"]
            seg_id = item["seg_id"]
            cv2.rectangle(out, (x0, y0), (x1, y1), (255, 0, 0), 2)
            label = f"{sc:.2f}"
            if show_angles:
                label += f" {ang:+.0f}°"
            if show_ids:
                label = f"id={seg_id} " + label
            cv2.putText(out, label, (x0, min(out.shape[0] - 2, y1 + 14)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1, cv2.LINE_AA)

    return out


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")

    now = datetime.now()
    ap.add_argument("--outdir", default=f"out_labws_{now.strftime('%d_%H_%M_%S')}")

    ap.add_argument("--max-side", type=int, default=0, help="If >0, resize so max(H,W)=this")

    # timing controls
    ap.add_argument("--timing", action="store_true")
    ap.add_argument("--time-skeleton", action="store_true")

    # LAB + blackhat
    ap.add_argument("--blur-ksize", type=int, default=5)
    ap.add_argument("--clahe-clip", type=float, default=2.0)
    ap.add_argument("--clahe-grid", type=int, default=8)

    ap.add_argument("--blackhat-ksize", type=int, default=41)
    ap.add_argument("--multi-scale", action="store_true")
    ap.add_argument("--small-blackhat-ksize", type=int, default=21)

    ap.add_argument("--no-chroma", action="store_true")
    ap.add_argument("--chroma-gamma", type=float, default=1.2)
    ap.add_argument("--chroma-blur", type=int, default=0)
    ap.add_argument("--chroma-floor", type=float, default=0.05)

    # threshold
    ap.add_argument("--adaptive", action="store_true")
    ap.add_argument("--block-size", type=int, default=51)
    ap.add_argument("--C", type=int, default=2)
    ap.add_argument("--min-thresh", type=int, default=0)

    # morph
    ap.add_argument("--open-ksize", type=int, default=3)
    ap.add_argument("--open-iter", type=int, default=1)
    ap.add_argument("--close-iter", type=int, default=1)

    # line suppression (HV)
    ap.add_argument("--line-suppress", action="store_true")
    ap.add_argument("--line-len", type=int, default=75)
    ap.add_argument("--line-dilate", type=int, default=1)

    # line suppression (any angle via Hough)
    ap.add_argument("--hough-suppress", action="store_true", help="Remove long straight lines from mask (any angle)")
    ap.add_argument("--hough-canny1", type=int, default=50)
    ap.add_argument("--hough-canny2", type=int, default=150)
    ap.add_argument("--hough-threshold", type=int, default=60)
    ap.add_argument("--hough-min-len", type=int, default=110)
    ap.add_argument("--hough-max-gap", type=int, default=10)
    ap.add_argument("--hough-thickness", type=int, default=5)
    ap.add_argument("--hough-dilate", type=int, default=0)

    # diagonal suppression (rotated morphology)
    ap.add_argument("--diag-suppress", action="store_true", help="Remove long diagonal strokes by rotated morphology")
    ap.add_argument("--diag-angles", type=str, default="-25,-20,-15,15,20,25",
                    help="Comma-separated angles in degrees, e.g. -25,-20,-15. Use = to pass negatives: --diag-angles=-30,-25")
    ap.add_argument("--diag-len", type=int, default=85)
    ap.add_argument("--diag-thick", type=int, default=3)
    ap.add_argument("--diag-dilate", type=int, default=0)

    # pre-watershed CC filter
    ap.add_argument("--mask-min-area", type=int, default=45)
    ap.add_argument("--mask-max-aspect", type=float, default=12.0)
    ap.add_argument("--mask-min-extent", type=float, default=0.02)
    ap.add_argument("--no-prefilter", action="store_true", help="Skip connectedComponents prefilter (debug/speed)")

    # watershed
    ap.add_argument("--dist-ratio", type=float, default=0.42)
    ap.add_argument("--bg-dilate", type=int, default=2)

    # counting
    ap.add_argument("--area-min", type=int, default=20)
    ap.add_argument("--area-single-max", type=int, default=520)
    ap.add_argument("--area-blob-min", type=int, default=750)

    # post-filter segments
    ap.add_argument("--seg-max-aspect", type=float, default=14.0)
    ap.add_argument("--seg-min-aspect", type=float, default=1.0)
    ap.add_argument("--seg-min-extent", type=float, default=0.04)
    ap.add_argument("--seg-max-extent", type=float, default=0.95)
    ap.add_argument("--seg-min-meanbh", type=float, default=0.0, help="Drop FINAL segments with meanBH < this")
    ap.add_argument("--seg-min-linearity", type=float, default=0.0, help="Drop FINAL segments with PCA linearity < this")

    # obvious
    ap.add_argument("--obvious", action="store_true")
    ap.add_argument("--ob-area-lo", type=int, default=55)
    ap.add_argument("--ob-area-hi", type=int, default=420)
    ap.add_argument("--ob-meanbh-lo", type=float, default=30.0)
    ap.add_argument("--ob-aspect-lo", type=float, default=1.6)
    ap.add_argument("--ob-extent-hi", type=float, default=0.85)
    ap.add_argument("--ob-linearity-lo", type=float, default=0.35)
    ap.add_argument("--ob-chroma-hi", type=float, default=18.0)
    ap.add_argument("--ob-orient", choices=["pca", "skeleton"], default="pca")
    ap.add_argument("--ob-show-angles", action="store_true")
    ap.add_argument("--ob-show-ids", action="store_true")

    # debug list
    ap.add_argument("--near-obvious", action="store_true")
    ap.add_argument("--near-top", type=int, default=60)

    args = ap.parse_args()
    T = Timer(enabled=args.timing)

    img = cv2.imread(args.image)
    if img is None:
        raise SystemExit(f"Не могу прочитать: {args.image}")
    T.lap("read image")

    # optional resize
    scale = 1.0
    if int(args.max_side) > 0:
        h0, w0 = img.shape[:2]
        m = max(h0, w0)
        if m > int(args.max_side):
            scale = float(args.max_side) / float(m)
            new_w = max(1, int(round(w0 * scale)))
            new_h = max(1, int(round(h0 * scale)))
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            print(f"[I] resized to {new_w}x{new_h} (scale={scale:.3f})")
    T.lap("resize (optional)")

    Lb, Lc, bh_big, bh_small, bh_used, w, bh_w, A_u8, B_u8 = preprocess_lab_blackhat(
        img,
        blur_ksize=args.blur_ksize,
        clahe_clip=args.clahe_clip,
        clahe_grid=args.clahe_grid,
        blackhat_ksize=args.blackhat_ksize,
        multi_scale=args.multi_scale,
        small_blackhat_ksize=args.small_blackhat_ksize,
        use_chroma_suppress=(not args.no_chroma),
        chroma_gamma=args.chroma_gamma,
        chroma_blur=args.chroma_blur,
        chroma_floor=args.chroma_floor,
    )
    T.lap("LAB+CLAHE+blackhat")

    raw, mask = make_initial_mask(
        bh_w,
        adaptive=args.adaptive,
        block_size=args.block_size,
        C=args.C,
        min_thresh=args.min_thresh,
        open_ksize=args.open_ksize,
        open_iter=args.open_iter,
        close_iter=args.close_iter,
    )
    T.lap("threshold+morph")

    # HV suppress
    hv_line_mask = None
    if args.line_suppress:
        mask, hv_line_mask = suppress_long_lines(mask, line_len=args.line_len, dilate=args.line_dilate)
    T.lap("line suppress HV (opt)")

    # Hough suppress (any angle)
    hough_line_mask = None
    hough_lines_n = 0
    if args.hough_suppress:
        mask, hough_line_mask, hough_lines_n = suppress_lines_hough(
            mask,
            canny1=args.hough_canny1,
            canny2=args.hough_canny2,
            hough_threshold=args.hough_threshold,
            min_line_length=args.hough_min_len,
            max_line_gap=args.hough_max_gap,
            thickness=args.hough_thickness,
            dilate=args.hough_dilate,
        )
        print(f"[I] Hough suppress: removed lines = {hough_lines_n}")
    T.lap("line suppress Hough (opt)")

    # diagonal strokes suppression
    diag_mask = None
    if args.diag_suppress:
        angles = [float(s.strip()) for s in args.diag_angles.split(",") if s.strip()]
        mask, diag_mask = suppress_diagonal_strokes(
            mask,
            angles_deg=angles,
            line_len=args.diag_len,
            thickness=args.diag_thick,
            dilate=args.diag_dilate,
        )
        print(f"[I] diag suppress: angles={angles}, len={args.diag_len}, thick={args.diag_thick}")
    T.lap("line suppress DIAG (opt)")

    # pre-watershed CC filter (can be slow)
    if args.no_prefilter:
        print("[I] skip mask CC prefilter", flush=True)
    else:
        print("[I] starting mask CC prefilter...", flush=True)
        mask = filter_mask_components(
            mask,
            min_area=args.mask_min_area,
            max_aspect=args.mask_max_aspect,
            min_extent=args.mask_min_extent,
        )
        print("[I] done mask CC prefilter", flush=True)
    T.lap("mask CC prefilter")

    print("[I] starting watershed...", flush=True)
    dist_norm, sure_fg, sure_bg, unknown, markers = watershed_split(
        img, mask, dist_thresh_ratio=args.dist_ratio, sure_bg_dilate=args.bg_dilate
    )
    print("[I] done watershed", flush=True)
    T.lap("watershed")

    stats = compute_segstats_from_markers(markers, bh_w, A_u8=A_u8, B_u8=B_u8)
    T.lap("segment stats (1 pass)")

    total0, median_single, est = count_from_stats(
        stats.area,
        area_min=args.area_min,
        area_single_max=args.area_single_max,
        area_blob_min=args.area_blob_min,
    )
    T.lap("count from stats")

    parts, total = filter_segments_by_shape_from_stats(
        stats,
        est,
        max_aspect=args.seg_max_aspect,
        min_aspect=args.seg_min_aspect,
        min_extent=args.seg_min_extent,
        max_extent=args.seg_max_extent,
        min_meanbh=args.seg_min_meanbh,
        min_linearity=args.seg_min_linearity,
    )
    T.lap("shape+BH filter")

    obvious_list = []
    near_rows = []
    if args.obvious or args.near_obvious:
        obvious_list, near_rows = obvious_from_stats(
            stats,
            parts,
            chroma_hi=args.ob_chroma_hi,
            orient=args.ob_orient,
            area_lo=args.ob_area_lo,
            area_hi=args.ob_area_hi,
            meanbh_lo=args.ob_meanbh_lo,
            aspect_lo=args.ob_aspect_lo,
            extent_hi=args.ob_extent_hi,
            linearity_lo=args.ob_linearity_lo,
            time_skeleton=args.time_skeleton,
        )
    T.lap("obvious/near-obvious")

    if args.near_obvious:
        near_rows.sort(key=lambda t: (t[0], t[1]), reverse=True)
        print(f"\n[Near-obvious] showing top {args.near_top} segments:")
        for passed, meanbh, seg_id, d in near_rows[: int(args.near_top)]:
            print(
                f"id={seg_id:4d} ok={passed}/6 area={d['area']:4d} meanBH={d['meanBH']:5.1f} "
                f"chr={d['meanChr']:4.1f} lin={d['lin']:.2f} ang={d['ang']:+6.0f}° "
                f"asp={d['asp']:.2f} ext={d['ext']:.2f} bb={d['bb']}"
            )

    overlay = draw_overlay(
        img,
        stats,
        parts,
        obvious_list=obvious_list if args.obvious else None,
        show_angles=args.ob_show_angles,
        show_ids=args.ob_show_ids,
        markers=markers,
    )
    T.lap("draw overlay")

    os.makedirs(args.outdir, exist_ok=True)

    cv2.imwrite(os.path.join(args.outdir, "0_L_blur.png"), Lb)
    cv2.imwrite(os.path.join(args.outdir, "0_L_clahe.png"), Lc)
    cv2.imwrite(os.path.join(args.outdir, "1_blackhat_big.png"), bh_big)
    if args.multi_scale and bh_small is not None:
        cv2.imwrite(os.path.join(args.outdir, "1b_blackhat_small.png"), bh_small)
        cv2.imwrite(os.path.join(args.outdir, "1c_blackhat_used_max.png"), bh_used)

    if w is not None:
        cv2.imwrite(os.path.join(args.outdir, "1d_chroma_weight.png"), (np.clip(w, 0, 1) * 255).astype(np.uint8))

    cv2.imwrite(os.path.join(args.outdir, "1e_blackhat_weighted.png"), bh_w)
    cv2.imwrite(os.path.join(args.outdir, "2_mask_raw.png"), raw)
    cv2.imwrite(os.path.join(args.outdir, "3_mask_final.png"), mask)

    if args.line_suppress and hv_line_mask is not None:
        cv2.imwrite(os.path.join(args.outdir, "3d_line_mask_hv.png"), hv_line_mask)
    if args.hough_suppress and hough_line_mask is not None:
        cv2.imwrite(os.path.join(args.outdir, "3e_line_mask_hough.png"), hough_line_mask)
    if args.diag_suppress and diag_mask is not None:
        cv2.imwrite(os.path.join(args.outdir, "3f_line_mask_diag.png"), diag_mask)

    cv2.imwrite(os.path.join(args.outdir, "4_dist_norm.png"), (dist_norm * 255).astype(np.uint8))
    cv2.imwrite(os.path.join(args.outdir, "5_sure_fg.png"), sure_fg)
    cv2.imwrite(os.path.join(args.outdir, "6_overlay.png"), overlay)
    T.lap("save debug images")

    obvious_count = len(obvious_list) if args.obvious else 0
    print(f"\nОценка муравьёв (с учётом слипшихся): {total}")
    if args.obvious:
        print(f"Явных муравьёв (high-confidence): {obvious_count}")
    print(f"Median площадь 'одиночного' сегмента: {median_single}")
    print(f"Сегментов после фильтров: {len(parts)}")
    if scale != 1.0:
        print(f"[I] scale={scale:.3f} (уменьшали картинку). Отключить: --max-side 0")
    if args.ob_orient == "skeleton":
        ok = hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning")
        print(f"[I] ximgproc thinning available: {ok}")
    if args.hough_suppress:
        print(f"[I] Hough lines removed: {hough_lines_n}")
        print(f"[I] Hough mask saved: {args.outdir}/3e_line_mask_hough.png")
    if args.diag_suppress:
        print(f"[I] Diag mask saved: {args.outdir}/3f_line_mask_diag.png")
    print(f"Смотри оверлей: {args.outdir}/6_overlay.png")

    T.summary()


if __name__ == "__main__":
    main()


# для запуска на 830 муравьев такие параметры
#python ant_counter_fast_timed.py image3.jpg --max-side 1800 `
#>>   --multi-scale --small-blackhat-ksize 21 `
#>>   --adaptive --block-size 71 --C 3 `
#>>   --line-suppress --line-len 75 --line-dilate 1 `
#>>   --dist-ratio 0.38 `
#>>   --mask-min-extent 0.04 --mask-min-area 55 `
#>>   --seg-min-meanbh 28 --seg-min-linearity 0.10 `
#>>   --obvious --ob-show-angles --ob-show-ids --ob-orient pca `
#>>   --ob-meanbh-lo 22 --ob-linearity-lo 0.25 --ob-area-lo 45 --ob-chroma-hi 12 `
#>>   --chroma-gamma 1.8 `
#>>   --near-obvious --timing
