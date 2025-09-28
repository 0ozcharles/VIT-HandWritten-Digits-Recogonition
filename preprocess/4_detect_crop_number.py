import os, cv2, numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.cluster import KMeans
import json
#constants for image processing
UPSCALE = 6
BLUR_SIGMA = 0.8

os.environ["OMP_NUM_THREADS"] = "1"
#Global parameters
MIN_AREA, MIN_DENSITY, MAX_ASPECT = 60, 0.004, 10
BREAK_RATIO, MAX_RECUR = 1.2, 2
VALLEY_T, ROW_EMPTY_T, COL_BLANK_FR, MIN_SIDE_FR = 0.28, 0.85, 0.80, 0.03
KMEANS_FRAC_PIX, CENTER_DIST_FR = 0.30, 0.30
SMALL_MIN_AREA, SMALL_MAX_AREA = 70, 500
SMALL_MIN_DENS = 0.08
SMALL_MIN_ASP, SMALL_MAX_ASP = 0.35, 3.2
MIN_HALF_W_H_FR = 0.48
SMALL_MAX_FRAC_PARENT = 0.65
NESTED_THRESH, AREA_DIFF_IGNORE = 0.1, 120
OUT_SIZE, PAD = 28, 4
OUT_DIR, CHAR_DIR = "output", "chars_28"
os.makedirs(OUT_DIR, exist_ok=True);
os.makedirs(CHAR_DIR, exist_ok=True)
DILATE_KSIZE_TUPLE = (2, 2)
CONTENT_BOX_SIZE = 20
FINAL_OUTPUT_SIZE = 28

# Utility to calculate the longest blank region in a column
def longest_blank(col):
    run = cur = 0
    for v in col:
        cur = cur + 1 if v == 0 else 0
        run = max(run, cur)
    return run / len(col) if len(col) > 0 else 0


#Stage 0: Basic connected components detection
def stage0_boxes(bw):
    n, _, st, _ = cv2.connectedComponentsWithStats(bw)
    return [tuple(st[i][:4]) for i in range(1, n)]


#Projection-based box splitting function
def split_projection(x, y, w, h, b, lvl=0):
    if lvl >= MAX_RECUR or w <= h * 0.8:
        return [(x, y, w, h)]

    roi = b[y:y + h, x:x + w]
    if roi.shape[0] == 0 or roi.shape[1] == 0:
        return [(x, y, w, h)]

    col = gaussian_filter1d(roi.sum(0), 1)
    if col.max() == 0:
        return [(x, y, w, h)]

    valley = col < col.max() * VALLEY_T
    if not valley.any():
        return [(x, y, w, h)]

    groups = np.split(np.where(valley)[0], np.where(np.diff(np.where(valley)[0]) != 1)[0] + 1)
    if not groups or not any(g.size > 0 for g in groups):
        return [(x, y, w, h)]
    longest_valley = max(groups, key=len)
    if len(longest_valley) < 3:
        return [(x, y, w, h)]
    cut = int(longest_valley.mean())

    if cut <= 0 or cut >= w:
        return [(x, y, w, h)]

    lsum, rsum = roi[:, :cut].sum() / 255, roi[:, cut:].sum() / 255
    min_side_fr_thresh = MIN_SIDE_FR * (lsum + rsum)
    if min(lsum, rsum) < min_side_fr_thresh:
        return [(x, y, w, h)]

    row_empty_mean = (roi[:, cut] == 0).mean()
    if row_empty_mean < ROW_EMPTY_T:
        return [(x, y, w, h)]

    longest_blank_val = longest_blank(roi[:, cut])
    if longest_blank_val < COL_BLANK_FR:
        return [(x, y, w, h)]

    min_part_w_h_check = min(cut, w - cut)
    thresh_min_half_w_h = MIN_HALF_W_H_FR * h
    if min_part_w_h_check < thresh_min_half_w_h:
        return [(x, y, w, h)]

    min_part_w_w_check = min(cut, w - cut)
    thresh_min_part_w = 0.45 * w
    if min_part_w_w_check < thresh_min_part_w:
        return [(x, y, w, h)]

    return (
            split_projection(x, y, cut, h, b, lvl + 1) +
            split_projection(x + cut, y, w - cut, h, b, lvl + 1)
    )

#K-means based box splitting function
def split_kmeans(x, y, w, h, b):
    if w < h * BREAK_RATIO:
        return [(x, y, w, h)]

    roi_k = b[y:y + h, x:x + w]
    if roi_k.shape[0] == 0 or roi_k.shape[1] == 0:
        return [(x, y, w, h)]

    xs = np.where(roi_k)[1]
    if xs.size < 80:
        return [(x, y, w, h)]

    km = KMeans(2, n_init=5, random_state=0).fit(xs.reshape(-1, 1).astype(np.float32))
    cnt = np.bincount(km.labels_)
    if len(cnt) < 2:
        return [(x, y, w, h)]

    min_cnt_thresh = KMEANS_FRAC_PIX * sum(cnt)
    if min(cnt) < min_cnt_thresh:
        return [(x, y, w, h)]

    c0, c1 = sorted(km.cluster_centers_.flatten())
    center_dist_thresh = CENTER_DIST_FR * w
    if (c1 - c0) < center_dist_thresh:
        return [(x, y, w, h)]

    mid = int((c0 + c1) // 2)
    col_k = gaussian_filter1d(roi_k.sum(0) / 255, 1)
    range_start = max(0, mid - 3)
    range_end = min(w, mid + 7)
    if range_start >= range_end:
        cut = w // 2 if w > 0 else 0
    else:
        cut = min(range(range_start, range_end), key=lambda c: col_k[c])

    if cut <= 0 or cut >= w:
        return [(x, y, w, h)]

    lsum = roi_k[:, :cut].sum() / 255
    rsum = roi_k[:, cut:].sum() / 255
    min_sum_thresh = 0.30 * (lsum + rsum)
    if min(lsum, rsum) < min_sum_thresh:
        return [(x, y, w, h)]

    min_part_w_h_check_km = min(cut, w - cut)
    thresh_min_half_w_h_km = MIN_HALF_W_H_FR * h
    if min_part_w_h_check_km < thresh_min_half_w_h_km:
        return [(x, y, w, h)]

    min_part_w_w_check_km = min(cut, w - cut)
    aspect_ratio_char = w / h if h > 0 else float('inf')

    if aspect_ratio_char > 1.8:
        factor_w = 0.45
    else:
        factor_w = 0.25
    thresh_min_part_w_km = factor_w * w

    if min_part_w_w_check_km < thresh_min_part_w_km:
        return [(x, y, w, h)]

    return [(x, y, cut, h), (x + cut, y, w - cut, h)]


#Filtering large components
def filter_big(boxes, b):
    keep = []
    for x, y, w, h in boxes:
        if w == 0 or h == 0: continue
        if w * h < MIN_AREA:
            continue
        dens = np.count_nonzero(b[y:y + h, x:x + w]) / (w * h)
        asp = w / h
        if dens < MIN_DENSITY or asp > MAX_ASPECT or asp < 0.15:
            continue
        keep.append((x, y, w, h))
    return keep

#Detecting smaller components within larger ones
def find_small(bw, parents):
    n, _, st, _ = cv2.connectedComponentsWithStats(bw)
    out = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if not (SMALL_MIN_AREA <= a <= SMALL_MAX_AREA):
            continue
        if w == 0 or h == 0: continue
        if w * h == 0: continue
        dens = np.count_nonzero(bw[y:y + h, x:x + w]) / (w * h)
        asp = w / h

        if dens < SMALL_MIN_DENS or not (SMALL_MIN_ASP <= asp <= SMALL_MAX_ASP):
            continue
        for px, py, pw, ph in parents:
            if ph == 0 or pw == 0: continue
            if (
                    x > px + pw * 0.25 and y > py + ph * 0.25 and
                    x + w < px + pw and y + h < py + ph
            ):
                if h >= SMALL_MAX_FRAC_PARENT * ph or w >= SMALL_MAX_FRAC_PARENT * pw:
                    break
                out.append((x, y, w, h))
                break
    return out

#Removing nested boxes
def rm_nested(boxes):
    original_count = len(boxes)
    unique_boxes = sorted(list(set(boxes)), key=lambda z: z[0])
    boxes = unique_boxes

    final = []
    for i, b_coords in enumerate(boxes):
        bx, by, bw, bh = b_coords
        if bw == 0 or bh == 0: continue
        ba = bw * bh
        is_b_kept = True

        for j, p_coords in enumerate(boxes):
            if i == j:
                continue
            px, py, pw, ph = p_coords
            if pw == 0 or ph == 0: continue
            pa = pw * ph

            if ba > pa - AREA_DIFF_IGNORE:
                continue

            is_b_geometrically_inside_p = (bx >= px and by >= py and
                                           bx + bw <= px + pw and by + bh <= py + ph)
            is_b_area_much_smaller_than_p_area = (ba < NESTED_THRESH * pa)

            if is_b_geometrically_inside_p and is_b_area_much_smaller_than_p_area:
                is_b_kept = False
                break
        if is_b_kept:
            final.append(b_coords)
    return sorted(final, key=lambda z: z[0])

#Post-processing for split boxes
def post_split_boxes(bw, boxes):
    def looks_like_digit(x, y, w, h):
        if h == 0: return False
        if w < 0.35 * h: return False
        area = w * h
        if area == 0: return False
        dens = np.count_nonzero(bw[y:y + h, x:x + w]) / area
        asp = w / h
        return area >= 150 and 0.3 <= asp <= 3.2 and dens >= 0.08

    refined = []
    for x, y, w, h in boxes:
        if w == 0 or h == 0:
            refined.append((x, y, w, h))
            continue

        roi = bw[y:y + h, x:x + w]
        if roi.shape[0] == 0 or roi.shape[1] == 0:
            refined.append((x, y, w, h))
            continue

        n, _, st, _ = cv2.connectedComponentsWithStats(roi)
        stats = st[1:][st[1:, 0].argsort()] if n > 2 else []
        min_gap = min([
            stats[i + 1, 0] - (stats[i, 0] + stats[i, 2])
            for i in range(len(stats) - 1)
        ], default=999)

        if not (w >= 1.6 * h and min_gap <= 1):
            refined.append((x, y, w, h))
            continue

        col = gaussian_filter1d(roi.sum(0), 1)
        maxv = col.max()
        if maxv == 0:
            refined.append((x, y, w, h))
            continue

        valley_idx = np.where(col < maxv * 0.4)[0]
        if valley_idx.size:
            segs = np.split(valley_idx, np.where(np.diff(valley_idx) != 1)[0] + 1)
            if not segs or not any(s.size > 0 for s in segs):
                refined.append((x, y, w, h))
                continue
            deep = max(segs, key=len)
            cut0 = int(np.mean(deep))
            left = max(0, cut0 - 5)
            right = min(w - 1, cut0 + 5)
            if left >= right:
                cut = w // 2
            else:
                cut = min(range(left, right + 1), key=lambda c: col[c])
        else:
            cut = w // 2

        if cut <= 0 or cut >= w:
            refined.append((x, y, w, h))
            continue

        parts = [(x, y, cut, h), (x + cut, y, w - cut, h)]
        if parts[0][2] > 0 and parts[0][3] > 0 and parts[1][2] > 0 and parts[1][3] > 0:
            if looks_like_digit(*parts[0]) and looks_like_digit(*parts[1]):
                refined.extend(parts)
            else:
                refined.append((x, y, w, h))
        else:
            refined.append((x, y, w, h))
    return sorted(refined, key=lambda b: b[0])

#Keep the largest connected component in an ROI
def keep_largest_cc(bw_roi):
    n, lbl, stats, _ = cv2.connectedComponentsWithStats(bw_roi)
    if n <= 1: return bw_roi
    idx = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    return (lbl == idx).astype(np.uint8) * 255

#Deskew a character image
def deskew(img):
    M = cv2.moments(img)
    if abs(M['mu02']) < 1e-2: return img
    skew = M['mu11'] / M['mu02']
    h, w = img.shape
    A = np.float32([[1, skew, -0.5 * skew * h], [0, 1, 0]])
    return cv2.warpAffine(img, A, (w, h), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_NEAREST, borderValue=0)

#Convert binary mask to 28x28 normalized image
def to_28(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask > 0)
    if xs.size == 0 or ys.size == 0:
        return np.zeros((OUT_SIZE, OUT_SIZE), np.uint8)

    x1, x2 = np.min(xs), np.max(xs)
    y1, y2 = np.min(ys), np.max(ys)
    char = mask[y1:y2 + 1, x1:x2 + 1]

    char = cv2.morphologyEx(char, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))

    n, labels, stats, _ = cv2.connectedComponentsWithStats(char)
    if n > 1:
        max_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        char = (labels == max_label).astype(np.uint8) * 255

    char = cv2.dilate(char, cv2.getStructuringElement(cv2.MORPH_RECT, DILATE_KSIZE_TUPLE), iterations=1)

    h, w = char.shape
    S = max(h, w) + 2 * PAD
    canvas = np.zeros((S, S), dtype=np.uint8)
    dy, dx = (S - h) // 2, (S - w) // 2
    canvas[dy:dy + h, dx:dx + w] = char

    up = cv2.resize(canvas, (UPSCALE * S, UPSCALE * S), interpolation=cv2.INTER_LINEAR)
    up = cv2.GaussianBlur(up, (3, 3), sigmaX=BLUR_SIGMA)
    _, up = cv2.threshold(up, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    up = deskew(up)

    return cv2.resize(up, (FINAL_OUTPUT_SIZE, FINAL_OUTPUT_SIZE), interpolation=cv2.INTER_AREA)

#Main detection pipeline
def detect(img_path="mask_deskew_pca.png"):
    g = cv2.imread(img_path, 0);
    assert g is not None, img_path
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    s0 = stage0_boxes(bw)
    # ---- Stage-0 噪声过滤但不拆分 ----
    big_nosplit = filter_big(s0, bw)  # 只做面积/密度/长宽比过滤

    dbg_stage1_nosplit = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
    for i, (x1, y1, w1, h1) in enumerate(big_nosplit, 1):
        cv2.rectangle(dbg_stage1_nosplit, (x1, y1), (x1 + w1, y1 + h1),
                      (0, 255, 0), 1)  # 蓝框
        cv2.putText(dbg_stage1_nosplit, str(i), (x1, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (0, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(os.path.join(OUT_DIR, "stage1_nosplit_boxes.png"),
                dbg_stage1_nosplit)
    print("[+] 仅噪声过滤（未拆分）连通域框已保存到 "
          f"{OUT_DIR}/stage1_nosplit_boxes.png")

    dbg_stage0 = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)  # 把二值图转 3 通道
    for i, (x0, y0, w0, h0) in enumerate(s0, 1):  # 从 1 开始编号
        cv2.rectangle(dbg_stage0, (x0, y0), (x0 + w0, y0 + h0),
                      (0, 255, 0), 1)  # 红色细框
        cv2.putText(dbg_stage0, str(i), (x0, y0 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (0, 255, 255), 1, cv2.LINE_AA)  # 编号方便对照
    cv2.imwrite(os.path.join(OUT_DIR, "stage0_components.png"), dbg_stage0)
    print(f"[+] Stage-0 连通域框已保存到 {OUT_DIR}/stage0_components.png")
    s1 = []
    for x, y, w, h in s0:
        if w > h * BREAK_RATIO:
            parts = split_projection(x, y, w, h, bw)
            if len(parts) == 1:
                parts = split_kmeans(x, y, w, h, bw)
        else:
            parts = [(x, y, w, h)]
        s1.extend(parts)
    big = filter_big(s1, bw)
    final0 = sorted(list(set(big)), key=lambda z: z[0])
    final = post_split_boxes(bw, final0)
    return g, bw, final


if __name__ == "__main__":
    input_image_for_detection = "mask_deskew_pca.png"

    os.makedirs(CHAR_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)

    if not os.path.exists(input_image_for_detection):
        print(f"Error: Input image for detection '{input_image_for_detection}' not found.")
        print("Please ensure extract_from_background.py and deskew.py have been run to generate this file.")
    else:
        print(f"Using '{input_image_for_detection}' for character detection.")

    gray, bw, final_boxes_from_detect = detect(img_path=input_image_for_detection)
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    idx = 0
    ordered_boxes_for_prediction = []

    for x, y, w, h in final_boxes_from_detect:
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)

        if w > 0 and h > 0:
            roi_bw = bw[y:y + h, x:x + w]
            if roi_bw.size > 0:
                try:
                    roi = keep_largest_cc(roi_bw)
                    roi_deskewed_char = deskew(roi)
                    img28 = to_28(roi_deskewed_char)
                    cv2.imwrite(f"{CHAR_DIR}/char_{idx:02d}.png", img28)
                    ordered_boxes_for_prediction.append((int(x), int(y), int(w), int(h)))
                    idx += 1
                except Exception as e:
                    print(f"Error processing box ({x},{y},{w},{h}): {e}")
            else:
                print(f"Warning: ROI for box ({x},{y},{w},{h}) is empty or invalid after slicing.")
        else:
            print(f"Warning: Box ({x},{y},{w},{h}) has zero width or height.")

    boxes_info_path = os.path.join(CHAR_DIR, "boxes_info.json")
    with open(boxes_info_path, 'w') as f:
        json.dump({
            "image_source_for_boxes": input_image_for_detection,
            "boxes": ordered_boxes_for_prediction
        }, f)
    print(f"Bounding boxes info saved to: {boxes_info_path}")

    cv2.imwrite(f"{OUT_DIR}/boxes_final_refined.png", vis)
    print(f"Exported {idx} characters to {CHAR_DIR}/")
    print(f"Final detection visualization saved to {OUT_DIR}/boxes_final_refined.png")