
"""
Color-mask + heatmap-assisted boxing with:
- per-seed, radius-limited geodesic growth in the heatmap
- morphological opening and closing
- per-component hole filling
- minimum-area and shape filtering
- tight bounding boxes
- non-maximum suppression
- GT points drawn on top of predicted boxes
"""

import csv, glob
from pathlib import Path
import cv2
import numpy as np

INPUT_DIR = r"/DRAEM/PAPER_RESULTS/CAH/CAH_DRAEM/CAH_RESULTS_DRAEM/TEST_Results/Final_Test_thresholds_det_447_seg_65_Binary_maps"
CSV_OUTPUT_PATH = r"/DRAEM/PAPER_RESULTS/CAH/CAH_DRAEM/CAH_RESULTS_DRAEM/CAH_Post_Processing/CAH_cah_FINAL_DRAEM//POINTS_Test_draem_overlays/Detections.csv"
OVERLAY_OUTPUT_DIR = r"/DRAEM/PAPER_RESULTS/CAH/CAH_DRAEM/CAH_RESULTS_DRAEM/CAH_Post_Processing/CAH_cah_FINAL_DRAEM//POINTS_Test_draem_overlays/Points_DRAEM_Points_boxes"
FILE_GLOB = "*_mask_color.png"  

GT_CSV_PATH = r"/DRAEM/PAPER_DATA/CAH_allocations_PATCHES/TEST_CAH_2019_NoM/gt.csv"

COLOR_MAP_THRESHOLD = 0.8
SCORE_THR = 0.60
MIN_AREA_PX = 25
NMS_IOU = 0.30
FG_RGB = (255, 255, 0)
MAX_COLOR_DIST = 12
EXACT_TOL = 0
GT_SOURCE_WIDTH = 512
GT_SOURCE_HEIGHT = 512
CLOSE_ITER = 2
OPEN_KERNEL = 2
OPEN_ITER = 1
CLOSE_GAP_PX = 1
FILL_HOLES = True
THIN_NARROW_MAX_PX = 2
THIN_LONG_MIN_PX = 8
FILL_RATIO_MIN = 0.18
USE_HEATMAP = True
HEATMAP_NPY_DIR = Path("/DRAEM/PAPER_RESULTS/CAH/CAH_DRAEM/Final_CAH_Inference_Results/heatmaps_npy")
HMAP_LOW = 0.35


GT_OUTER_COLOR = (0, 0, 0)     
GT_INNER_COLOR = (0, 255, 0)   


def normalize_stem(name: str) -> str:
    s = Path(str(name).strip()).stem.strip().lower()
   
    for suf in ("_mask_color", "_mask", "_binary", "_pred"):
        if s.endswith(suf):
            s = s[: -len(suf)]

    s = s.replace(".png", "").replace(".jpg", "").replace(".jpeg", "")
    return s

def _geodesic_reconstruct(marker: np.ndarray, mask_allow: np.ndarray) -> np.ndarray:
    """Grow a marker while restricting growth to the allowed mask."""
    kernel = np.ones((3, 3), dtype=np.uint8)

    current = (marker > 0).astype(np.uint8)

    allowed = (mask_allow > 0).astype(np.uint8)

    current &= allowed

    while True:
        dilated = cv2.dilate(current,kernel,iterations=1)

        updated = (dilated & allowed)

        if np.array_equal(updated,current):
            return current

        current = updated

def iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / float(area_a + area_b - inter)


def nms(boxes, scores, iou_thr: float):
    if not boxes:
        return []
    boxes = np.array(boxes, dtype=float)
    scores = np.array(scores, dtype=float)
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = int(order[0]); keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        overlaps = np.array([iou(boxes[i], boxes[j]) for j in rest])
        order = rest[overlaps <= iou_thr]
    return keep


def anomaly_map_from_color(bgr, fg_rgb=(255, 255, 0), max_color_dist=35):
    rgb = bgr[:, :, ::-1].astype(np.int16)
    fg = np.array(fg_rgb, dtype=np.int16)[None, None, :]
    diff = np.linalg.norm(rgb - fg, axis=2).astype(np.float32)
    if max_color_dist <= 0:
        return (diff == 0).astype(np.float32)
    score = 1.0 - (diff / float(max_color_dist))
    score = np.clip(score, 0.0, 1.0)
    mx, mn = float(score.max()), float(score.min())
    if mx > mn:
        score = (score - mn) / (mx - mn)
    return score.astype(np.float32)


def exact_yellow_mask_tol(bgr, fg_rgb=(255, 255, 0), tol=0):
    rgb = bgr[:, :, ::-1].astype(np.int16)
    fg = np.array(fg_rgb, dtype=np.int16)[None, None, :]
    return (np.all(np.abs(rgb - fg) <= tol, axis=2)).astype(np.uint8)


def _hm_path_for(stem: str) -> str | None:
    stem = normalize_stem(stem)
    candidates = [
        HEATMAP_NPY_DIR / f"{stem}.npy",
        HEATMAP_NPY_DIR / f"{stem}_heat.npy",
        HEATMAP_NPY_DIR / f"{stem}_heatmap.npy",
        HEATMAP_NPY_DIR / f"{stem}_hm.npy",
    ]
    for p in candidates:
        if p.exists():
            return str(p)
    return None


def _read_heatmap(
    path: str,
    target_hw,
) -> np.ndarray:
    """Load and resize a numerical heatmap."""
    hm = np.load(path).astype(np.float32)

    if hm.shape != target_hw:
        hm = cv2.resize(
            hm,
            (target_hw[1], target_hw[0]),
            interpolation=cv2.INTER_LINEAR,
        )

    return hm


def _fill_holes_bool(mask_bool: np.ndarray) -> np.ndarray:
    m8 = (mask_bool.astype(np.uint8) * 255)
    contours, _ = cv2.findContours(m8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(m8)
    cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)
    return (filled > 0)





def _components_from_bool(mask_bool: np.ndarray):
    num, labels = cv2.connectedComponents(mask_bool.astype(np.uint8), connectivity=8)
    return [(labels == i) for i in range(1, num)]


def build_components(
    color_mask: np.ndarray,
    hm: np.ndarray | None,
    min_area: int,
):
    """Build filtered anomaly components and bounding boxes."""
    k3 = np.ones(
        (3, 3),
        np.uint8,
    )

    m = color_mask.astype(np.uint8)

    if OPEN_KERNEL and OPEN_ITER > 0:
        m = cv2.morphologyEx(
            m,
            cv2.MORPH_OPEN,
            np.ones(
                (OPEN_KERNEL, OPEN_KERNEL),
                np.uint8,
            ),
            iterations=OPEN_ITER,
        )

    if CLOSE_GAP_PX > 0:
        kernel_size = (
            2 * CLOSE_GAP_PX + 1
        )

        m = cv2.morphologyEx(
            m,
            cv2.MORPH_CLOSE,
            np.ones(
                (kernel_size, kernel_size),
                np.uint8,
            ),
            iterations=CLOSE_ITER,
        )

    seed_comps = _components_from_bool(
        m > 0
    )

    grown_list: list[np.ndarray] = []

    taken = np.zeros_like(
        m,
        dtype=np.uint8,
    )

    if hm is not None:
        hm_low = (
            hm >= HMAP_LOW
        ).astype(np.uint8)

        seed_scores: list[float] = []
        seed_grow_px: list[int] = []

        for seed_component in seed_comps:
            peak = (
                float(
                    hm[
                        seed_component
                    ].max()
                )
                if np.any(
                    seed_component
                )
                else 0.0
            )

            seed_scores.append(
                peak
            )

            if peak >= 0.80:
                grow_px = 10
            elif peak >= 0.65:
                grow_px = 8
            else:
                grow_px = 5

            seed_grow_px.append(
                grow_px
            )

        order = np.argsort(
            np.asarray(
                seed_scores
            )
        )[::-1]

        for idx in order:
            seed = (
                seed_comps[idx]
                & (taken == 0)
            )

            if not np.any(seed):
                continue

            grow_px = int(
                seed_grow_px[idx]
            )

            if grow_px > 0:
                reach = cv2.dilate(
                    seed.astype(np.uint8),
                    k3,
                    iterations=grow_px,
                )

                allow = (
                    (
                        (hm_low > 0)
                        | (m > 0)
                    )
                    & (reach > 0)
                    & (taken == 0)
                )
            else:
                allow = (
                    (
                        (hm_low > 0)
                        | (m > 0)
                    )
                    & (taken == 0)
                )

            region = _geodesic_reconstruct(
                seed.astype(np.uint8),
                allow.astype(np.uint8),
            )

            if region.max() == 0:
                region = seed.astype(
                    np.uint8
                )

            taken |= region

            grown_list.append(
                region > 0
            )

    else:
        for seed_component in seed_comps:
            if np.any(
                seed_component
            ):
                grown_list.append(
                    seed_component
                )

    boxes = []
    areas = []
    scores = []

    for reg_bool in grown_list:
        comp = reg_bool.copy()

        if FILL_HOLES:
            comp = _fill_holes_bool(
                comp
            )

        ys, xs = np.where(
            comp
        )

        if ys.size == 0:
            continue

        area = int(
            ys.size
        )

        if area < min_area:
            continue

        x0 = int(xs.min())
        x1 = int(xs.max())
        y0 = int(ys.min())
        y1 = int(ys.max())

        width = (
            x1 - x0 + 1
        )

        height = (
            y1 - y0 + 1
        )

        if (
            width < 2
            or height < 2
        ):
            continue

        narrow = min(
            width,
            height,
        )

        long_side = max(
            width,
            height,
        )

        if (
            narrow
            <= THIN_NARROW_MAX_PX
            and long_side
            >= THIN_LONG_MIN_PX
        ):
            continue

        fill_ratio = (
            area
            / float(
                width * height
            )
        )

        if (
            fill_ratio
            < FILL_RATIO_MIN
        ):
            continue

        if hm is not None:
            vals = hm[
                comp
            ].astype(
                np.float32
            )

            score = float(
                np.percentile(
                    vals,
                    90,
                )
            )
        else:
            score = 1.0

        boxes.append(
            [
                x0,
                y0,
                width,
                height,
            ]
        )

        areas.append(
            area
        )

        scores.append(
            score
        )

    return (
        boxes,
        areas,
        scores,
    )


def load_gt_points(gt_csv_path: str):
    if not gt_csv_path:
        return {}

    p = Path(gt_csv_path)
    if not p.exists():
        print(f"[WARN] GT CSV not found: {gt_csv_path}")
        return {}

    with open(p, "r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            print(f"[WARN] GT CSV has no header: {gt_csv_path}")
            return {}

        cols = [c.strip().lower() for c in reader.fieldnames]

        def pick(*names):
            for n in names:
                if n in cols:
                    return n
            return None

        img_col = pick("images", "image", "base_images", "img", "file", "filename", "patch", "name")
        x_col = pick("x", "cx", "center_x", "col")
        y_col = pick("y", "cy", "center_y", "row")

        if img_col is None or x_col is None or y_col is None:
            raise ValueError(f"GT CSV must contain image + x + y columns. Found: {reader.fieldnames}")

        gt = {}
        for row in reader:
            img = (row.get(img_col, "") or "").strip()
            if not img:
                continue

            stem = normalize_stem(img)

            try:
                x = float(row[x_col])
                y = float(row[y_col])
            except Exception:
                continue

            gt.setdefault(stem, []).append((x, y))

    print(f"[INFO] Loaded GT points for {len(gt)} images from {gt_csv_path}")
    return gt


def main():
    in_dir = Path(INPUT_DIR)
    csv_path = Path(CSV_OUTPUT_PATH)
    ovl_dir = Path(OVERLAY_OUTPUT_DIR)

    csv_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    ovl_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    images = sorted(
        glob.glob(
            str(
                in_dir / FILE_GLOB
            )
        )
    )

    gt_points_by_stem = load_gt_points(
        GT_CSV_PATH
    )

    if not images:
        print(
            f"[WARN] No images matching "
            f"'{FILE_GLOB}' under {in_dir}"
        )
        return

    absolute_count_errors: list[int] = []
    evaluated_images = 0
    total_predicted_animals = 0
    total_gt_animals = 0

    with open(
        csv_path,
        "w",
        newline="",
    ) as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                "image",
                "x",
                "y",
                "w",
                "h",
                "area",
                "score",
            ]
        )

        for image_path in images:
            bgr = cv2.imread(
                image_path,
                cv2.IMREAD_COLOR,
            )

            if bgr is None:
                print(
                    f"[SKIP] Failed to read "
                    f"{image_path}"
                )
                continue

            height, width = bgr.shape[:2]

            stem = normalize_stem(
                image_path
            )

            points = (
                gt_points_by_stem.get(
                    stem,
                    [],
                )
            )

            if MAX_COLOR_DIST <= 0:
                color_mask = (
                    exact_yellow_mask_tol(
                        bgr,
                        fg_rgb=FG_RGB,
                        tol=EXACT_TOL,
                    )
                )
            else:
                anomaly_map = (
                    anomaly_map_from_color(
                        bgr,
                        fg_rgb=FG_RGB,
                        max_color_dist=MAX_COLOR_DIST,
                    )
                )

                color_mask = (
                    anomaly_map
                    >= COLOR_MAP_THRESHOLD
                ).astype(np.uint8)

            heatmap = None

            if USE_HEATMAP:
                heatmap_path = (
                    _hm_path_for(
                        stem
                    )
                )

                if heatmap_path is not None:
                    heatmap = (
                        _read_heatmap(
                            heatmap_path,
                            (
                                height,
                                width,
                            ),
                        )
                    )

            (
                wh_boxes,
                component_areas,
                component_scores,
            ) = build_components(
                color_mask,
                heatmap,
                MIN_AREA_PX,
            )

            score_keep = [
                index
                for index, score
                in enumerate(
                    component_scores
                )
                if score >= SCORE_THR
            ]

            wh_boxes = [
                wh_boxes[index]
                for index in score_keep
            ]

            component_areas = [
                component_areas[index]
                for index in score_keep
            ]

            component_scores = [
                component_scores[index]
                for index in score_keep
            ]

            boxes_xyxy = [
                [
                    x,
                    y,
                    x + box_width,
                    y + box_height,
                ]
                for (
                    x,
                    y,
                    box_width,
                    box_height,
                ) in wh_boxes
            ]

            keep = nms(
                boxes_xyxy,
                component_scores,
                iou_thr=NMS_IOU,
            )

            predicted_count = len(
                keep
            )

            gt_count = len(
                points
            )

            absolute_count_error = abs(
                predicted_count
                - gt_count
            )

            absolute_count_errors.append(
                absolute_count_error
            )

            total_predicted_animals += (
                predicted_count
            )

            total_gt_animals += (
                gt_count
            )

            evaluated_images += 1

            vis = bgr.copy()

            for index in keep:
                (
                    x1,
                    y1,
                    x2,
                    y2,
                ) = boxes_xyxy[
                    index
                ]

                box_width = (
                    x2 - x1
                )

                box_height = (
                    y2 - y1
                )

                area = (
                    component_areas[
                        index
                    ]
                )

                writer.writerow(
                    [
                        stem,
                        x1,
                        y1,
                        box_width,
                        box_height,
                        area,
                        component_scores[
                            index
                        ],
                    ]
                )

                cv2.rectangle(
                    vis,
                    (
                        x1,
                        y1,
                    ),
                    (
                        x2,
                        y2,
                    ),
                    (
                        0,
                        0,
                        255,
                    ),
                    2,
                )

            if points:
                if (
                    width
                    == GT_SOURCE_WIDTH
                    and height
                    == GT_SOURCE_HEIGHT
                ):
                    scale_x = 1.0
                    scale_y = 1.0
                else:
                    scale_x = (
                        width - 1
                    ) / float(
                        GT_SOURCE_WIDTH
                        - 1
                    )

                    scale_y = (
                        height - 1
                    ) / float(
                        GT_SOURCE_HEIGHT
                        - 1
                    )

                for gt_x, gt_y in points:
                    x_image = int(
                        round(
                            gt_x
                            * scale_x
                        )
                    )

                    y_image = int(
                        round(
                            gt_y
                            * scale_y
                        )
                    )

                    if (
                        0 <= x_image < width
                        and 0 <= y_image < height
                    ):
                        cv2.circle(
                            vis,
                            (
                                x_image,
                                y_image,
                            ),
                            5,
                            GT_OUTER_COLOR,
                            2,
                            cv2.LINE_AA,
                        )

                        cv2.circle(
                            vis,
                            (
                                x_image,
                                y_image,
                            ),
                            3,
                            GT_INNER_COLOR,
                            -1,
                            cv2.LINE_AA,
                        )

            cv2.imwrite(
                str(
                    ovl_dir
                    / f"{stem}_boxes_gt.png"
                ),
                vis,
            )

    if evaluated_images == 0:
        raise RuntimeError(
            "No valid images were evaluated."
        )

    mae = float(
        np.mean(
            absolute_count_errors
        )
    )

    print(
        f"[COUNT] Images evaluated: "
        f"{evaluated_images}"
    )

    print(
        f"[COUNT] GT animals: "
        f"{total_gt_animals}"
    )

    print(
        f"[COUNT] Predicted animals: "
        f"{total_predicted_animals}"
    )

    print(
        f"[COUNT] MAE: {mae:.4f}"
    )

    print(
        f"[DONE] CSV saved at: "
        f"{csv_path}"
    )

    print(
        f"[DONE] Overlays saved: "
        f"{ovl_dir}"
    )


if __name__ == "__main__":
    main()
