import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List
from ultralytics import YOLO


@dataclass
class ParkingResult:
    annotated_image: np.ndarray
    total_slots: int
    occupied_slots: int
    available_slots: int
    occupancy_rate: float
    congestion_level: str
    recommendation: str
    status_color: str
    engine_mode: str
    vacant_boxes: List[List[float]] = field(default_factory=list)
    occupied_boxes: List[List[float]] = field(default_factory=list)


class ParkVisionEngine:
    # COCO vehicle classes: 2=car, 3=motorcycle, 5=bus, 7=truck
    COCO_VEHICLE_CLASSES = [2, 3, 5, 7]

    def __init__(self, custom_weights_path: str, general_weights_path: str = "yolov8s.pt"):
        self.custom_model = YOLO(custom_weights_path)
        self.general_model = YOLO(general_weights_path)

    @staticmethod
    def _compute_iou(box1: List[float], box2: List[float]) -> float:
        x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
        x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if inter <= 0.0:
            return 0.0
        area1 = max(1.0, (box1[2] - box1[0]) * (box1[3] - box1[1]))
        area2 = max(1.0, (box2[2] - box2[0]) * (box2[3] - box2[1]))
        return inter / (area1 + area2 - inter)

    @staticmethod
    def _containment_ratio(inner: List[float], outer: List[float]) -> float:
        """Checks if a smaller false-positive box is mostly inside a larger vehicle box."""
        x1, y1 = max(inner[0], outer[0]), max(inner[1], outer[1])
        x2, y2 = min(inner[2], outer[2]), min(inner[3], outer[3])
        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        inner_area = max(1.0, (inner[2] - inner[0]) * (inner[3] - inner[1]))
        return inter / inner_area

    def _nms_boxes(
        self, boxes_with_scores: List[tuple], iou_thresh: float = 0.42
    ) -> List[List[float]]:
        """Merges overlapping detections from multi-angle rotational inference using confidence scores."""
        if not boxes_with_scores:
            return []
        sorted_items = sorted(boxes_with_scores, key=lambda item: item[1], reverse=True)
        keep: List[List[float]] = []
        for box, _ in sorted_items:
            if all(
                self._compute_iou(box, k) < iou_thresh
                and self._containment_ratio(box, k) < 0.75
                and self._containment_ratio(k, box) < 0.75
                for k in keep
            ):
                keep.append(box)
        return keep

    def _detect_vehicles_multi_angle(self, image_rgb: np.ndarray, conf: float) -> List[List[float]]:
        """Runs inference across 0°, 90°, 180°, and 270° rotations so overhead aerial cars are caught."""
        h, w = image_rgb.shape[:2]
        img_area = float(h * w)
        candidates: List[tuple] = []

        # Calibrate effective confidence for overhead aerial imagery
        eff_conf = min(conf, 0.28) if conf > 0.35 else conf

        rotations = [
            (None, 1.0),
            (cv2.ROTATE_90_CLOCKWISE, 0.85),
            (cv2.ROTATE_180, 0.85),
            (cv2.ROTATE_90_COUNTERCLOCKWISE, 0.85),
        ]

        for rot_code, conf_scale in rotations:
            rotated = image_rgb if rot_code is None else cv2.rotate(image_rgb, rot_code)
            pass_conf = max(0.08, eff_conf * conf_scale)

            res = self.general_model.predict(
                rotated,
                conf=pass_conf,
                imgsz=1280,
                classes=self.COCO_VEHICLE_CLASSES,
                augment=(rot_code is None),
                verbose=False,
            )[0]

            for box in res.boxes:
                rx1, ry1, rx2, ry2 = box.xyxy[0].tolist()
                score = float(box.conf[0])

                if rot_code is None:
                    ox1, oy1, ox2, oy2 = rx1, ry1, rx2, ry2
                elif rot_code == cv2.ROTATE_90_CLOCKWISE:
                    ox1, oy1, ox2, oy2 = ry1, h - rx2, ry2, h - rx1
                elif rot_code == cv2.ROTATE_180:
                    ox1, oy1, ox2, oy2 = w - rx2, h - ry2, w - rx1, h - ry1
                else:  # ROTATE_90_COUNTERCLOCKWISE
                    ox1, oy1, ox2, oy2 = w - ry2, rx1, w - ry1, rx2

                bw, bh = ox2 - ox1, oy2 - oy1
                area = bw * bh
                aspect = max(bw, bh) / max(1.0, min(bw, bh))

                # Filter out giant background boxes or extreme sliver artifacts
                if 0.0004 * img_area <= area <= 0.22 * img_area and aspect <= 4.2:
                    candidates.append(([ox1, oy1, ox2, oy2], score))

        return self._nms_boxes(candidates, iou_thresh=0.42)

    @staticmethod
    def _has_parking_lines(gray_edges: np.ndarray, box: List[float]) -> bool:
        """Checks that a candidate empty slot has painted pavement markings rather than bare road."""
        x1, y1, x2, y2 = map(int, box)
        roi = gray_edges[
            max(0, y1) : min(gray_edges.shape[0], y2),
            max(0, x1) : min(gray_edges.shape[1], x2),
        ]
        if roi.size == 0:
            return False
        edge_density = float(np.count_nonzero(roi)) / float(roi.size)
        return 0.008 <= edge_density <= 0.26

    def _infer_empty_slots_along_aisles(
        self, occupied_boxes: List[List[float]], image_rgb: np.ndarray
    ) -> List[List[float]]:
        """Infers empty slots along parked rows/columns without bridging across driving lanes."""
        if len(occupied_boxes) < 2:
            return []

        h_img, w_img = image_rgb.shape[:2]
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 40, 130)

        widths = [b[2] - b[0] for b in occupied_boxes]
        heights = [b[3] - b[1] for b in occupied_boxes]
        med_w = float(np.median(widths))
        med_h = float(np.median(heights))

        # Determine dominant stacking direction from median vehicle aspect ratio
        vertical_stacking = med_w >= med_h
        centers = [((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0) for b in occupied_boxes]
        empty_boxes: List[List[float]] = []

        for i in range(len(occupied_boxes)):
            cx1, cy1 = centers[i]
            for j in range(i + 1, len(occupied_boxes)):
                cx2, cy2 = centers[j]
                dx, dy = cx2 - cx1, cy2 - cy1

                if vertical_stacking:
                    is_same_aisle = abs(dx) < 0.55 * med_w and (1.35 * med_h <= abs(dy) <= 3.8 * med_h)
                    step_size = med_h
                    dist = abs(dy)
                else:
                    is_same_aisle = abs(dy) < 0.55 * med_h and (1.35 * med_w <= abs(dx) <= 3.8 * med_w)
                    step_size = med_w
                    dist = abs(dx)

                if not is_same_aisle:
                    continue

                num_gaps = int(round(dist / step_size)) - 1
                if num_gaps < 1 or num_gaps > 2:
                    continue

                for k in range(1, num_gaps + 1):
                    frac = k / (num_gaps + 1)
                    gx = cx1 + frac * dx
                    gy = cy1 + frac * dy
                    candidate = [
                        max(0.0, gx - med_w * 0.44),
                        max(0.0, gy - med_h * 0.44),
                        min(float(w_img), gx + med_w * 0.44),
                        min(float(h_img), gy + med_h * 0.44),
                    ]

                    if all(self._compute_iou(candidate, occ) < 0.12 for occ in occupied_boxes):
                        if all(self._compute_iou(candidate, emp) < 0.25 for emp in empty_boxes):
                            if self._has_parking_lines(edges, candidate):
                                empty_boxes.append(candidate)

        return empty_boxes

    @staticmethod
    def _draw_annotations(
        img_rgb: np.ndarray,
        occupied_boxes: List[List[float]],
        empty_boxes: List[List[float]],
    ) -> np.ndarray:
        canvas = cv2.cvtColor(img_rgb.copy(), cv2.COLOR_RGB2BGR)
        thickness = max(2, int(min(canvas.shape[:2]) / 320))
        font_scale = max(0.45, min(canvas.shape[:2]) / 1300)

        for box in empty_boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (151, 220, 61), thickness)
            cv2.putText(
                canvas, "EMPTY", (x1, max(15, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (151, 220, 61), thickness
            )

        for box in occupied_boxes:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (115, 93, 255), thickness)
            cv2.putText(
                canvas, "OCCUPIED", (x1, max(15, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, (115, 93, 255), thickness
            )

        return canvas

    def analyze(
        self,
        image_rgb: np.ndarray,
        conf_threshold: float = 0.20,
        mode: str = "Auto (Hybrid)",
        manual_capacity: int = 0,
    ) -> ParkingResult:
        occupied_boxes: List[List[float]] = []
        empty_boxes: List[List[float]] = []
        active_engine = "Custom PKLot Engine"

        # 1. Custom PKLot Evaluation
        if mode in ("Auto (Hybrid)", "Custom Dataset Mode"):
            res = self.custom_model.predict(image_rgb, conf=conf_threshold, verbose=False)[0]
            for box in res.boxes:
                cls_id = int(box.cls[0])
                coords = box.xyxy[0].tolist()
                if cls_id == 2:
                    occupied_boxes.append(coords)
                elif cls_id == 1:
                    empty_boxes.append(coords)

        # 2. Smart Fallback Check
        custom_total = len(occupied_boxes) + len(empty_boxes)
        is_suspicious_custom = len(occupied_boxes) == 0 and len(empty_boxes) < 10

        if mode == "Universal Generalization Mode" or (
            mode == "Auto (Hybrid)" and (custom_total < 5 or is_suspicious_custom)
        ):
            active_engine = "Universal Multi-Angle Engine"
            occupied_boxes = self._detect_vehicles_multi_angle(image_rgb, conf=conf_threshold)
            empty_boxes = self._infer_empty_slots_along_aisles(occupied_boxes, image_rgb)

        # 3. Calculate Insights
        occupied_count = len(occupied_boxes)
        if manual_capacity > 0:
            total_slots = max(manual_capacity, occupied_count)
            empty_count = total_slots - occupied_count
        else:
            empty_count = len(empty_boxes)
            total_slots = occupied_count + empty_count

        occupancy_rate = (occupied_count / total_slots * 100.0) if total_slots > 0 else 0.0

        if occupancy_rate < 40.0:
            congestion, rec, color = "Low", "Slots available - proceed", "green"
        elif occupancy_rate <= 75.0:
            congestion, rec, color = "Moderate", "Limited slots available - proceed with caution", "orange"
        else:
            congestion, rec, color = "High", "Parking full - try another area", "red"

        annotated_bgr = self._draw_annotations(image_rgb, occupied_boxes, empty_boxes)

        return ParkingResult(
            annotated_image=annotated_bgr,
            total_slots=total_slots,
            occupied_slots=occupied_count,
            available_slots=empty_count,
            occupancy_rate=occupancy_rate,
            congestion_level=congestion,
            recommendation=rec,
            status_color=color,
            engine_mode=active_engine,
            vacant_boxes=empty_boxes,
            occupied_boxes=occupied_boxes,
        )