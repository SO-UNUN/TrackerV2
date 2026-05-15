"""
modules/marker_tracker.py
Track marker position in video frames using:
1. Color-based tracking (HSV thresholding) — fast, simple
2. ROI/object tracking (CSRT) — robust to color changes

Author: University of Phayao | Physics Department
"""

import cv2
import numpy as np
import pandas as pd
from typing import Iterator, Tuple, Optional, Dict, List
from dataclasses import dataclass


@dataclass
class ColorRange:
    """ช่วงสีใน HSV ที่ใช้ track marker"""
    h_low: int = 0      # 0-179 (OpenCV HSV scale)
    h_high: int = 179
    s_low: int = 100    # 0-255
    s_high: int = 255
    v_low: int = 100    # 0-255
    v_high: int = 255
    label: str = "marker"

    def to_hsv_bounds(self):
        """คืน lower, upper bounds สำหรับ cv2.inRange()"""
        lower = np.array([self.h_low, self.s_low, self.v_low])
        upper = np.array([self.h_high, self.s_high, self.v_high])
        return lower, upper


# Preset colors สำหรับสีมาร์กเกอร์ที่นิยม
COLOR_PRESETS = {
    "Red (red marker)":      ColorRange(0, 10, 120, 255, 100, 255, "red"),
    "Red high (alt)":        ColorRange(170, 179, 120, 255, 100, 255, "red2"),
    "Orange":                ColorRange(10, 25, 120, 255, 100, 255, "orange"),
    "Yellow":                ColorRange(25, 40, 100, 255, 100, 255, "yellow"),
    "Green":                 ColorRange(40, 80, 100, 255, 50, 255, "green"),
    "Cyan":                  ColorRange(80, 100, 100, 255, 100, 255, "cyan"),
    "Blue":                  ColorRange(100, 130, 100, 255, 50, 255, "blue"),
    "Purple/Magenta":        ColorRange(130, 165, 100, 255, 100, 255, "purple"),
}


def detect_marker_color_based(
    frame_rgb: np.ndarray,
    color_range: ColorRange,
    min_area: int = 50,
) -> Optional[Tuple[float, float, float]]:
    """หาตำแหน่ง marker จากสี
    
    Args:
        frame_rgb: เฟรม RGB (numpy uint8)
        color_range: ช่วงสี HSV ที่ใช้
        min_area: พื้นที่ขั้นต่ำ (พิกเซล) เพื่อเป็น marker (กรอง noise)

    Returns:
        (x, y, area) หรือ None ถ้าไม่พบ
        x, y = centroid ของพื้นที่ที่ตรงสี (พิกเซล)
        area = ขนาดของพื้นที่ที่ track ได้
    """
    # แปลง RGB → BGR → HSV (OpenCV convention)
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    lower, upper = color_range.to_hsv_bounds()
    mask = cv2.inRange(hsv, lower, upper)

    # ลด noise ด้วย morphology
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # หา contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # เลือก contour ที่ใหญ่ที่สุด
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    if area < min_area:
        return None

    # คำนวณ centroid
    M = cv2.moments(largest)
    if M["m00"] == 0:
        return None
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]
    return (cx, cy, area)


def get_mask_preview(
    frame_rgb: np.ndarray,
    color_range: ColorRange,
    overlay_alpha: float = 0.5,
) -> np.ndarray:
    """สร้างภาพ preview แสดงพื้นที่ที่จะถูก track (สำหรับให้นิสิตปรับค่า)
    
    Returns:
        ภาพ RGB ที่ highlight พื้นที่ที่ตรงสี
    """
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    lower, upper = color_range.to_hsv_bounds()
    mask = cv2.inRange(hsv, lower, upper)

    # Morphology cleanup
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # สร้าง overlay: พื้นที่ที่ตรงสี = สีเขียวสด
    overlay = frame_rgb.copy()
    green_mask = np.zeros_like(frame_rgb)
    green_mask[:] = (50, 255, 50)
    overlay = np.where(mask[..., None] > 0, green_mask, overlay)

    # Blend
    result = cv2.addWeighted(frame_rgb, 1 - overlay_alpha, overlay, overlay_alpha, 0)

    # วาด centroid + bounding box ถ้าหาเจอ
    pos = detect_marker_color_based(frame_rgb, color_range)
    if pos is not None:
        cx, cy, _ = pos
        cv2.circle(result, (int(cx), int(cy)), 8, (255, 255, 0), 2)
        cv2.circle(result, (int(cx), int(cy)), 2, (255, 0, 0), -1)

    return result


def auto_detect_color_from_roi(
    frame_rgb: np.ndarray,
    x: int, y: int, w: int, h: int,
    h_tolerance: int = 15,
    s_tolerance: int = 60,
    v_tolerance: int = 60,
) -> ColorRange:
    """ดูดสีจาก ROI ที่ผู้ใช้เลือก แล้วคำนวณ HSV range ให้อัตโนมัติ
    
    ใช้เมื่อผู้ใช้ลากกล่องรอบ marker → โปรแกรมเดาช่วงสีให้
    """
    h_img, w_img = frame_rgb.shape[:2]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(w_img, x + w), min(h_img, y + h)
    patch = frame_rgb[y1:y2, x1:x2]
    if patch.size == 0:
        return ColorRange()

    # แปลง patch เป็น HSV
    patch_bgr = cv2.cvtColor(patch, cv2.COLOR_RGB2BGR)
    patch_hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)

    # ค่าเฉลี่ย HSV ใน patch
    mean_h = int(np.median(patch_hsv[..., 0]))
    mean_s = int(np.median(patch_hsv[..., 1]))
    mean_v = int(np.median(patch_hsv[..., 2]))

    # สร้างช่วง ± tolerance
    return ColorRange(
        h_low=max(0, mean_h - h_tolerance),
        h_high=min(179, mean_h + h_tolerance),
        s_low=max(0, mean_s - s_tolerance),
        s_high=min(255, mean_s + s_tolerance),
        v_low=max(0, mean_v - v_tolerance),
        v_high=min(255, mean_v + v_tolerance),
        label="auto_picked",
    )


# ============================================================
#  TRACKING ENTRY POINT — Color-based
# ============================================================
def track_video_color(
    frame_iterator: Iterator[Tuple[float, np.ndarray]],
    color_range: ColorRange,
    min_area: int = 50,
    progress_callback=None,
    total_estimate: int = 0,
) -> pd.DataFrame:
    """ติดตาม marker ผ่านวิดีโอด้วย color-based tracking

    Returns:
        DataFrame columns: time_sec, x_px, y_px, area, found
    """
    rows = []
    count = 0
    last_pos = None  # สำหรับเติมเมื่อ tracking หลุด

    for timestamp, frame in frame_iterator:
        pos = detect_marker_color_based(frame, color_range, min_area=min_area)
        if pos is not None:
            cx, cy, area = pos
            rows.append({
                "time_sec": timestamp,
                "x_px": cx,
                "y_px": cy,
                "area": area,
                "found": True,
            })
            last_pos = (cx, cy)
        else:
            # ถ้า tracking หลุด ใช้ค่าก่อนหน้า (forward-fill)
            if last_pos is not None:
                rows.append({
                    "time_sec": timestamp,
                    "x_px": last_pos[0],
                    "y_px": last_pos[1],
                    "area": 0,
                    "found": False,
                })
            else:
                rows.append({
                    "time_sec": timestamp,
                    "x_px": np.nan,
                    "y_px": np.nan,
                    "area": 0,
                    "found": False,
                })

        count += 1
        if progress_callback:
            progress_callback(count, total_estimate)

    return pd.DataFrame(rows)


# ============================================================
#  TRACKING ENTRY POINT — CSRT (Object tracker)
# ============================================================
def track_video_csrt(
    frame_iterator: Iterator[Tuple[float, np.ndarray]],
    init_bbox: Tuple[int, int, int, int],  # (x, y, w, h)
    progress_callback=None,
    total_estimate: int = 0,
) -> pd.DataFrame:
    """ติดตาม marker ด้วย CSRT tracker (OpenCV contrib)
    
    ใช้เมื่อ marker ไม่มีสีโดดเด่น — robust กว่า แต่ช้ากว่า color tracking
    Need: opencv-contrib-python-headless
    
    Args:
        init_bbox: bounding box ของ marker ใน frame แรก (x, y, w, h)
    """
    try:
        tracker = cv2.TrackerCSRT_create()
    except AttributeError:
        # OpenCV รุ่นใหม่ย้ายไปอยู่ใน legacy
        tracker = cv2.legacy.TrackerCSRT_create()

    rows = []
    count = 0
    initialized = False

    for timestamp, frame in frame_iterator:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        if not initialized:
            tracker.init(frame_bgr, init_bbox)
            initialized = True
            x, y, w, h = init_bbox
            cx, cy = x + w/2, y + h/2
            rows.append({
                "time_sec": timestamp,
                "x_px": cx, "y_px": cy,
                "bbox_w": w, "bbox_h": h,
                "found": True,
            })
        else:
            success, bbox = tracker.update(frame_bgr)
            if success:
                x, y, w, h = bbox
                cx, cy = x + w/2, y + h/2
                rows.append({
                    "time_sec": timestamp,
                    "x_px": cx, "y_px": cy,
                    "bbox_w": w, "bbox_h": h,
                    "found": True,
                })
            else:
                rows.append({
                    "time_sec": timestamp,
                    "x_px": np.nan, "y_px": np.nan,
                    "bbox_w": 0, "bbox_h": 0,
                    "found": False,
                })

        count += 1
        if progress_callback:
            progress_callback(count, total_estimate)

    return pd.DataFrame(rows)


# ============================================================
#  PIXEL → REAL UNITS Conversion
# ============================================================
def pixels_to_real_units(
    df: pd.DataFrame,
    pixels_per_cm: float,
    origin_x_px: float = None,
    origin_y_px: float = None,
) -> pd.DataFrame:
    """แปลงตำแหน่งพิกเซลเป็นหน่วยจริง (เซนติเมตร)
    
    Args:
        df: DataFrame จาก track_video_*()
        pixels_per_cm: scale factor (จาก reference object เช่น ไม้บรรทัด)
        origin_x_px, origin_y_px: จุดศูนย์ (default = mean position)

    Returns:
        DataFrame เดิม + columns x_cm, y_cm
    """
    df = df.copy()
    if origin_x_px is None:
        origin_x_px = df["x_px"].mean()
    if origin_y_px is None:
        origin_y_px = df["y_px"].mean()

    df["x_cm"] = (df["x_px"] - origin_x_px) / pixels_per_cm
    # Y axis: invert เพราะ image y เพิ่มลง แต่ฟิสิกส์เพิ่มขึ้น
    df["y_cm"] = -(df["y_px"] - origin_y_px) / pixels_per_cm

    return df
