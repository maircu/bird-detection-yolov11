import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import cv2
import numpy as np
import time
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

BASE_DIR = Path(__file__).parent.resolve()
VIDEO_DIR = BASE_DIR / "video"
PHOTO_DIR = BASE_DIR / "photo"
OUTPUT_DIR = BASE_DIR / "output"
FONTS_DIR = Path(r"C:\Windows\Fonts")

BIRD_DETECT_MODEL = BASE_DIR / "runs" / "detect" / "bird_detect_v4" / "weights" / "best.pt"
SPECIES_MODEL = BASE_DIR / "runs" / "detect" / "species_occlusion_v2" / "weights" / "best.pt"

SPECIES_NAMES = {
    0: '白顶鹎', 1: '白鹭', 2: '白眉鸭', 3: '斑嘴鸭', 4: '北红尾鸲',
    5: '苍鹭', 6: '赤颈鸭', 7: '大山雀', 8: '反嘴鹬', 9: '黑翅长脚鹬',
    10: '黑水鸡', 11: '鹤鹬', 12: '红胁蓝尾鸲', 13: '黄腰柳莺', 14: '环颈鸻',
    15: '灰头鹀', 16: '矶鹬', 17: '罗纹鸭', 18: '绿翅鸭', 19: '绿头鸭',
    20: '琵嘴鸭', 21: '普通鸬鹚', 22: '山斑鸠', 23: '四声杜鹃', 24: '乌鸫',
    25: '喜鹊', 26: '燕雀', 27: '夜鹭', 28: '银喉长尾山雀', 29: '鸳鸯',
    30: '针尾鸭', 31: '珠颈斑鸠', 32: '棕背伯劳',
}

FONT_CANDIDATES = ["msyh.ttc", "simhei.ttf", "msyhbd.ttc", "simsun.ttc"]


def load_chinese_font(size=24):
    for fname in FONT_CANDIDATES:
        fpath = FONTS_DIR / fname
        if fpath.exists():
            try:
                return ImageFont.truetype(str(fpath), size)
            except Exception:
                continue
    return ImageFont.load_default()


def put_chinese_text(img, text, position, font, fill=(255, 255, 255)):
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    draw.text(position, text, font=font, fill=fill)
    return cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)


def list_videos():
    videos = []
    for ext in ("*.mp4", "*.avi", "*.mkv"):
        videos.extend(sorted(VIDEO_DIR.glob(ext)))
    return videos


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


def cascade_nms(boxes, scores, labels, iou_threshold=0.35):
    if len(boxes) == 0:
        return np.array([]), np.array([]), np.array([])
    order = scores.argsort()[::-1]
    keep = []
    for i in range(len(order)):
        should_keep = True
        for j in keep:
            iou = compute_iou(boxes[order[i]], boxes[order[j]])
            if iou > iou_threshold:
                if labels[order[i]] == "鸟" and labels[order[j]] != "鸟":
                    should_keep = False
                    break
                elif labels[order[i]] != "鸟" and labels[order[j]] == "鸟":
                    keep.remove(j)
                    break
                else:
                    should_keep = False
                    break
        if should_keep:
            keep.append(i)
    idx = order[keep]
    return boxes[idx], scores[idx], labels[idx]


def draw_results(frame, boxes, confs, labels, font=None):
    for i in range(len(boxes)):
        x1, y1, x2, y2 = map(int, boxes[i])
        conf = float(confs[i])
        label = str(labels[i])

        if label == "鸟":
            color = (0, 200, 0)
        else:
            color = (0, 165, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        if font is not None:
            text = f"{label} {conf:.2f}"
            bbox_size = font.getbbox(text)
            tw = bbox_size[2] - bbox_size[0]
            th = bbox_size[3] - bbox_size[1]
            cv2.rectangle(frame, (x1, y1 - th - 12), (x1 + tw + 8, y1), color, -1)
            frame = put_chinese_text(frame, text, (x1 + 4, y1 - th - 8), font, fill=(255, 255, 255))
        else:
            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return frame


class BirdDetector:
    def __init__(self, bird_model_path, species_model_path, motion_detect=False):
        print("Loading bird detection model...")
        self.bird_model = YOLO(str(bird_model_path))
        print("Loading species model...")
        self.species_model = YOLO(str(species_model_path))
        print("All models loaded!")

        # 时序一致性：记录静态框历史
        self._static_history = {}  # key: (rounded_x1,rounded_y1,rounded_x2,rounded_y2) -> count
        self._static_threshold = 5  # 连续N帧在同一位置则视为静态物体(建筑)
        self._static_position_tol = 15  # 位置容差(像素)

        # 运动检测
        self.motion_detect = motion_detect
        self._bg_subtractor = None
        self._motion_mask = None
        self._motion_min_overlap = 0.05  # 检测框与运动区域最小重叠比例
        if motion_detect:
            self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=120, varThreshold=50, detectShadows=True
            )

    def _slice_detect(self, frame, conf=0.05, imgsz=1280, slice_size=640, overlap=0.15):
        """切片推理：将大图切成重叠小块批量检测，合并结果。专门提升小目标检测率。"""
        h, w = frame.shape[:2]
        all_boxes, all_confs = [], []

        # 自适应切片大小（仅当slice_size为默认640时自适应）
        if slice_size >= 640:
            if max(h, w) < 800:
                slice_size = min(h, w, 400)
            elif max(h, w) < 1200:
                slice_size = 480

        # 切片推理分辨率：小切片用1280放大，大切片用640
        tile_imgsz = 1280 if slice_size <= 480 else 640

        step = int(slice_size * (1 - overlap))
        if step < 1:
            step = 1

        # 收集所有切片
        tiles = []
        tile_offsets = []
        for y_start in range(0, h, step):
            for x_start in range(0, w, step):
                x_end = min(x_start + slice_size, w)
                y_end = min(y_start + slice_size, h)
                if (x_end - x_start) < slice_size // 2 or (y_end - y_start) < slice_size // 2:
                    continue
                tiles.append(frame[y_start:y_end, x_start:x_end])
                tile_offsets.append((x_start, y_start))

        if not tiles:
            return np.array([]), np.array([])

        # 批量推理所有切片（一次GPU调用）
        tile_results = self.bird_model.predict(source=tiles, conf=conf, verbose=False, imgsz=tile_imgsz)

        for idx, result in enumerate(tile_results):
            if result.boxes is not None and len(result.boxes) > 0:
                tile_boxes = result.boxes.xyxy.cpu().numpy()
                tile_confs = result.boxes.conf.cpu().numpy()
                x_off, y_off = tile_offsets[idx]
                tile_boxes[:, [0, 2]] += x_off
                tile_boxes[:, [1, 3]] += y_off
                all_boxes.append(tile_boxes)
                all_confs.append(tile_confs)

        if not all_boxes:
            return np.array([]), np.array([])

        boxes = np.concatenate(all_boxes, axis=0)
        confs = np.concatenate(all_confs, axis=0)

        # NMS去重
        if len(boxes) > 0:
            keep = self._nms(boxes, confs, iou_threshold=0.45)
            boxes = boxes[keep]
            confs = confs[keep]

        return boxes, confs

    @staticmethod
    def _nms(boxes, scores, iou_threshold=0.45):
        """标准NMS"""
        if len(boxes) == 0:
            return np.array([], dtype=int)
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep = []
        while len(order) > 0:
            i = order[0]
            keep.append(i)
            if len(order) == 1:
                break
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            mask = iou <= iou_threshold
            order = order[1:][mask]
        return np.array(keep, dtype=int)

    @staticmethod
    def _is_bird_like_shape(bbox, frame_h, frame_w):
        """过滤明显不像鸟的检测框：建筑物通常宽高比极端或面积过大"""
        x1, y1, x2, y2 = bbox
        bw = x2 - x1
        bh = y2 - y1
        if bw <= 0 or bh <= 0:
            return False
        aspect = bw / bh
        # 鸟类宽高比一般在0.3~3.0之间，建筑物常出现极端比例
        if aspect < 0.25 or aspect > 4.0:
            return False
        # 检测框占画面面积过大（>40%）不可能是单只鸟
        area_ratio = (bw * bh) / (frame_h * frame_w)
        if area_ratio > 0.40:
            return False
        return True

    def _update_static_history(self, boxes):
        """更新静态框计数，返回应被过滤的静态框索引集合"""
        current_keys = set()
        for bbox in boxes:
            x1, y1, x2, y2 = bbox
            # 量化位置以容忍微小抖动
            key = (int(x1) // self._static_position_tol,
                   int(y1) // self._static_position_tol,
                   int(x2) // self._static_position_tol,
                   int(y2) // self._static_position_tol)
            current_keys.add(key)

        # 增加当前帧出现的框计数，衰减未出现的框
        new_history = {}
        for key in current_keys:
            new_history[key] = self._static_history.get(key, 0) + 1
        self._static_history = new_history

        # 找出连续出现超过阈值的静态框
        static_keys = {k for k, v in self._static_history.items() if v >= self._static_threshold}
        return static_keys

    def _is_static_detection(self, bbox, static_keys):
        """判断某个框是否属于静态物体"""
        key = (int(bbox[0]) // self._static_position_tol,
               int(bbox[1]) // self._static_position_tol,
               int(bbox[2]) // self._static_position_tol,
               int(bbox[3]) // self._static_position_tol)
        return key in static_keys

    def _update_motion_mask(self, frame):
        """使用MOG2背景减除更新运动掩码，返回前景运动区域掩码"""
        if self._bg_subtractor is None:
            return None
        fg_mask = self._bg_subtractor.apply(frame)
        # 去除阴影（阴影像素值为127）
        _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        # 形态学操作去噪
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.dilate(fg_mask, kernel, iterations=2)
        self._motion_mask = fg_mask
        return fg_mask

    def _has_motion_overlap(self, bbox):
        """判断检测框是否与运动区域有足够重叠，无重叠则说明是静态误检"""
        if self._motion_mask is None:
            return True  # 未启用运动检测时不过滤
        x1, y1, x2, y2 = map(int, bbox)
        h, w = self._motion_mask.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return False
        roi = self._motion_mask[y1:y2, x1:x2]
        box_area = (x2 - x1) * (y2 - y1)
        motion_pixels = cv2.countNonZero(roi)
        overlap_ratio = motion_pixels / box_area if box_area > 0 else 0
        return overlap_ratio >= self._motion_min_overlap

    @staticmethod
    def _estimate_blur(frame):
        """估计帧模糊程度，返回Laplacian方差，值越小越模糊"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var()

    @staticmethod
    def _sharpen_frame(frame, strength=1.5):
        """Unsharp Mask锐化，增强模糊帧的边缘细节"""
        blurred = cv2.GaussianBlur(frame, (0, 0), 3)
        sharpened = cv2.addWeighted(frame, 1 + strength, blurred, -strength, 0)
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    def detect_frame(self, frame, bird_conf=0.25, species_conf=0.45,
                     imgsz=1280, crop_verify=True, slice_detect=False, ultra_fine=False):
        h, w = frame.shape[:2]

        # 模糊检测与锐化增强
        blur_score = self._estimate_blur(frame)
        is_blurry = blur_score < 80  # Laplacian方差<80视为模糊帧
        if is_blurry:
            frame = self._sharpen_frame(frame, strength=2.0)
            # 模糊帧降低检测阈值，提高召回率
            bird_conf_adj = max(0.10, bird_conf * 0.6)
            species_conf_adj = max(0.20, species_conf * 0.7)
        else:
            bird_conf_adj = bird_conf
            species_conf_adj = species_conf

        # 运动检测：更新背景减除掩码
        if self.motion_detect:
            self._update_motion_mask(frame)

        # 鸟检测：先全图推理捕捉大目标，再切片推理捕捉小目标
        all_bird_boxes, all_bird_confs = [], []

        # 第一步：全图推理（对大鸟效果好）
        full_results = self.bird_model.predict(source=frame, conf=bird_conf_adj, verbose=False, imgsz=imgsz)
        if full_results[0].boxes is not None and len(full_results[0].boxes) > 0:
            all_bird_boxes.append(full_results[0].boxes.xyxy.cpu().numpy())
            all_bird_confs.append(full_results[0].boxes.conf.cpu().numpy())

        # 第二步：切片推理（仅slice_detect模式，捕捉小目标）
        if slice_detect:
            if ultra_fine:
                slice_boxes, slice_confs = self._slice_detect(
                    frame, conf=0.01, imgsz=1280,
                    slice_size=320, overlap=0.30,
                )
            else:
                slice_boxes, slice_confs = self._slice_detect(
                    frame, conf=0.05, imgsz=imgsz,
                    slice_size=640, overlap=0.15,
                )
            if len(slice_boxes) > 0:
                all_bird_boxes.append(slice_boxes)
                all_bird_confs.append(slice_confs)

        if not all_bird_boxes:
            self._static_history = {}
            return [], [], []

        bird_boxes = np.concatenate(all_bird_boxes, axis=0)
        bird_confs = np.concatenate(all_bird_confs, axis=0)

        # 合并去重：全图和切片可能重叠检出同一目标
        if len(bird_boxes) > 1:
            keep = self._nms(bird_boxes, bird_confs, iou_threshold=0.45)
            bird_boxes = bird_boxes[keep]
            bird_confs = bird_confs[keep]

        if len(bird_boxes) == 0:
            self._static_history = {}
            return [], [], []

        # 过滤0.5：去除面积=0的垃圾框和极小框
        areas = (bird_boxes[:, 2] - bird_boxes[:, 0]) * (bird_boxes[:, 3] - bird_boxes[:, 1])
        valid_area = areas > 100  # 至少100像素面积
        bird_boxes = bird_boxes[valid_area]
        bird_confs = bird_confs[valid_area]

        if len(bird_boxes) == 0:
            self._static_history = {}
            return [], [], []

        # 过滤0.6：包含关系去重 — 如果小框大部分在大框内，丢弃小框
        if len(bird_boxes) > 1:
            keep_mask = np.ones(len(bird_boxes), dtype=bool)
            for i in range(len(bird_boxes)):
                if not keep_mask[i]:
                    continue
                for j in range(len(bird_boxes)):
                    if i == j or not keep_mask[j]:
                        continue
                    # 计算j框在i框内的比例
                    x1 = max(bird_boxes[i, 0], bird_boxes[j, 0])
                    y1 = max(bird_boxes[i, 1], bird_boxes[j, 1])
                    x2 = min(bird_boxes[i, 2], bird_boxes[j, 2])
                    y2 = min(bird_boxes[i, 3], bird_boxes[j, 3])
                    if x2 > x1 and y2 > y1:
                        overlap = (x2 - x1) * (y2 - y1)
                        area_j = (bird_boxes[j, 2] - bird_boxes[j, 0]) * (bird_boxes[j, 3] - bird_boxes[j, 1])
                        area_i = (bird_boxes[i, 2] - bird_boxes[i, 0]) * (bird_boxes[i, 3] - bird_boxes[i, 1])
                        # 如果j框80%以上在i框内，且i框更大，丢弃j
                        if overlap / area_j > 0.8 and area_i > area_j:
                            keep_mask[j] = False
            bird_boxes = bird_boxes[keep_mask]
            bird_confs = bird_confs[keep_mask]

        if len(bird_boxes) == 0:
            self._static_history = {}
            return [], [], []
        valid_mask = np.array([self._is_bird_like_shape(b, h, w) for b in bird_boxes])
        bird_boxes = bird_boxes[valid_mask]
        bird_confs = bird_confs[valid_mask]

        if len(bird_boxes) == 0:
            return [], [], []

        # 过滤1.5：运动检测过滤（仅保留与运动区域重叠的检测框）
        if self.motion_detect and self._motion_mask is not None:
            motion_mask = np.array([self._has_motion_overlap(b) for b in bird_boxes])
            # 前几帧背景模型未稳定，不过滤
            motion_count = cv2.countNonZero(self._motion_mask)
            frame_area = h * w
            if motion_count > frame_area * 0.01:  # 至少1%画面有运动信息才过滤
                bird_boxes = bird_boxes[motion_mask]
                bird_confs = bird_confs[motion_mask]

        if len(bird_boxes) == 0:
            return [], [], []

        # 过滤2：时序一致性（视频中建筑物位置固定不动）
        static_keys = self._update_static_history(bird_boxes)

        species_results = self.species_model.predict(source=frame, conf=0.05, verbose=False, imgsz=1280)
        sp_boxes, sp_confs, sp_cls = [], [], []
        if species_results[0].boxes is not None and len(species_results[0].boxes) > 0:
            sp_boxes = species_results[0].boxes.xyxy.cpu().numpy()
            sp_confs = species_results[0].boxes.conf.cpu().numpy()
            sp_cls = species_results[0].boxes.cls.cpu().numpy().astype(int)

        final_boxes, final_confs, final_labels = [], [], []

        for i, bbox in enumerate(bird_boxes):
            # 过滤2检查：静态物体（建筑物）直接跳过
            if self._is_static_detection(bbox, static_keys):
                continue

            best_iou = 0
            best_idx = -1
            for j, sp_bbox in enumerate(sp_boxes):
                iou = compute_iou(bbox, sp_bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = j

            # 过滤3：品种模型否决 — 全图品种模型对该框置信度极低则大概率不是鸟
            if best_iou < 0.10 or best_idx < 0:
                # 品种模型没在该位置检出，但鸟检测模型检出了
                # 对大框优先尝试裁剪验证，小框可能是33类之外的鸟
                box_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                if crop_verify and box_area > 1500 and bird_confs[i] >= 0.10:
                    # 大框：裁剪验证识别品种
                    x1, y1, x2, y2 = map(int, bbox)
                    pad = int(max(x2 - x1, y2 - y1) * 0.15)
                    cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
                    cx2, cy2 = min(w, x2 + pad), min(h, y2 + pad)
                    crop = frame[cy1:cy2, cx1:cx2]
                    if crop.shape[0] > 32 and crop.shape[1] > 32:
                        crop_results = self.species_model.predict(source=crop, conf=0.05, verbose=False, imgsz=1280)
                        if crop_results[0].boxes is not None and len(crop_results[0].boxes) > 0:
                            crop_conf = crop_results[0].boxes.conf.cpu().numpy()
                            crop_cls = crop_results[0].boxes.cls.cpu().numpy().astype(int)
                            best_crop = crop_conf.argmax()
                            if crop_conf[best_crop] >= species_conf:
                                final_labels.append(SPECIES_NAMES.get(crop_cls[best_crop], f"species_{crop_cls[best_crop]}"))
                                final_confs.append(float(crop_conf[best_crop]))
                                final_boxes.append(bbox)
                                continue
                            if crop_conf[best_crop] >= 0.15:
                                final_labels.append("鸟")
                                final_confs.append(bird_confs[i])
                                final_boxes.append(bbox)
                                continue
                # 小框或裁剪验证也失败：如果鸟检测置信度高，标注为"鸟"
                min_bird_conf_fallback = 0.05 if ultra_fine else (0.15 if slice_detect else 0.40)
                if bird_confs[i] >= min_bird_conf_fallback:
                    final_labels.append("鸟")
                    final_confs.append(bird_confs[i])
                    final_boxes.append(bbox)
                continue

            if best_iou > 0.20 and best_idx >= 0:
                sp_conf = sp_confs[best_idx]
                sp_cls_id = sp_cls[best_idx]
                # 品种置信度足够高时输出具体品种名，否则标注为"鸟"避免误判
                high_conf_threshold = 0.80  # 品种高置信度阈值
                if sp_conf >= high_conf_threshold:
                    final_labels.append(SPECIES_NAMES.get(sp_cls_id, f"species_{sp_cls_id}"))
                    final_confs.append(sp_conf)
                    final_boxes.append(bbox)
                    continue
                elif sp_conf >= species_conf:
                    # 中等置信度：标注为"鸟"而非具体品种，避免误判
                    final_labels.append("鸟")
                    final_confs.append(sp_conf)
                    final_boxes.append(bbox)
                    continue

            min_bird_conf = 0.05 if ultra_fine else (0.15 if slice_detect else 0.40)

            if crop_verify and bird_confs[i] >= min_bird_conf:
                x1, y1, x2, y2 = map(int, bbox)
                pad = int(max(x2 - x1, y2 - y1) * 0.15)
                cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
                cx2, cy2 = min(w, x2 + pad), min(h, y2 + pad)
                crop = frame[cy1:cy2, cx1:cx2]
                if crop.shape[0] > 32 and crop.shape[1] > 32:
                    crop_results = self.species_model.predict(source=crop, conf=0.10, verbose=False, imgsz=1280)
                    if crop_results[0].boxes is not None and len(crop_results[0].boxes) > 0:
                        crop_conf = crop_results[0].boxes.conf.cpu().numpy()
                        crop_cls = crop_results[0].boxes.cls.cpu().numpy().astype(int)
                        best_crop = crop_conf.argmax()
                        if crop_conf[best_crop] >= 0.80:
                            # 高置信度：输出具体品种名
                            final_labels.append(SPECIES_NAMES.get(crop_cls[best_crop], f"species_{crop_cls[best_crop]}"))
                            final_confs.append(float(crop_conf[best_crop]))
                            final_boxes.append(bbox)
                            continue
                        # 裁剪验证品种置信度不够高，标注为"鸟"避免误判
                        if crop_conf[best_crop] >= 0.15:
                            final_labels.append("鸟")
                            final_confs.append(bird_confs[i])
                            final_boxes.append(bbox)
                            continue

            if bird_confs[i] >= min_bird_conf:
                final_labels.append("鸟")
                final_confs.append(bird_confs[i])
                final_boxes.append(bbox)

        if len(final_boxes) == 0:
            return [], [], []

        final_boxes = np.array(final_boxes)
        final_confs = np.array(final_confs)
        final_labels = np.array(final_labels)

        final_boxes, final_confs, final_labels = cascade_nms(
            final_boxes, final_confs, final_labels, iou_threshold=0.35
        )

        return final_boxes, final_confs, final_labels


def list_photos():
    photos = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
        photos.extend(sorted(PHOTO_DIR.glob(ext)))
    return photos


def detect_photo(detector, photo_path, font, output_dir, bird_conf=0.25, species_conf=0.45, crop_verify=True, slice_detect=False, ultra_fine=False):
    img = cv2.imread(str(photo_path))
    if img is None:
        print(f"  Cannot read: {photo_path.name}")
        return 0, 0

    boxes, confs, labels = detector.detect_frame(
        img, bird_conf=bird_conf, species_conf=species_conf,
        imgsz=1280, crop_verify=crop_verify, slice_detect=slice_detect,
        ultra_fine=ultra_fine,
    )

    annotated = draw_results(img.copy(), boxes, confs, labels, font=font)

    n_birds = sum(1 for l in labels if l == "鸟")
    n_species = sum(1 for l in labels if l != "鸟")
    summary = f"total:{len(labels)} bird:{n_birds} sp:{n_species}"
    cv2.putText(annotated, summary, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    output_name = f"det_{photo_path.stem}{photo_path.suffix}"
    output_path = output_dir / output_name
    cv2.imwrite(str(output_path), annotated)

    detail = ", ".join([f"{labels[i]}({confs[i]:.2f})" for i in range(len(labels))])
    print(f"  {photo_path.name}: {len(labels)} detected [{detail}]")
    return n_birds, n_species


def main():
    print("=" * 60)
    print("  Bird Detection Pipeline v3")
    print("=" * 60)

    if not BIRD_DETECT_MODEL.exists():
        print(f"Error: Bird model not found: {BIRD_DETECT_MODEL}")
        return
    if not SPECIES_MODEL.exists():
        print(f"Error: Species model not found: {SPECIES_MODEL}")
        return

    print(f"\n  Detection mode:")
    print(f"    1. Video detection")
    print(f"    2. Photo detection (photo/ folder)")
    mode = input("  Select (default=1): ").strip()
    if mode == "2":
        photo_mode(detector=None)
        return

    video_mode()


def photo_mode(detector=None):
    PHOTOS_OUTPUT = OUTPUT_DIR / "photos"
    PHOTOS_OUTPUT.mkdir(parents=True, exist_ok=True)

    photos = list_photos()
    if not photos:
        print(f"\n  No images found in: {PHOTO_DIR}")
        print(f"  Please put jpg/png/bmp/webp images in the 'photo' folder")
        return

    print(f"\n  Found {len(photos)} images in photo/")

    crop_choice = input("  Enable crop verify? (y/n, default=y): ").strip().lower()
    crop_verify = crop_choice != 'n'

    species_conf_input = input("  Species confidence (default=0.45): ").strip()
    species_conf = float(species_conf_input) if species_conf_input else 0.45

    if detector is None:
        detector = BirdDetector(BIRD_DETECT_MODEL, SPECIES_MODEL)

    font = load_chinese_font(size=20)

    print(f"\n  Processing {len(photos)} images (ultra-fine slice, crop={'on' if crop_verify else 'off'}, conf={species_conf})...")
    print()

    total_birds = 0
    total_species = 0

    for i, photo in enumerate(photos, 1):
        print(f"  [{i}/{len(photos)}] ", end="")
        n_birds, n_species = detect_photo(detector, photo, font, PHOTOS_OUTPUT,
                                          bird_conf=0.25, species_conf=species_conf,
                                          crop_verify=crop_verify, slice_detect=True,
                                          ultra_fine=True)
        total_birds += n_birds
        total_species += n_species

    print(f"\n  Done! Processed {len(photos)} images")
    print(f"  Birds: {total_birds}, Species identified: {total_species}")
    print(f"  Results saved to: {PHOTOS_OUTPUT}")


def video_mode():
    videos = list_videos()
    if not videos:
        print(f"No videos found in: {VIDEO_DIR}")
        return

    print(f"\n  Available videos:")
    for i, v in enumerate(videos, 1):
        size_mb = v.stat().st_size / (1024 * 1024)
        print(f"    {i}. {v.name}  ({size_mb:.1f} MB)")

    print()
    choice = input("  Select video number (or 'q' to quit): ").strip()
    if choice.lower() == 'q':
        return
    try:
        idx = int(choice) - 1
        if idx < 0 or idx >= len(videos):
            print(f"Invalid choice: {choice}")
            return
    except ValueError:
        print(f"Invalid input: {choice}")
        return
    selected = videos[idx]

    print(f"\n  Detection mode:")
    print(f"    1. Fast (YOLO only, ~15fps)")
    print(f"    2. + Crop verify (YOLO+crop, ~8fps, better for occlusion)")
    print(f"    3. + Slice (YOLO+slice, ~3fps, best for small/distant birds)")
    mode_choice = input("  Select (default=1): ").strip()
    crop_verify = mode_choice in ("2", "3")
    slice_detect = mode_choice == "3"

    motion_choice = input("  Enable motion detection? (y/n, default=n): ").strip().lower()
    motion_detect = motion_choice == 'y'

    print(f"\n  Frame skip (process every Nth frame):")
    print(f"    1. Every frame (slowest)")
    print(f"    2. Every 2nd frame")
    print(f"    3. Every 3rd frame (recommended)")
    skip_choice = input("  Select (default=3): ").strip()
    frame_skip = {"1": 1, "2": 2}.get(skip_choice, 3)

    bird_conf_input = input("  Bird detection confidence (default=0.25): ").strip()
    bird_conf = float(bird_conf_input) if bird_conf_input else 0.25
    species_conf_input = input("  Species identification confidence (default=0.45): ").strip()
    species_conf = float(species_conf_input) if species_conf_input else 0.45

    mode_tag = "slice" if slice_detect else ("crop" if crop_verify else "fast")
    if motion_detect:
        mode_tag += "_motion"
    output_name = f"v3_{mode_tag}_skip{frame_skip}_{selected.stem}.mp4"
    output_path = OUTPUT_DIR / output_name

    detector = BirdDetector(BIRD_DETECT_MODEL, SPECIES_MODEL, motion_detect=motion_detect)

    cap = cv2.VideoCapture(str(selected))
    if not cap.isOpened():
        print(f"Error: Cannot open video {selected}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    font = load_chinese_font(size=20)

    print(f"\n  Video: {selected.name}")
    print(f"  Resolution: {width}x{height}, FPS: {fps}, Frames: {total_frames}")
    print(f"  Mode: {mode_tag}, Frame skip: {frame_skip}")
    print(f"  Output: {output_path}")
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    frame_idx = 0
    total_birds = 0
    total_species = 0
    last_boxes, last_confs, last_labels = [], [], []
    t_start = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_skip == 0:
            last_boxes, last_confs, last_labels = detector.detect_frame(
                frame, bird_conf=bird_conf, species_conf=species_conf,
                imgsz=1280, crop_verify=crop_verify, slice_detect=slice_detect,
            )

        annotated = draw_results(frame.copy(), last_boxes, last_confs, last_labels, font=font)

        n_birds = sum(1 for l in last_labels if l == "鸟")
        n_species = sum(1 for l in last_labels if l != "鸟")
        total_birds += n_birds
        total_species += n_species

        elapsed = time.time() - t_start
        cur_fps = (frame_idx + 1) / elapsed if elapsed > 0 else 0
        summary = f"fps:{cur_fps:.1f} total:{len(last_labels)} bird:{n_birds} sp:{n_species}"
        cv2.putText(annotated, summary, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        writer.write(annotated)

        frame_idx += 1
        if frame_idx % 50 == 0:
            pct = frame_idx / total_frames * 100 if total_frames > 0 else 0
            print(f"\r  Progress: {frame_idx}/{total_frames} ({pct:.1f}%), "
                  f"fps:{cur_fps:.1f}, birds:{total_birds} species:{total_species}",
                  end="", flush=True)

    cap.release()
    writer.release()

    elapsed = time.time() - t_start
    print(f"\n  Done! {frame_idx} frames in {elapsed:.1f}s ({frame_idx/elapsed:.1f} fps)")
    print(f"  Birds: {total_birds}, Species identified: {total_species}")
    print(f"  Saved to: {output_path}")


if __name__ == '__main__':
    main()
