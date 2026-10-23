# -*- coding: utf-8 -*-
"""
从摄像头实时串流中识别指定 LAB 颜色范围内的多边形物体（允许非凸）。

阈值 (21, 64, 17, 95, 22, 88) 按 (L_min, L_max, a_min, a_max, b_min, b_max) 解析。
OpenCV 8U LAB 中：
    L: 0~100   -> 0~255
    a: -128~127 -> 0~255 (加 128)
    b: -128~127 -> 0~255 (加 128)
若阈值已是 OpenCV 8U 数值，把 THRESH_IS_OPENCV_8U 设为 True。

按键：
    q / Esc : 退出
    s       : 保存当前帧识别结果
    m       : 切换显示二值掩码窗口
"""
import cv2
import numpy as np
import time

# ==================== 参数 ====================
CAM_INDEX   = 0          # 摄像头索引，外接摄像头可试 1 / 2
FRAME_W     = 1280       # 请求分辨率（不支持会自动回退）
FRAME_H     = 720
FPS_TARGET  = 30

LAB_THRESH = (21, 64, 17, 95, 22, 88)   # L_min,L_max, a_min,a_max, b_min,b_max
THRESH_IS_OPENCV_8U = False

MIN_AREA      = 300      # 最小面积，过滤噪点
EPS_RATIO     = 0.02     # 多边形逼近精度（相对轮廓周长）
MIN_VERTICES  = 3
MAX_VERTICES  = 30
MIN_SOLIDITY  = 0.0      # 实心度下限；0 表示不过滤
MORPH_KSIZE   = 5
SHOW_HULL_REF = True     # 是否用浅黄色虚线画出凸包参考
# =============================================


def build_lab_range(thresh, is_8u=False):
    """把 LAB 阈值转成 cv2.inRange 需要的 lower/upper (uint8)"""
    lmin, lmax, amin, amax, bmin, bmax = thresh
    if is_8u:
        lower = np.array([lmin, amin, bmin], dtype=np.uint8)
        upper = np.array([lmax, amax, bmax], dtype=np.uint8)
    else:
        lower = np.array([
            np.clip(round(lmin * 255.0 / 100.0), 0, 255),
            np.clip(amin + 128, 0, 255),
            np.clip(bmin + 128, 0, 255),
        ], dtype=np.uint8)
        upper = np.array([
            np.clip(round(lmax * 255.0 / 100.0), 0, 255),
            np.clip(amax + 128, 0, 255),
            np.clip(bmax + 128, 0, 255),
        ], dtype=np.uint8)
    return lower, upper


def find_polygons(frame_bgr, lower, upper):
    """
    在单帧上做阈值分割 + 多边形识别。
    返回：
        approx_list : [Nx1x2 ndarray, ...] 多边形逼近结果（可能非凸）
        hull_list   : [Nx1x2 ndarray, ...] 对应凸包
        meta_list   : [(area, n_vertices, solidity), ...]
        mask        : 二值掩码
    """
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB)
    mask = cv2.inRange(lab, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MORPH_KSIZE, MORPH_KSIZE))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    approx_list, hull_list, meta_list = [], [], []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA:
            continue

        peri = cv2.arcLength(cnt, True)
        if peri <= 0:
            continue

        approx = cv2.approxPolyDP(cnt, EPS_RATIO * peri, True)
        n_vertices = len(approx)
        if n_vertices < MIN_VERTICES or n_vertices > MAX_VERTICES:
            continue

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = area / hull_area if hull_area > 0 else 0.0
        if solidity < MIN_SOLIDITY:
            continue

        approx_list.append(approx)
        hull_list.append(hull)
        meta_list.append((area, n_vertices, solidity))

    return approx_list, hull_list, meta_list, mask


def draw_dashed_polyline(img, pts, color, thickness=1, dash=8):
    """画虚线多边形（用于凸包参考）"""
    pts = pts.reshape(-1, 2).astype(np.int32)
    n = len(pts)
    for i in range(n):
        p1 = pts[i].astype(float)
        p2 = pts[(i + 1) % n].astype(float)
        dist = np.linalg.norm(p2 - p1)
        if dist < 1e-6:
            continue
        steps = max(int(dist // dash), 1)
        for k in range(steps):
            t0 = k / steps
            t1 = min((k + 0.5) / steps, 1.0)
            q1 = tuple((p1 + (p2 - p1) * t0).astype(int))
            q2 = tuple((p1 + (p2 - p1) * t1).astype(int))
            cv2.line(img, q1, q2, color, thickness, cv2.LINE_AA)


def draw_results(frame, approx_list, hull_list, meta_list):
    """把识别结果画到帧上"""
    for idx, (approx, hull, meta) in enumerate(zip(approx_list, hull_list, meta_list)):
        area, n_vertex, solidity = meta

        # 绿色实线：多边形逼近结果（可能非凸）
        cv2.polylines(frame, [approx], True, (0, 255, 0), 2)

        # 浅黄色虚线：凸包参考
        if SHOW_HULL_REF:
            draw_dashed_polyline(frame, hull, (0, 255, 255), 1, dash=6)

        # 红色顶点
        for p in approx.reshape(-1, 2):
            cv2.circle(frame, tuple(p), 3, (0, 0, 255), -1)

        # 标注信息
        M = cv2.moments(approx)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            text = f"#{idx} v={n_vertex} A={int(area)} S={solidity:.2f}"
            cv2.putText(frame, text, (cx - 80, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, text, (cx - 80, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


def main():
    lower, upper = build_lab_range(LAB_THRESH, THRESH_IS_OPENCV_8U)
    print("inRange lower =", lower.tolist(), " upper =", upper.tolist())

    # ---------- 打开摄像头 ----------
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开摄像头 index={CAM_INDEX}")

    # 尝试设置分辨率（不支持会被驱动回退，无需报错）
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)
    cap.set(cv2.CAP_PROP_FPS,          FPS_TARGET)

    real_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    real_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"摄像头已打开: {real_w}x{real_h}")

    show_mask = True
    window_name = "Camera Stream - Polygon Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    prev_t = time.time()
    fps_smooth = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("读取帧失败，退出")
                break

            # ---------- 识别 ----------
            approx_list, hull_list, meta_list, mask = find_polygons(frame, lower, upper)

            # ---------- 绘制 ----------
            vis = frame.copy()
            draw_results(vis, approx_list, hull_list, meta_list)

            # ---------- FPS ----------
            now = time.time()
            dt = now - prev_t
            prev_t = now
            if dt > 1e-6:
                fps_inst = 1.0 / dt
                fps_smooth = 0.9 * fps_smooth + 0.1 * fps_inst if fps_smooth > 0 else fps_inst
            cv2.putText(vis, f"FPS: {fps_smooth:5.1f}  Objs: {len(approx_list)}",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0), 2, cv2.LINE_AA)

            cv2.imshow(window_name, vis)
            if show_mask:
                cv2.imshow("Mask", mask)

            # ---------- 按键 ----------
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:      # q 或 Esc
                break
            elif key == ord('s'):
                ts = time.strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(f"capture_{ts}.png", vis)
                print(f"已保存 capture_{ts}.png")
            elif key == ord('m'):
                show_mask = not show_mask
                if not show_mask:
                    cv2.destroyWindow("Mask")
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()