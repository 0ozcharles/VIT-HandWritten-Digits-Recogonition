import cv2
import numpy as np
import os

# ---- 路径准备 ------------------------------------------------
root_out = "debug_masks"          # 统一保存目录
os.makedirs(root_out, exist_ok=True)

# ---- 1 读图、灰度、去噪 --------------------------------------
img  = cv2.imread('../line_crop/line_1.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5,5), 0)
cv2.imwrite(f"{root_out}/01_blur.png", blur)

# ---- 2 自适应阈值 -------------------------------------------
mask = cv2.adaptiveThreshold(
    blur, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    blockSize=19, C=6
)
cv2.imwrite(f"{root_out}/02_adapt_thresh.png", mask)

# ---- 3 形态学闭运算(3×3, iter=2) -----------------------------
kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)
cv2.imwrite(f"{root_out}/03_close_3x3.png", mask)

# ---- 4 水平闭运算(5×1) --------------------------------------
kernel_horiz = cv2.getStructuringElement(cv2.MORPH_RECT, (5,1))
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_horiz, iterations=1)
cv2.imwrite(f"{root_out}/04_close_5x1.png", mask)

# ---- 5 开运算(2×2) ------------------------------------------
kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
cv2.imwrite(f"{root_out}/05_open_2x2.png", mask)

# ---- 6 轻度膨胀(2×2) ----------------------------------------
mask = cv2.dilate(mask, kernel_open, iterations=1)
cv2.imwrite(f"{root_out}/06_dilate_2x2.png", mask)

# ---- 7 最终结果 ---------------------------------------------
cv2.imwrite(f"{root_out}/07_final_mask.png", mask)
print("[+] 所有中间结果已保存至", root_out)
