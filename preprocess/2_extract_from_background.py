import cv2
import numpy as np

# Read the image, convert to grayscale, and apply Gaussian blur to reduce noise
img   = cv2.imread('../line_crop/line_1.jpg')
gray  = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur  = cv2.GaussianBlur(gray, (5,5), 0)

# Apply adaptive thresholding to create a binary mask (white text on black background)
mask = cv2.adaptiveThreshold(
    blur, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY_INV,
    blockSize=19,
    C=6
)
#Define structuring elements (kernels) for morphological operations
#A horizontally elongated kernel is helpful for reconnecting broken horizontal strokes
kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
kernel_horiz = cv2.getStructuringElement(cv2.MORPH_RECT, (5,1))
kernel_open  = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
#Perform standard closing multiple times to fill in most broken parts
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)
#Use a horizontal kernel to specifically reconnect broken horizontal strokes
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_horiz, iterations=1)
#Apply opening to remove small noise
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
#Slight dilation to fill very narrow gaps further
mask = cv2.dilate(mask, kernel_open, iterations=1)
cv2.imwrite('mask.png', mask)
