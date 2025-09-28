import sys, os
import cv2
import numpy as np
from paddleocr import PaddleOCR

def detect_lines(image_path: str, output_dir: str):
    """
    Use PaddleOCR to detect text lines in an image and output:
    """
    #Initialize OCR with line detection only (no recognition)
    ocr = PaddleOCR(
        use_angle_cls=False,
        lang='en',
        det=True, rec=False, cls=False,
        use_gpu=True
    )

    #Run OCR detection (returns a list of polygons per image)
    raw = ocr.ocr(image_path, cls=False, rec=False)
    line_polys = raw[0]  #Each poly is a 4×2 array of points

    #Load the original image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    #Prepare output directory and visualization image
    os.makedirs(output_dir, exist_ok=True)
    vis = img.copy()
    saved = []

    #Iterate over detected lines, convert polygon to bounding box, crop and save each line
    for i, poly in enumerate(line_polys):
        pts = np.array(poly).reshape(-1,2).astype(int)
        x,y,w,h = cv2.boundingRect(pts)
        cv2.rectangle(vis, (x,y), (x+w,y+h), (0,255,0), 2)
        crop = img[y:y+h, x:x+w]
        fn = os.path.join(output_dir, f"line_{i}.jpg")
        cv2.imwrite(fn, crop)
        saved.append(fn)

    cv2.imwrite(os.path.join(output_dir, "lines_bboxes.jpg"), vis)
    print(f"[+] Detected {len(saved)} lines, crops saved to {output_dir}")
    for fn in saved:
        print("   ", fn)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python line_localization.py <image_path> <output_dir>")
        sys.exit(1)
    detect_lines(sys.argv[1], sys.argv[2])
