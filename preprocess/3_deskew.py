import cv2
import numpy as np

def deskew_by_pca(mask):
    """
    Estimate the principal direction of the foreground (white pixels) using PCA,
    then rotate the image to align it horizontally.
    Returns (deskewed_mask, angle_deg).
    """
    # Extract coordinates of all foreground pixels
    ys, xs = np.where(mask > 0)
    coords = np.stack([xs, ys], axis=1).astype(np.float32)
    # Perform PCA on the coordinates
    mean, eigenvectors = cv2.PCACompute(coords, mean=None)
    # Get the principal direction vector
    vx, vy = eigenvectors[0]
    angle = np.arctan2(vy, vx)  # 弧度
    angle_deg = (angle * 180.0 / np.pi)
    # Rotate the image to deskew it (note the sign of the angle)
    h, w = mask.shape
    M = cv2.getRotationMatrix2D((w/2, h/2), angle_deg, 1.0)
    deskew = cv2.warpAffine(mask, M, (w, h),
                            flags=cv2.INTER_NEAREST,
                            borderValue=0)
    return deskew, angle_deg




if __name__ == "__main__":
    mask = cv2.imread("mask.png", 0)
    deskew_pca, ang1 = deskew_by_pca(mask)
    print(f"Estimated skew angle by PCA = {ang1:.2f}°")
    cv2.imwrite("mask_deskew_pca.png", deskew_pca)


