import os
import argparse
import torch
import cv2  # Add OpenCV import
import numpy as np
from torchvision import transforms
from PIL import Image
import json  # Add JSON import

from model import VisionTransformer  #


def parse_args():
    p = argparse.ArgumentParser("Batch predict pre-cropped MNIST chars with ViT and draw on image")
    p.add_argument('--chars_dir', default='chars_28',
                   help="Folder of 28x28 character images to recognize")  #
    p.add_argument('--model_pt', required=True,
                   help="Path to trained ViT weights file (.pt)")  #

    # New arguments for drawing predictions
    p.add_argument('--base_image_path', required=True,
                   help="Path to the base image on which to draw predictions (e.g., the deskewed mask like mask_deskew_pca.png)")
    p.add_argument('--boxes_info_path', default=os.path.join('chars_28', 'boxes_info.json'),
                   help="Path to the JSON file containing bounding box information (default: chars_28/boxes_info.json)")
    p.add_argument('--output_image_path', default='predictions_on_image.png',
                   help="Path to save the image with predictions drawn on it")

    p.add_argument('--device', default='cpu', help="Running device: cpu or cuda")  #
    p.add_argument('--img_size', type=int, default=28, help="Input model image size (default 28)")  #
    p.add_argument('--patch_size', type=int, default=4, help="Patch size for ViT (consistent with training)")  #
    p.add_argument('--n_channels', type=int, default=1, help="Number of input channels (consistent with training)")  #
    p.add_argument('--embed_dim', type=int, default=64, help="Embedding dimension (consistent with training)")  #
    p.add_argument('--n_attention_heads', type=int, default=4,
                   help="Number of attention heads (consistent with training)")  #
    p.add_argument('--forward_mul', type=int, default=2, help="Forward multiplier (consistent with training)")  #
    p.add_argument('--n_layers', type=int, default=6, help="Number of transformer layers (consistent with training)")  #
    p.add_argument('--n_classes', type=int, default=10, help="Number of output classes (consistent with training)")  #
    return p.parse_args()


def feature_based_repair(img_tensor, pred):
    if pred != 7:  #
        return pred  #

    img = img_tensor.squeeze().cpu().numpy()  #
    img = ((img + 1) * 127.5).astype(np.uint8)  #
    _, fg = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  #
    h_char, w_char = fg.shape  # # Use different var names to avoid conflict with box w,h

    contours, hierarchy = cv2.findContours(fg, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)  #
    if hierarchy is None:  #
        return pred  #

    hierarchy = hierarchy[0]  #
    hole_centers = []  #

    for idx, hinfo in enumerate(hierarchy):  #
        parent = hinfo[3]  #
        if parent != -1 and hierarchy[parent][3] == -1:  #
            area = cv2.contourArea(contours[idx])  #
            if area < 6:  #
                continue  #
            M = cv2.moments(contours[idx])  #
            if M["m00"] == 0:  #
                continue  #
            cy = M["m01"] / M["m00"]  #
            hole_centers.append(cy)  #

    if len(hole_centers) >= 1:  #
        if np.min(hole_centers) < h_char * 0.6:  #
            return 9  #
    return pred  #


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')  #

    #Build model and load weights
    model = VisionTransformer(args).to(device)  #
    ckpt = torch.load(args.model_pt, map_location=device)  #
    state = ckpt.get('model_state_dict', ckpt)  #
    model.load_state_dict(state)  #
    model.eval()  #

    #Preprocessing
    transform = transforms.Compose([  #
        transforms.Resize((args.img_size, args.img_size)),  #
        transforms.ToTensor(),  #
        transforms.Normalize([0.5], [0.5]),  #
    ])

    #Batch predict
    filenames = sorted([  #
        f for f in os.listdir(args.chars_dir)  #
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))  #
    ])

    predictions_map = {}  # Stores fname -> prediction
    ordered_predictions = []  # Stores predictions in the sorted order of filenames

    for fname in filenames:
        path = os.path.join(args.chars_dir, fname)  #
        img = Image.open(path).convert('L')  #
        inp = transform(img).unsqueeze(0).to(device)  #

        with torch.no_grad():  #
            logits = model(inp)  #
            orig_pred = int(logits.argmax(dim=1).item())  #
            repaired_pred = feature_based_repair(inp[0], orig_pred)  #

        predictions_map[fname] = repaired_pred  #
        ordered_predictions.append(repaired_pred)
        print(f"{fname} → {repaired_pred}")  #

    #Save results to CSV
    out_csv = os.path.join(args.chars_dir, "predictions.csv")  #
    with open(out_csv, 'w') as f:  #
        f.write("filename,prediction\n")  #
        for fn, pd_val in predictions_map.items():  #
            f.write(f"{fn},{pd_val}\n")  #
    print(f"\nAll predictions completed, results saved to: {out_csv}")  #

    #Load base image and bounding boxes, then draw predictions
    try:
        with open(args.boxes_info_path, 'r') as f:
            boxes_data = json.load(f)

        # The 'boxes' in boxes_info.json correspond to char_00, char_01, etc.,
        # matching the order of 'ordered_predictions'.
        bounding_boxes = boxes_data["boxes"]

        # Verify the image source if needed, though args.base_image_path is used directly.
        # stored_image_source = boxes_data.get("image_source_for_boxes")
        # if stored_image_source and stored_image_source != args.base_image_path:
        #    print(f"Warning: Base image path '{args.base_image_path}' differs from "
        #          f"source in boxes_info.json '{stored_image_source}'. Using provided path.")

        if len(ordered_predictions) != len(bounding_boxes):
            print(f"Warning: Number of predictions ({len(ordered_predictions)}) "
                  f"does not match number of boxes ({len(bounding_boxes)}). "
                  "Annotation might be misaligned. Drawing for the minimum available.")
            min_len = min(len(ordered_predictions), len(bounding_boxes))
            ordered_predictions = ordered_predictions[:min_len]
            bounding_boxes = bounding_boxes[:min_len]

        base_img = cv2.imread(args.base_image_path)
        if base_img is None:
            print(f"Error: Could not load base image from {args.base_image_path}")
            return

        # Ensure base_img is BGR for colored text and boxes.
        # The deskewed mask is likely grayscale.
        if len(base_img.shape) == 2 or base_img.shape[2] == 1:
            output_img_with_predictions = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)
        else:
            output_img_with_predictions = base_img.copy()

        for i, prediction_label in enumerate(ordered_predictions):
            if i < len(bounding_boxes):
                x, y, w, h = bounding_boxes[i]
                label_text = str(prediction_label)

                font_face = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.6
                font_thickness = 2
                text_color = (0, 0, 255)  # 红色 (BGR格式)
                (text_width, text_height), baseline = cv2.getTextSize(label_text, font_face, font_scale, font_thickness)
                text_x = x
                text_y = y - baseline

                if text_y < text_height:
                    text_y = y + text_height + (baseline // 2)

                cv2.putText(output_img_with_predictions, label_text, (text_x, text_y),
                            font_face, font_scale, text_color, font_thickness)

        cv2.imwrite(args.output_image_path, output_img_with_predictions)
        print(f"Predictions drawn on image and saved to: {args.output_image_path}")

    except FileNotFoundError:
        print(
            f"Error: Boxes info file not found at '{args.boxes_info_path}' or base image '{args.base_image_path}' not found.")
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{args.boxes_info_path}'.")
    except Exception as e:
        print(f"An error occurred while drawing predictions: {e}")


if __name__ == "__main__":
    main()