import cv2
import torch
import numpy as np
import time

# Imports from YOLOX
from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.exp import get_exp
from yolox.utils import postprocess

# --- Settings ---
MODEL_PATH = "yolox_l.pth"                    # Path to the model file
INPUT_SIZE = (640, 640)                       # Input size for YOLOX model
CONF_THRESHOLD = 0.4                          # Confidence threshold for detection
NMS_THRESHOLD = 0.5                           # NMS (Non-Maximum Suppression) threshold
GRADIENT_ALPHA = 0.6                          # Alpha for gradient overlay (closer to 0.0 is more transparent)
MIX_ALPHA = 0.9                               # Weight for the previous frame in mix mode (smaller value creates stronger afterimages)

# --- Check for GPU availability ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


def setup_model(model_path):
    """Function to load and set up the YOLOX model."""
    exp = get_exp(None, "yolox-l")
    model = exp.get_model()
    model.eval()
    
    print(f"Loading model from {model_path}...")
    try:
        # Set weights_only=True for security
        ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model"])
    except FileNotFoundError:
        print(f"Error: Model file not found. Please check if '{model_path}' is in the correct directory.")
        return None
    
    # Move model to the selected device (GPU/CPU)
    model.to(device)
    
    print("Model loaded successfully.")
    return model

def generate_color_palette(num_classes):
    """Function to generate a colorful palette based on the number of classes."""
    np.random.seed(42)
    colors = (np.random.rand(num_classes, 3) * 255).astype(np.uint8)
    return colors

def create_gradient_mask(w, h, color1, color2):
    """Function to create a linear gradient mask between two colors."""
    gradient = np.zeros((h, w, 3), dtype=np.float32)
    # Avoid division by zero if width is 1
    if w <= 1:
        ratio = 0
    for i in range(w):
        if w > 1:
            ratio = i / (w - 1)
        inv_ratio = 1.0 - ratio
        r = int(color1[0] * inv_ratio + color2[0] * ratio)
        g = int(color1[1] * inv_ratio + color2[1] * ratio)
        b = int(color1[2] * inv_ratio + color2[2] * ratio)
        gradient[:, i] = [b, g, r]
    return gradient.astype(np.uint8)

# --- Main Processing ---
if __name__ == "__main__":
    model = setup_model(MODEL_PATH)
    if model is None:
        exit()

    palette1 = generate_color_palette(len(COCO_CLASSES))
    palette2 = generate_color_palette(len(COCO_CLASSES))
    preproc = ValTransform(legacy=False)

    # --- Variables for mode management ---
    effect_mode = 0  # 0: YOLOX Negative Mode, 1: Mix Mode
    prev_frame = None  # Variable to store the previous frame for mix mode

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        exit()

    print("\nCamera started.")
    print("------------------------------------")
    print("q: Quit the program.")
    print("m: Switch effect mode.")
    print("------------------------------------")
    print("Current mode: YOLOX & Negative")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Exiting...")
            break
        
        # Switch mode with 'm' key
        if key == ord('m'):
            effect_mode = 1 - effect_mode  # Toggle between 0 and 1
            if effect_mode == 1:
                print("Mode: Mix Effect")
                # Initialize prev_frame on the first frame of mix mode
                prev_frame = np.zeros_like(frame, dtype=np.float32)
            else:
                print("Mode: YOLOX & Negative")

        output_frame = None

        # --- Process frame based on the current mode ---

        # --- Mode 0: YOLOX & Negative Mode ---
        if effect_mode == 0:
            # 1. Invert the base frame (negative effect)
            base_frame = cv2.bitwise_not(frame)

            # 2. Perform YOLOX inference on the original 'frame' for better accuracy
            original_h, original_w = frame.shape[:2]
            img, _ = preproc(frame, None, INPUT_SIZE)
            scale = min(INPUT_SIZE[0] / original_h, INPUT_SIZE[1] / original_w)
            
            img_tensor = torch.from_numpy(img).unsqueeze(0).to(device)
            with torch.no_grad():
                outputs = model(img_tensor)
                outputs = postprocess(
                    outputs, len(COCO_CLASSES), CONF_THRESHOLD, NMS_THRESHOLD, class_agnostic=False
                )

            # 3. Draw detection results on the inverted base frame
            if outputs[0] is not None:
                output = outputs[0].cpu()
                bboxes = output[:, 0:4]
                cls_ids = output[:, 6]
                bboxes /= scale

                for i in range(len(bboxes)):
                    box = bboxes[i].int().numpy()
                    cls_id = int(cls_ids[i])
                    x1, y1, x2, y2 = box
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                    width, height = x2 - x1, y2 - y1
                    
                    if width > 0 and height > 0:
                        # Extract the Region of Interest (ROI) from the inverted frame
                        roi = base_frame[y1:y2, x1:x2]
                        # Generate the gradient mask
                        color1, color2 = palette1[cls_id], palette2[cls_id]
                        gradient_mask = create_gradient_mask(width, height, color1, color2)
                        # Blend the ROI with the gradient mask
                        blended_roi = cv2.addWeighted(roi, 1.0 - GRADIENT_ALPHA, gradient_mask, GRADIENT_ALPHA, 0)
                        # Place the blended ROI back into the inverted frame
                        base_frame[y1:y2, x1:x2] = blended_roi
            
            output_frame = base_frame

        # --- Mode 1: Mix Effect Mode ---
        elif effect_mode == 1:
            current_frame_float = frame.astype(np.float32)
            mixed_frame = cv2.addWeighted(current_frame_float, 1.0 - MIX_ALPHA, prev_frame, MIX_ALPHA, 0)
            prev_frame = mixed_frame
            output_frame = mixed_frame.astype(np.uint8)

        # --- Display the result ---
        cv2.imshow("YOLOX Pop Art Camera", output_frame)

    # --- Cleanup ---
    cap.release()
    cv2.destroyAllWindows()
