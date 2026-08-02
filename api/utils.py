from PIL import Image
import io
import cv2
import numpy as np


def simple_xray_validator(content: bytes) -> bool:
    image_array = np.frombuffer(content, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        return False

    # --- ADDED: Color Saturation Check ---
    # Medical X-rays are grayscale. If R, G, and B differ significantly, it's a logo/photo.
    b, g, r = cv2.split(image)
    # Calculate the mean absolute difference between channels
    rg_diff = np.mean(cv2.absdiff(r, g))
    rb_diff = np.mean(cv2.absdiff(r, b))
    
    if rg_diff > 8 or rb_diff > 8: # Threshold for "colorfulness"
        return False

    # 1. Convert to Grayscale for existing logic
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 2. Feature: Variance of Pixel Intensity
    std_dev = np.std(gray)

    # 3. Feature: Center ROI brightness check
    h, w = gray.shape
    center_roi = gray[h // 4: 3 * h // 4, w // 4: 3 * w // 4]
    center_avg = np.mean(center_roi)

    # Thresholds
    if std_dev < 40:
        return False
    if center_avg > 180:
        return False
        
    return True

def process_xray(content):
    # 1. Load and convert to RGB
    image = Image.open(io.BytesIO(content)).convert('RGB').resize((224, 224))
    img_array = np.array(image).astype('float32') / 255.0
    
    # 2. ImageNet Normalization (This MUST be here)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_array = (img_array - mean) / std
    
    # 3. Transpose and ensure final type is float32
    img_final = img_array.transpose(2, 0, 1)
    
    # Use astype('float32') one last time to be safe
    return np.expand_dims(img_final.astype('float32'), axis=0)