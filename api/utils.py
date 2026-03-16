from PIL import Image
import io
import numpy as np
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