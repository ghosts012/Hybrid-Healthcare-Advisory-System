from PIL import Image
import io
import numpy as np
def process_xray(content):
    image = Image.open(io.BytesIO(content)).convert('RGB').resize((224, 224))
    return np.expand_dims(np.array(image).transpose(2,0,1).astype('float32')/255.0, axis=0)