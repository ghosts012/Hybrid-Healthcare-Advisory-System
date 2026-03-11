from fastapi import FastAPI, File, UploadFile, HTTPException
import onnxruntime as ort
from api.utils import process_xray
from api.regression import calculate_severity

app = FastAPI(title="Healthcare Advisory Vision API")

# Load the model into memory ONCE at startup
try:
    # Use the 44MB file we just successfully exported
    ort_session = ort.InferenceSession("models/vision_production_v1.onnx")
    print("ONNX Model Loaded Successfully")
except Exception as e:
    print(f"Error loading model: {e}")

@app.post("/predict/severity")
async def predict_severity(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    # 1. Read and Preprocess
    content = await file.read()
    input_tensor = process_xray(content)

    # 2. Run Inference via ONNX
    ort_inputs = {ort_session.get_inputs()[0].name: input_tensor}
    logits = ort_session.run(None, ort_inputs)[0] # Shape: [[val1, val2]]

    # 3. Clinical Severity Logic (Regression)
    severity_score = calculate_severity(logits)

    # 4. Categorization
    status = "Normal"
    if severity_score > 70:
        status = "High Risk / Critical"
    elif severity_score > 30:
        status = "Moderate Risk / Review Required"

    return {
        "filename": file.filename,
        "clinical_metrics": {
            "pneumonia_probability": f"{severity_score}%",
            "severity_index": severity_score,
            "status": status
        },
        "recommendation": "Consult Radiologist immediately" if severity_score > 70 else "Routine check-up"
    }