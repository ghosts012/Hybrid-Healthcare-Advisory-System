# Hybrid Healthcare Advisory System
A distributed AI pipeline for medical diagnostics, combining cloud-accelerated Deep Learning with local inference and Retrieval-Augmented Generation (RAG).

## Distributed Environment Setup
Unlike traditional local setups, this system utilizes a Hybrid Development Architecture:

* Compute Engine: Google Colab T4 GPU (Remote Kernel).

* IDE Interface: VS Code via the Jupyter & Google Colab Extensions.

* Synchronization: Leverages a custom binary stream (Base64) and remote file system mapping to synchronize model artifacts between the cloud runtime and the local project repository.

## Vision Intelligence Module (In-Progress)
The system's core is a ResNet-18 architecture fine-tuned for the PneumoniaMNIST (MedMNIST) dataset.

### Technical Challenges & Niche Solutions
During implementation, several non-trivial environment constraints were addressed:

* **Remote-to-Local Serialization**: Encountered "ephemeral storage" limitations where standard files.download() triggers were intercepted by the VS Code sandbox. Resolved by implementing a Base64 binary data stream to force-download serialized weights.

## Roadmap & Next Steps
1. Model Serialization & Optimization (ONNX)
Moving the raw .pth state dictionary into the ONNX (Open Neural Network Exchange) format using Opset 18. This will decouple the project from the multi-gigabyte PyTorch dependency, allowing for lightweight local CPU inference.

2. Clinical Risk Analytics (Regression Layer)
Implementation of a Probabilistic Risk Scorer. This module will process the Softmax outputs from the Vision module through a regression function to calculate a Clinical Severity Index (0-100%), providing more granular data than binary classification.

3. Medical RAG Module (Retreival-Augmented Generation)
Development of a localized medical knowledge base.

Vector Database: Utilizing FAISS for indexing clinical protocols.

Advisory Logic: Linking high-risk regression scores to automated retrieval of patient-care guidelines.

## Repository Structure (For Now)

```text
Healthcare_Advisory_System/
├── api/
├── models/
│   ├── cnn_vision_v1.pth   
│   
├── src/
│   └── vision_module/    # Research & Training notebooks
|         |--train_cnn.ipynb
└── requirements.txt      
```

## 🛠️ Update: Phase 2 - Model Optimization & API Architecture
**Commit Date:** March 11, 2026
**Focus:** Reproducibility, Model Persistence, and Hybrid Inference Layer

### 🚀 Key Technical Milestones

#### 1. Transition to Hybrid Inference Architecture
We have successfully implemented the first two layers of our three-tier integration:
* **CNN Layer:** Feature extraction using ResNet-18 (MedMNIST).
* **Regression Layer:** A stochastic probability mapping that converts raw CNN logits into a **Clinical Severity Index (0-100%)**.
* **Next:** This index will serve as the primary trigger for the RAG-based Advisory System.

#### 2. Validation-Driven Checkpointing
To resolve issues with training variance, we moved away from static epoch-saving to a **Best-Model Checkpoint** strategy. 
* **The Problem:** Standard seeding was inconsistent due to non-deterministic CUDA atomic operations on the T4 GPU.
* **The Solution:** Implemented a stratified validation split (80/20) and a copy-on-improvement protocol. The model now only saves weights when a new global maximum in validation accuracy is achieved.

#### 3. Optimized ONNX Serialization
Successfully resolved the "Hollow Export" issue (0.08 MB files) by bypassing the TorchDynamo engine in favor of a **Legacy Trace Export**.
* **Optimization:** The final production model (`vision_production.onnx`) is now **44.7 MB**, ensuring all weights are baked into the computational graph.
* **Opset Alignment:** Standardized on Opset 18 to prevent version converter crashes.

### 4. Backend Implementation: FastAPI Service Layer
We transitioned from standalone script execution to a dedicated Backend Service architecture.

* Framework: FastAPI (Asynchronous Python Framework).

* Inference Engine: ONNX Runtime (CPU-optimized for low-latency clinical response).

* Workflow: The backend accepts an X-ray image via multipart/form-data, performs image normalization via PIL, executes the ONNX graph, and applies a logistic regression layer to calculate risk.

5. Sample API Output (The Hybrid Result)
The following JSON payload demonstrates the integration of the CNN Features and the Regression Mapping. This response serves as the input for the upcoming RAG module.

{
    "filename": "pneumonia_yes.jpeg",
    "clinical_metrics": {
        "pneumonia_probability": "79.46176147460938%",
        "severity_index": 79.46176147460938,
        "status": "High Risk / Critical"
    },
    "recommendation": "Consult Radiologist immediately"
}

---

### 📂 Updated Project Structure
```text
Healthcare_Advisory_System/
├── venv/                 # Virtual environment (Root)
├── api/
│   ├── main.py           # FastAPI entry point
│   ├── utils.py          # Image preprocessing utilities
│   └── regression.py     # Severity Index math logic
├── models/
│   ├── cnn_vision_v1.pth   # Best found PyTorch weights
│   └── vision_prodcution_v1.onnx # 44MB Production-ready model
├── src/
│   └── vision_module/    # Research & Training notebooks
|        |--train_cnn.ipynb
└── requirements.txt      # Updated with FastAPI & ONNX Runtime
```
