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

│   README.md
│   requirements.txt
│   
├───api 
├───dashboard
├───data
├───models
│       cnn_vision_v1.pth
│
└───src
    └───vision_module
            train_cnn.ipynb