# ParkVision AI 🚗🏙️

**Intelligent Urban Parking Analytics & Space Optimisation Platform**

**Student Name:** Saurav Kamble  
**Candidate ID:** 1000323  
**Course:** CRS Artificial Intelligence - Machine Learning and Deep Learning  
**School:** Ryan Global School  

> **Live Web Application:** [Launch ParkVision AI on Streamlit Cloud](https://iadai201-1000323-saurav-k27fxpcgphsfswtgmohm9t.streamlit.app)

---

## 📖 Project Overview

Urban traffic congestion is heavily exacerbated by drivers searching for parking spaces. **ParkVision AI** is a computer vision-based decision-support platform designed for the *UrbanFlow AI Pvt. Ltd.* scenario. It analyzes unstructured parking lot imagery from CCTV cameras at the slot-level, localizing every parking bay and classifying it as either **Occupied** (Red) or **Vacant/Available** (Green).

The system processes these detections to output real-time capacity metrics and automated congestion routing recommendations (Low, Moderate, High) via a user-friendly web dashboard.

## 🚀 Getting Started (How to Run Locally)

Follow these steps to run the ParkVision AI Streamlit dashboard on your local machine.

### Prerequisites
* Python 3.8 or higher installed on your system.
* Git installed.

### Installation Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/monkey8D8luffy/IADAI201-1000323-saurav.git
   cd IADAI201-1000323-saurav
   ```

2. **Create and activate a virtual environment (Recommended):**
   * **Windows:**
     ```bash
     python -m venv .venv
     .venv\Scripts\activate
     ```
   * **Mac/Linux:**
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: The `requirements.txt` uses `opencv-python-headless` for cloud compatibility. It works perfectly for the Streamlit app locally as well).*

4. **Run the Streamlit Application:**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser:**
   The application will automatically open in your default web browser at `http://localhost:8501`.

---

## 🔬 1. Main Research Findings Influencing the Project

The development of ParkVision AI was guided by key academic and technical research:

* **PKLot Benchmark Studies (de Almeida et al., 2015):** Research demonstrated that lighting variations (shadows, nighttime) and weather (rain) are the primary causes of detection failure in smart parking systems. This influenced my decision to implement **CLAHE (Contrast Limited Adaptive Histogram Equalization)** to normalize luminance before inference.
* **Full-Image Detection vs. Cropped Classification:** Unlike traditional approaches that use MobileNet to classify pre-cropped 224x224 images of single slots, research into **Ultralytics YOLOv8** proved that state-of-the-art anchor-free object detection can process entire parking lots simultaneously in one forward pass, making it highly scalable for varying camera angles.

## 📊 2. Data Preparation for Training & Testing

The system was trained on a YOLOv8-formatted subset of the **PKLot Dataset** containing sunny, cloudy, and rainy conditions.

* **Class Normalization:** Raw dataset labels were cleaned and mapped strictly to 2 active classes to prevent logical confusion: `0: space-empty` and `1: space-occupied`.
* **Dataset Splitting:** Data was strictly partitioned into **70% Training**, **15% Validation** (for hyperparameter tuning), and **15% Testing** (unseen data for final evaluation).
* **Preprocessing:**
  * **Resizing:** Images were standardized to `640x640` pixels to preserve spatial resolution for distant vehicles.
  * **Illumination Normalization:** Applied CLAHE in the CIE LAB color space to equalize harsh shadows and underexposed nighttime imagery.
* **Online Augmentations:** Applied Mosaic (1.0), MixUp (0.1), rotation (15°), and horizontal flipping (0.5) to prevent the neural network from merely memorizing the static asphalt background of the training camera.

## 🧠 3. Models Used, Training Parameters & Techniques

The core intelligence is powered by **YOLOv8s** (Small architecture) due to its optimal balance of high mean Average Precision (mAP) and real-time inference speed.

**Training Hardware & Configuration:**
* **Hardware:** NVIDIA GeForce RTX 5090 GPU (Local Workstation)
* **Epochs & Early Stopping:** `epochs=50` with early stopping `patience=12` to prevent overfitting.
* **Batch Size:** `batch=16`
* **Optimizer:** `AdamW` with initial learning rate `lr0=0.001` and cosine annealing (`cos_lr=True`).
* **Loss Weights:** Box regression gain `box=7.5` and classification gain `cls=0.5`.

**Hybrid Inference Engine:**
Raw YOLO predictions are routed through a custom Python `ParkVisionEngine`, which applies **Geometric Aisle Filtering** (deleting false "empty" boxes detected in driving lanes) and **Class-Priority Non-Maximum Suppression** (ensuring overlapping empty/occupied boxes always default to Occupied).

## 📈 4. Performance Metrics & Simulation Results

The model (`train-11` checkpoint / `best.pt`) was rigorously evaluated against the 15% testing split:

* **Detection Accuracy:** Achieved **>96% precision** on `space-occupied` classifications.
* **Recall & Robustness:** By integrating the CLAHE preprocessor, the model successfully recovered recall on severely shadowed and nighttime test frames that initially failed under raw RGB inference.
* **Confusion Matrix Analysis:** Simulation outputs demonstrated a clean separation between empty and occupied states, with false-positive background detections heavily minimized by the Hybrid Engine's geometric filtering.

## 📸 5. Project Demo & Screenshots

*(Instructor Note: Screenshots of the live application running on Streamlit Community Cloud)*

### Dashboard Interface & Real-Time Upload
![Upload Interface](https://github.com/monkey8D8luffy/IADAI201-1000323-saurav/blob/main/Screenshot%202026-10-30%20172202.jpg?raw=true)
*The clean Streamlit UI allowing drag-and-drop image upload and dynamic confidence threshold adjustments.*

### Live AI Inference & Telemetry Metrics
![Telemetry Dashboard](https://github.com/monkey8D8luffy/IADAI201-1000323-saurav/blob/main/Screenshot%202026-10-30%20172241.jpg?raw=true)
*The deployed YOLOv8 model rendering visual smart overlays (Green = Empty, Red = Occupied) alongside real-time calculated metrics (Total, Occupied, Available, and Utilization %).*

### Congestion Insight Logic & Recommendations
![Recommendation Engine](https://github.com/monkey8D8luffy/IADAI201-1000323-saurav/blob/main/Screenshot%202026-10-30%20172317.jpg?raw=true)
*Automated driver recommendations dynamically generated based on occupancy thresholds (Low <40%, Moderate 40-75%, High >75%).*

## 🛠️ Repository Structure & Cloud Deployment

This project is fully containerized for **Streamlit Community Cloud (Linux Debian)**.

* `app.py`: Streamlit frontend UI and dashboard logic.
* `engine.py`: ParkVisionEngine (CLAHE processing, YOLO inference, geometric filtering).
* `train.py`: YOLOv8 training configuration script.
* `runs/detect/train-11/weights/best.pt`: Production model weights.
* `requirements.txt`: Python dependencies (`streamlit`, `ultralytics`, `opencv-python-headless`, etc.).
* `packages.txt`: Critical Linux system drivers (`libgl1`, `libglib2.0-0`) required to run OpenCV headlessly in the cloud.