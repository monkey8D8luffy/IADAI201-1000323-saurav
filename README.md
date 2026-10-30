# ParkVision AI 🚗🅿️
**Intelligent Urban Parking Analytics & Space Optimisation Platform**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://iadai201-1000323-saurav-k27fxpcgphsfswtgmohm9t.streamlit.app/)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/monkey8D8luffy/IADAI201-1000323-saurav)

**ParkVision AI** is a software-based solution designed to monitor parking availability in real-time using existing CCTV camera feeds. Instead of relying on expensive in-ground sensors, this system operates at the slot level using deep learning, actively locating every parking bay and classifying it as either "Occupied" or "Available".

---

## 🌟 Key Features

*   **Real-Time Slot Detection:** Utilizes YOLOv8 to process entire parking lot frames in a single pass, identifying both occupied and empty parking spaces.
*   **Adaptive Lighting Normalization:** Integrates CLAHE (Contrast Limited Adaptive Histogram Equalization) to automatically enhance dark and nighttime images, revealing vehicles hidden in shadows.
*   **Hybrid Inference Engine (ParkVisionEngine):** A custom logic filter that:
    *   Removes false "empty space" predictions in driving aisles using geometric filtering.
    *   Applies priority suppression to ensure parked cars are never ignored if overlapping bounding boxes occur.
*   **Live KPI Dashboard:** Displays real-time metrics including Total Slots, Occupied Slots, Available Slots, and Occupancy Percentage.
*   **Automated Congestion Alerts:**
    *   🟢 **Low (<40%):** "Slots available — proceed to park."
    *   🟡 **Moderate (40%-75%):** "Slots available — proceed with caution. Check highlighted green bays."
    *   🔴 **High (>75%):** "Parking full — try another area."

## 🛠️ Tech Stack

*   **Deep Learning Model:** YOLOv8 (Ultralytics)
*   **Computer Vision:** OpenCV (CLAHE Lighting Normalization)
*   **Frontend/UI:** Streamlit
*   **Language:** Python
*   **Deployment:** Streamlit Community Cloud

## 📊 Dataset

The model was trained using the **PKLot (Parking Lot Database)**. 
*   **Variability:** Includes thousands of real-world images across Sunny, Cloudy, and Rainy conditions.
*   **Split:** 70% Training, 15% Validation, 15% Testing.
*   **Classes:** `0: space-empty`, `1: space-occupied`
*   **Augmentation:** Mosaic, MixUp, random rotations, and flips were applied to ensure robust geometric learning.

## 🚀 Installation & Setup

To run this project locally on your machine, follow these steps:

### 1. Clone the repository
```bash
git clone https://github.com/monkey8D8luffy/IADAI201-1000323-saurav.git
cd IADAI201-1000323-saurav
```

### 2. Create a Virtual Environment (Optional but recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
*Note for Linux/Cloud deployment: Ensure core system graphics drivers are installed (e.g., `libgl1` and `libglib2.0-0`), as specified in `packages.txt`.*

### 4. Run the Application
```bash
streamlit run app.py
```

## 💻 Usage

1. Open the local Streamlit URL provided in your terminal (usually `http://localhost:8501`).
2. Use the drag-and-drop interface to upload a CCTV image of a parking lot.
3. Adjust the AI confidence threshold using the sidebar if necessary.
4. View the side-by-side comparison of the raw feed and the AI overlay (Green = Empty, Red = Occupied).
5. Check the live KPI metrics and congestion recommendations at the top of the dashboard.

## 👨‍💻 Author

**Saurav Kamble**
*   **Registration Number:** 1000323
*   **Course:** Machine Learning and Deep Learning (Artificial Intelligence)
*   **Institution:** Ryan Global School
