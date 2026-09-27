# 🌿 AI WASTE DOCTOR – Intelligent Waste Classification System

![Python Version](https://img.shields.io/badge/Python-3.10%2B-blue)
![GUI Framework](https://img.shields.io/badge/GUI-PySide6%20%2F%20Qt-emerald)
![Computer Vision](https://img.shields.io/badge/Vision-OpenCV-green)
![AI Engine](https://img.shields.io/badge/AI-TensorFlow%20%2F%20Keras-orange)

An intelligent, real-time desktop application built for Class 10 science-fair projects to demonstrate how Artificial Intelligence (Computer Vision and Deep Learning) can automate waste sorting for a cleaner planet.

---

## 🌟 Key Features

1. **Live Camera Classification**:
   - Real-time webcam feed with an on-screen **SCAN ZONE** overlay.
   - Real-time FPS overlay and connection status indicators.
   - Automatic cropping of the scan zone region for inference.

2. **Intelligent AI Prediction Panel**:
   - Dynamic reading of waste classes from `model/labels.txt`.
   - Visual class indicators (🌿 Organic Waste, 📄 Paper/Cardboard, ♻️ Plastic/Metal).
   - Animated progress bars for all category probabilities.
   - Clear category disposal guidance boxes.

3. **Stability Filter (Prediction Smoothing)**:
   - Rolling sliding-window filter across the latest 10 frames (configurable in Settings).
   - Confidence threshold check (default 70%). Shows `UNCERTAIN – MOVE OBJECT CLOSER` when unconfident.

4. **Dual Scan Modes**:
   - **Live Mode**: Continuous frame-by-frame classification.
   - **Manual Scan Mode**: Capture single frozen frames via the **SCAN OBJECT** button for live demonstrations.

5. **Scientific Experiment Mode**:
   - Test how environmental factors (*Normal Lighting, Dim Lighting, Harsh Lighting, Different Background, Tilted Object, Partial Occlusion*) impact AI accuracy.
   - Automated ground-truth comparison (`Yes`/`No` match).
   - Saves trial logs to `data/results.csv`.

6. **Experiment Results Dashboard**:
   - Live summary metrics (*Total Tests, Correct Predictions, Incorrect Predictions, Overall Accuracy %*).
   - Breakdown table per environmental condition.
   - Embedded **Matplotlib** bar chart visualization.
   - Action controls for refreshing dashboard and clearing data.

7. **✨ Science Fair Display Mode**:
   - Fullscreen presentation view (`showFullScreen()`) designed for booth visitors and judges standing several feet away.
   - Giant readable typography and high-contrast visuals.
   - Press **Escape** (`ESC`) to exit fullscreen mode.

8. **Demo Mode Fallback**:
   - If TensorFlow or a trained model file is not detected, the app automatically enables **DEMO MODE** with simulated predictions so the full GUI can be presented seamlessly.

---

## 📁 Project Folder Structure

```
ai_waste_doctor/
│
├── main.py                       # Main application launcher
├── config.py                     # Persistent JSON configuration manager
├── requirements.txt              # Required Python packages
├── README.md                     # Documentation & setup guide
│
├── ui/                           # Desktop User Interface (PySide6)
│   ├── __init__.py
│   ├── main_window.py            # Main window container & navbar tabs
│   ├── styles.py                 # Modern Dark Navy & Emerald Green QSS theme
│   ├── widgets.py                # Video label, progress bars, metric cards
│   ├── live_tab.py               # Live camera feed & AI Prediction panel
│   ├── experiment_tab.py         # Scientific experiment mode tab
│   ├── results_tab.py            # Results dashboard with Matplotlib charts
│   ├── how_it_works_tab.py       # Educational machine learning flowchart
│   ├── settings_tab.py           # Configuration tab
│   └── science_fair_view.py      # Fullscreen Science Fair presentation view
│
├── ai/                           # Machine Learning & Computer Vision
│   ├── __init__.py
│   ├── classifier.py             # Keras model loader, inference, smoothing
│   └── preprocessing.py          # Frame cropping, resizing, normalization
│
├── camera/                       # OpenCV Video Capture Thread
│   ├── __init__.py
│   └── camera_manager.py         # QThread webcam reader & overlay drawer
│
├── data/                         # Experiment Data Logging
│   ├── __init__.py
│   ├── experiment_logger.py      # CSV logger & metric calculation engine
│   ├── results.csv               # Experiment trial logs
│   └── config.json               # Persisted user settings
│
├── model/                        # AI Model & Category Labels
│   ├── labels.txt                # Target class labels
│   ├── keras_model.h5            # Keras model file
│   └── generate_sample_model.py  # Script to generate a starter model
│
└── assets/                       # App icons & graphics
    └── icons/
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- **Python 3.10** or **3.11** installed on Windows.

### 2. Create Virtual Environment
Open PowerShell or Command Prompt in the project folder:
```bash
py -m venv venv
```

Activate the environment:
```cmd
venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the Classifier
The checked-in `keras_model.h5` is only a starter model and is not trained on real waste photographs. For reliable object classification, create a dataset with this exact structure:

```
dataset/
├── Recyclable/
│   └── E-Waste/       # phones, laptops, chargers, keyboards, circuit boards
├── Dry Waste/
└── Wet Waste/
```

Add varied, clearly labeled photos to each folder, with the waste object visible and reasonably centered. For phones and electronics, add images under `Recyclable/E-Waste/`; they will be trained as `Recyclable`. Then train and replace the model:

```bash
python model/train_model.py --dataset dataset --epochs 25
```

The script uses transfer learning with MobileNetV2, writes `model/keras_model.h5`, and keeps the class order synchronized with `model/labels.txt`.

To create only a valid starter model for UI testing (not reliable classification):
```bash
python model/generate_sample_model.py
```

### 5. Run the Application
```bash
python main.py
```

## ☁️ Deploy the API on Render

The PySide6 desktop application runs locally on Windows. For a hosted service, this repository includes a separate FastAPI image-classification API for Render.

1. In Render, choose **New → Blueprint** and select this GitHub repository.
2. Render reads `render.yaml` and installs Python 3.11 from `runtime.txt`.
3. The API uses `requirements.render.txt`, which includes TensorFlow CPU support and excludes desktop-only packages.
4. After deployment, test:

```text
GET https://YOUR-RENDER-SERVICE.onrender.com/health
POST https://YOUR-RENDER-SERVICE.onrender.com/predict
```

For `/predict`, send an image as multipart form data with the field name `file`.

The deployed API returns the predicted category, confidence, all class probabilities, and whether the model is loaded. The Vercel frontend can call this API; the PySide6 desktop app remains the local Windows application.

---

## 🤖 Placing Your Trained AI Model

You can export a model directly from **Google Teachable Machine** or train a custom Keras model:

1. Copy your trained `.h5` model file to:
   `model/keras_model.h5`
   *(Alternatively, place a TensorFlow `SavedModel` directory in `model/saved_model/`)*
2. Update category labels in:
   `model/labels.txt`

### Expected `labels.txt` Format:
You can use numbered labels or plain category lines:
```
0 Organic Waste
1 Paper/Cardboard
2 Plastic/Metal
```
or simply:
```
Organic Waste
Paper/Cardboard
Plastic/Metal
```

---

## 🔧 Preprocessing Customization (Teachable Machine)

If using Google Teachable Machine exported models:
1. Open the **Settings** tab inside the app.
2. Change **Image Normalization Preprocessing Mode** to:
   `-1 to 1 (Teachable Machine: (pixel / 127.5) - 1.0)`

The application disables CUDA discovery on native Windows because TensorFlow 2.11+ does not support native-Windows GPU execution. This is expected and does not prevent CPU inference.
3. Save Configuration.

---

## ❓ Troubleshooting

- **No Camera Detected**:
  Check if your webcam is plugged in or change the **Webcam Device Index** (0, 1, 2) in the Settings tab.
- **Model Not Found**:
  If `model/keras_model.h5` is missing, the app automatically switches to **DEMO MODE**. Run `python model/generate_sample_model.py` to create a working starter model.
- **TensorFlow Warnings**:
  Ignore CPU instruction set optimization warnings printed by TensorFlow; they do not impact application execution.

---

## 🏆 Science Fair Presentation Tips
1. Use **✨ SCIENCE FAIR MODE** during your booth presentation.
2. Highlight how **Prediction Smoothing** prevents flicker during live camera movement.
3. Show judges your **Scientific Experiment Mode** data to prove how lighting affects computer vision accuracy!
