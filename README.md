# 🚗 License Plate Recognition (LPR) System

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-Classical_CV-green.svg)
![EasyOCR](https://img.shields.io/badge/EasyOCR-PyTorch-orange.svg)
![Flask](https://img.shields.io/badge/Flask-Web_App-lightgrey.svg)

## 📌 Overview
This project is an end-to-end License Plate Recognition (LPR) pipeline built for a Digital Image Processing course. Rather than relying entirely on deep learning bounding boxes (like YOLO), this system explores classical morphological operations and heuristic fallbacks to localize and extract Indian license plates in uncontrolled environments.

---

## 💾 Dataset
The system was evaluated against the **Indian Vehicle Dataset**, which contains varied images of Indian vehicles with diverse lighting conditions, angles, and vehicle types.
* **Source:** [Kaggle - Indian Vehicle Dataset by saisirishan](https://www.kaggle.com/datasets/saisirishan/indian-vehicle-dataset)

---

## 🛠️ Tech Stack
* **Core Language:** Python 3
* **Computer Vision:** OpenCV (`cv2`), `imutils`
* **Optical Character Recognition:** EasyOCR (PyTorch-based)
* **Data Processing & Evaluation:** Pandas, NumPy, Regex (`re`)
* **Web Interface:** Flask, HTML5, CSS3, Vanilla JavaScript

---

## 📂 Project Structure
```text
LPR-System/
│
├── .gitignore
├── requirements.txt
├── README.md
│
├── src/                        # Core DIP and OCR modules
│   ├── preprocess.py           # Grayscale & bilateral filtering
│   ├── detect.py               # Auto-Canny & geometric contours
│   ├── extract.py              # Bounding box cropping & enhancement
│   ├── segment.py              # Character segmentation (Otsu's Thresholding)
│   ├── ocr.py                  # Text extraction & heuristic validation
│   └── evaluate.py             # Levenshtein distance metrics calculations
│
├── web_app/                    # Flask interactive dashboard
│   ├── app.py                  # Routing and pipeline integration
│   ├── static/
│   │   ├── css/style.css       # Custom UI styling
│   │   ├── js/main.js          # Client-side interactions
│   │   └── uploads/            # Temporary storage for processing
│   └── templates/
│       ├── home.html           # Landing page
│       └── app.html            # ALPR scanner interface
│
├── main_pipeline.py            # CLI tool for single-image testing
├── prepare_dataset.py          # Script to parse Kaggle XMLs to ground_truth.csv
└── run_evaluation.py           # Bulk evaluator for dataset metrics
```

---

## ⚙️ The Pipeline Architecture
1. **Preprocessing:** Grayscale conversion, Bilateral Filtering (noise reduction while preserving edges).
2. **Detection:** Dynamic Auto-Canny edge detection and geometric contour validation (aspect ratio, area constraints).
3. **Extraction & Enhancement:** Mathematical deskewing, ROI cropping, and Otsu's Thresholding.
4. **Optical Character Recognition:** Text extraction via EasyOCR.
5. **Heuristic Post-Processing:**
   - **Regex Filtering:** Validates against standard Indian (`MH12AB1234`) and Bharat (`21BH1234AA`) formats.
   - **Positional Correction:** Corrects common OCR geometry errors (e.g., swapping 'O' for '0' based on character index).
   - **State-Code Dictionary:** Uses Levenshtein distance to suggest valid alternatives for ambiguous state codes (e.g., `HH` -> `MH`, `HR`).
   - **Dynamic Fallback:** Automatically defaults to whole-image scanning if morphological edge detection fails due to harsh shadows or extreme zoom.

---

## 📊 Performance Metrics
Evaluated on a challenging, uncleaned Kaggle dataset of **1,696 Indian vehicle images**:
* **System Exact Match Accuracy:** 34.91%
* **Overall Character Accuracy:** 63.77%
* **Isolated ROI Character Accuracy:** 83.39% *(Accuracy when the plate was successfully localized by the classical CV pipeline)*

*Note: The primary bottleneck identified in this classical CV approach is the localization stage under extreme lighting variances, which was partially mitigated by the heuristic fallback mechanism.*

---

## 🚀 How to Run

### 1. Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/YashChugani/License-Plate-Recognition-System.git
cd License-Plate-Recognition-System
pip install -r requirements.txt
```


### 2. Run the Web App (Interactive Interface)

Launch the Flask dashboard to upload images and view the extraction process:

```bash
cd web_app
python app.py
```

Open the browser and navigate to:

```
http://127.0.0.1:5000
```


### 3. Run the Bulk Evaluator

Ensure the dataset is downloaded into the `raw_kaggle_data` folder, then run:

```bash
# Synchronize raw Kaggle dataset XMLs into a Ground Truth CSV
python prepare_dataset.py

# Run the bulk evaluation pipeline
python run_evaluation.py
```

---

## 👨‍💻 Author

**Yash Chugani**