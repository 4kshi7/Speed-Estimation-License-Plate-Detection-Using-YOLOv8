# 🚗 Speed Estimation & License Plate Detection Using YOLOv8

This project is an **AI-based system** that detects **vehicles**, estimates **speed**, and extracts **license plates** using **YOLOv8** and **Tesseract OCR**. The system efficiently analyzes **real-time traffic** from a video feed and flags speed violations.

---

## 📌 **Features**
✅ **Real-time vehicle detection** using `YOLOv8`  
✅ **Speed estimation** with improved accuracy  
✅ **License plate extraction** using `Tesseract OCR`  
✅ **Optimized for GPU acceleration (CUDA Support)**  
✅ **Supports batch processing for efficient logging**  

---

## 🛠 **Tech Stack**
| Component                | Technology Used |
|--------------------------|----------------|
| **Object Detection**     | YOLOv8 (Ultralytics) |
| **Speed Estimation**     | OpenCV & Euclidean Distance |
| **License Plate OCR**    | Tesseract OCR |
| **Data Handling**        | Pandas, NumPy |
| **Hardware Acceleration** | CUDA / GPU Support (Optional) |

---

## 📂 **Project Structure**
```
📂 SPEED E
 ┣ 📜 license_plate_detector.pt   # (License plate detection model)
 ┣ 📜 new.py                      # (Other scripts - optional)
 ┣ 📜 SpeedEstimation.py          # (Main script - Run this!)
 ┣ 📜 vehicle_log.txt             # (Log file for detected vehicles)
 ┣ 📜 vehicle_log.xlsx            # (Excel sheet for logs)
 ┣ 📜 violations.txt              # (List of speed violations)
 ┣ 📜 yolov8n.pt                  # (YOLOv8 model for object detection)
```

---

## 🚀 **Installation & Setup**
### **1️⃣ Prerequisites**
Make sure you have **Python 3.8+** installed on your system.

### **2️⃣ Install Dependencies**
Run the following command in your terminal or command prompt:
```bash
pip install -r requirements.txt
```
*(If `requirements.txt` is not available, manually install these packages:)*  
```bash
pip install ultralytics opencv-python numpy pandas torch torchvision torchaudio scipy pytesseract
```

### **3️⃣ Download & Place YOLO Model**
Download **YOLOv8 model** from [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) and place it in the project folder:
```bash
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt -O yolov8n.pt
```
*(Or manually place `yolov8n.pt` in the project directory.)*

### **4️⃣ Install Tesseract OCR (For License Plate Recognition)**
- **Windows:** Download & Install Tesseract from [here](https://github.com/UB-Mannheim/tesseract/wiki)
- **Linux (Debian/Ubuntu):**
  ```bash
  sudo apt install tesseract-ocr
  ```
- **Mac (Homebrew):**
  ```bash
  brew install tesseract
  ```

Ensure the correct path to Tesseract is set in the script:
```python
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Windows
```

### **5️⃣ Run the Script**
Once everything is set up, run:
```bash
python SpeedEstimation.py
```
🎥 The system will start processing the video and displaying results.

---

## 📊 **Accuracy & Performance**
| Model Used | FPS (Speed) | Detection Accuracy |
|------------|------------|--------------------|
| **YOLOv8n** (Nano) | ⚡ Fast (~80+ FPS) | 🔴 Moderate |
| **YOLOv8m** (Medium) | ✅ Balanced (~40 FPS) | 🟠 Good |
| **YOLOv8l** (Large) | 🎯 Slower (~20 FPS) | 🟢 High Accuracy |

💡 **For Best Accuracy**, use:
```python
vehicle_model = YOLO("yolov8l.pt")  # Large Model
```
If speed is more important than accuracy, keep using:
```python
vehicle_model = YOLO("yolov8n.pt")  # Nano Model (Fastest)
```

---

## ⚡ **Performance Optimizations**
This script is optimized for **real-time execution**:
- **✅ Uses GPU (CUDA) when available** for fast object detection
- **✅ Runs license plate OCR only every 5 frames** to prevent lag
- **✅ Reduces `cv2.imshow()` updates** for smooth visualization

---

## 📩 **Troubleshooting**
### ❌ **Video File Not Found**
- Ensure the **video path is correct** in `SpeedEstimation.py`
- Example:
```python
video_path = r"C:\Users\your_name\Downloads\traffic.mp4"
```
- Use absolute paths if necessary.

### ❌ **Tesseract Not Found**
- If Tesseract OCR is installed but **not detected**, manually set its path:
```python
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"  # Linux
pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"  # Windows
```

### ❌ **YOLO Model Not Found**
- Download and place `yolov8n.pt` in the **project folder**.
- Check if `ultralytics` is installed correctly.

---

## 🤝 **Contributing**
Want to improve this project? Feel free to:
1. **Fork this repository**
2. **Make improvements**
3. **Create a pull request (PR)**

---

## 🏆 **Credits & Resources**
- **YOLOv8 (Ultralytics)**: [GitHub](https://github.com/ultralytics/ultralytics)
- **Tesseract OCR**: [GitHub](https://github.com/tesseract-ocr/tesseract)
- **OpenCV**: [Website](https://opencv.org/)
- **Torch (PyTorch)**: [Website](https://pytorch.org/)

```

