# Deepfake Video Detection 🎭

This is a web-based application for detecting deepfake videos using **Flask** and **PyTorch**.  
Users can upload a video, and the system predicts whether it’s **REAL** or **FAKE**.

## 🔹 How It Works
1. **Upload Video** → User uploads a video through the web interface.  
2. **Preprocessing** → Frames are extracted with OpenCV, faces are detected & cropped, resized, and normalized.  
3. **Feature Extraction** → Each frame is passed through a pre-trained **ResNeXt50 CNN** to extract spatial features.  
4. **Sequence Learning** → Features are fed into an **LSTM** to capture temporal patterns across frames.  
5. **Classification** → The model outputs a probability for REAL or FAKE.  
6. **Result Display** → Flask shows the prediction and confidence score in the web UI.  

## 🚀 Tech Stack
- **Flask** (Web Framework)  
- **PyTorch** (Model Training & Inference)  
- **OpenCV** (Video Processing)  
- **face_recognition** (Face Detection)  

## 📂 Project Structure
```bash
  ├── app.py # Flask app entry point
  ├── models/ # Trained .pt model files
  ├── static/ # Static assets (CSS/JS)
  ├── templates/ # HTML templates (index.html, result.html)
  ├── test_videos/ # Sample test videos
  └── requirements.txt # Python dependencies
```
## ⚡ Usage
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/deepfake-detection.git
   cd deepfake-detection
   ```
2. Install dependencies:
```bash
  pip install -r requirements.txt
```

3. Run the app:
```bash
  python app.py
```
