🚗 Vehicle License Plate Detection & Tracking
This project implements an end-to-end system to detect and track vehicle license plates in videos using YOLOv8 (custom-trained) for detection and SORT for tracking.

🔍 Built with PyTorch, OpenCV, and TensorRT engine for YOLOv8 inference.
📹 Works on input videos and outputs a new video with annotated bounding boxes and object IDs.

📌 Features
✅ License plate detection using custom-trained YOLOv8

✅ Object tracking using SORT (Simple Online and Realtime Tracking)

✅ Real-time video processing (video → annotated video)

✅ Output includes bounding boxes and unique tracking IDs

✅ Lightweight and fast

🧾 Project Structure

vehicle-plate-detection-tracking/
│
├── main.py                         # Main entry point

├── detect_license_plate.engine     # TensorRT engine for detection

├── yolov8s.engine                  # Another custom engine (optional

├── sort/                           # SORT tracking module

├── video_test.mp4                  # Input test video

├── video2.mp4                      # Optional input video

├── result.mp4                      # Output video with annotations

└── README.md                       # This file
🚀 Quick Start
1️⃣ Clone repo
bash
git clone https://github.com/Sinbad-04/vehicle-plate-detection-tracking.git
cd vehicle-plate-detection-tracking
2️⃣ Setup environment
Create virtual environment (recommended):

````
python -m venv venv
````
source venv/bin/activate      # Windows: venv\Scripts\activate
````
pip install -r requirements.txt
````
Dependencies include:
torch, opencv-python, numpy, filterpy, tensorrt, ultralytics (if training YOLOv8)

🎥 Running Inference

````
python main.py --video video_test.mp4 --engine detect_license_plate.engine --output result.mp4
````
✅ Arguments
Argument	Description
--video	Path to input video file
--engine	Path to YOLOv8 TensorRT engine (.engine file)
--output	Output video path with bounding boxes & IDs

📊 Output Example
The model detects license plates in real time and tracks them across frames using SORT. Below is a sample frame from the result video:

<p align="center"> <img src="https://github.com/Sinbad-04/vehicle-plate-detection-tracking/blob/main/frame_sample.png" width="640"/> </p>
✅ You can find the full demo video in the repository: result.mp4

🧠 How it Works
YOLOv8 Detection

Custom YOLOv8 model trained on license plate dataset

Exported as .engine using TensorRT for fast inference

SORT Tracking

Uses Kalman filter + Hungarian Algorithm

Simple and lightweight multi-object tracking algorithm

Video Processing

Frame-by-frame: detect → track → draw boxes → write frame

📈 Training (optional)
If you want to train your own YOLOv8 model:


````
pip install ultralytics
````
````
yolo task=detect mode=train model=yolov8s.pt data=your_dataset.yaml epochs=100 imgsz=640
````
Then convert to TensorRT engine if needed.

💡 Future Improvements
🔠 Integrate license plate OCR (e.g., EasyOCR or CRNN)

🌐 Deploy on edge devices using ONNX or Jetson

🧪 Add evaluation metrics: detection accuracy, FPS

🌈 Add real-time visualization using Streamlit or Gradio

📄 License
This project is open-source under the MIT License.
Feel free to use, modify, or distribute.

👤 Author
Sinbad-04
🔗 GitHub: Sinbad-04

👉 If you find this project helpful, give it a star ⭐ and feel free to contribute!





