# **1. Download dataset**
from google.colab import drive

drive.mount('/content/gdrive')
# https://drive.google.com/file/d/1--0QuKMwj31K-CSvD8oq5fceFweiFPuN/view?usp=share_link
!gdown https://drive.google.com/u/0/uc?id=1--0QuKMwj31K-CSvD8oq5fceFweiFPuN&export=download
!unzip /content/ultralytics/human_detection_dataset.zip
# **2. Install YOLOv8**
!git clone https://github.com/ultralytics/ultralytics
%cd ultralytics
!pip install ultralytics
import ultralytics

ultralytics.checks()
%cd ultralytics
!pip install -e .
# **3. Download pretrained model**
!wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt
# **4. Training**
!yolo train model=yolov8s.pt data=./human_detection_dataset/data.yaml epochs=20 imgsz=640
# **5. Validating**
!yolo val model=./runs/detect/train2/weights/best.pt data=../human_detection_dataset/data.yaml
# **6. Predict**
from google.colab import files

uploaded = files.upload()
filename = next(iter(uploaded))

print(f"Uploaded file: {filename}")
# With uploaded image
!yolo predict model=./runs/detect/train/weights/best.pt \
    source='/content/ultralytics/frame007.25.00-07.30.00.jpg'
# With online image
# https://c.files.bbci.co.uk/1260/production/_108240740_beatles-abbeyroad-index-reuters-applecorps.jpg
!yolo predict model=./runs/detect/train/weights/best.pt source='https://assets.weforum.org/article/image/XaHpf_z51huQS_JPHs-jkPhBp0dLlxFJwt-sPLpGJB0.jpg'
# With youtube video
!yolo predict model=./runs/detect/train/weights/best.pt source='https://youtu.be/MsXdUtlDVhk'
# **7. Export model (Optional)**
# Convert weight file to other formats
!yolo export model=./runs/detect/train/weights/best.pt format=onnx
!cp '/content/ultralytics/runs/detect/train/weights/best.onnx' '/content/gdrive/MyDrive/Coordinate/aio_2023_ta/module1/yolov8_project/solution/weights'
!cp '/content/ultralytics/runs/detect/train/weights/best.pt' '/content/gdrive/MyDrive/Coordinate/aio_2023_ta/module1/yolov8_project/solution/weights'# ML-Projects