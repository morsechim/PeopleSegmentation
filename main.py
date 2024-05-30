import os
import cv2
import torch
import numpy as np
import ultralytics
ultralytics.checks()
from ultralytics import YOLO
import supervision as sv


# define config
conf_threshold = 0.5

# define video path
video_path = os.path.join(".", "data", "person.mp4")
output_path = os.path.join(".", "data", "output.mp4")

# define video capture instance
cap = cv2.VideoCapture(video_path)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# define video output instance
cap_out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"MP4V"), cap.get(cv2.CAP_PROP_FPS), (frame_width, frame_height))

# define model path
model_path = os.path.join(".", "weights", "yolov8n-seg.pt")
# define processor device
device = "mps:0" if torch.backends.mps.is_available() else "cpu"
print(f"running by {device} device")
# define yolo model
model = YOLO(model_path).to(device)
model.fuse()

# define window name
cv2.namedWindow("Frame")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    
    # get detect results
    results = model(frame, imgsz=1280)[0]
    detections = sv.Detections.from_ultralytics(results)
    
    # filter
    selected_classes = [0] # person
    detections = detections[np.isin(detections.class_id, selected_classes)]
   
    # define mask annotator
    mask_annotator = sv.MaskAnnotator(color=sv.Color(255, 0, 255))
    # define label annotator
    label_annotator = sv.LabelAnnotator(text_position=sv.Position.TOP_LEFT, color=sv.Color(255, 0, 255), text_thickness=1)
    
    # custom label annotate
    labels = [
        f"{class_name} {confidence:.2f}"
        for class_name, confidence
        in zip(detections['class_name'], detections.confidence)
    ]
    
    # draw segment
    annotated_frame = mask_annotator.annotate(
        scene=frame, detections=detections)
    
    # draw label
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame, detections=detections, labels=labels)
    
    cv2.imshow("Frame", annotated_frame)
    cap_out.write(annotated_frame)
    if cv2.waitKey(1) == ord("q"):
        break
    
cap.release()
cap_out.release()
cv2.destroyAllWindows()