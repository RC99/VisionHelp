import torch
from transformers import pipeline
from PIL import Image
import numpy as np
import cv2

# Load the Depth Anything V2 model
depth_estimator = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Load and preprocess the image
image_path = '/Users/reetvikchatterjee/Desktop/living-room-article-chair-22.jpg'
image = Image.open(image_path)
image_np = np.array(image)

# Perform depth estimation
depth_map = depth_estimator(image)["depth"]
depth_array = np.array(depth_map)

# Perform object detection with YOLOv5
results = model(image_np)

# Set proximity threshold (in relative depth units)
PROXIMITY_THRESHOLD = 20  # Adjust this value based on your needs

# Check for close objects
def check_proximity(depth_array, results, threshold):
    close_objects = []
    for det in results.xyxy[0]:  # det: (x1, y1, x2, y2, conf, cls)
        x1, y1, x2, y2, conf, cls = det.tolist()
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        object_depth = np.mean(depth_array[y1:y2, x1:x2])
        #print(f"Object: {results.names[int(cls)]}, Depth: {object_depth}")


        if object_depth < threshold:
            close_objects.append(results.names[int(cls)])
    
    if close_objects:
        return f"Warning: {', '.join(set(close_objects))} {'is' if len(set(close_objects)) >= 1 else 'are'} very close!"
    else:
        return "All objects are at a safe distance."

# Get warning message
warning_message = check_proximity(depth_array, results, PROXIMITY_THRESHOLD)
