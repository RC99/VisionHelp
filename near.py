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

def normalize_depth(depth_array):
    return cv2.normalize(depth_array, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def smooth_depth(depth_array):
    return cv2.GaussianBlur(depth_array, (5, 5), 0)

def adaptive_threshold(depth_array):
    mean_depth = np.mean(depth_array)
    std_depth = np.std(depth_array)
    return mean_depth - std_depth

# Normalize and smooth depth array
depth_array = normalize_depth(depth_array)
depth_array = smooth_depth(depth_array)

# Define object-specific thresholds (in normalized depth units)
object_thresholds = {
    'person': 30,
    'chair': 15,
    'table': 25,
    'car': 40,
    'bicycle': 20,
    'motorcycle': 35,
    'bus': 50,
    'truck': 45,
    'dog': 10,
    'cat': 10,
    'bottle': 5,
    'laptop': 10,
    'tv': 20,
    'couch': 30,
    'bed': 35,
    'refrigerator': 25,
    'book': 5,
    'clock': 10,
    'vase': 8,
    'potted plant': 12
}


def check_proximity(depth_array, results, default_threshold):
    close_objects = []
    all_objects = []
    adaptive_thresh = adaptive_threshold(depth_array)
    
    for det in results.xyxy[0]:
        x1, y1, x2, y2, conf, cls = det.tolist()
        if conf < 0.5:  # Confidence filtering
            continue
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        object_name = results.names[int(cls)]
        object_depth = np.mean(depth_array[y1:y2, x1:x2])
        
        all_objects.append(f"{object_name} (depth: {object_depth:.2f})")
        
        threshold = object_thresholds.get(object_name, default_threshold)
        if object_depth < min(threshold, adaptive_thresh):
            close_objects.append(object_name)
    
    return close_objects, all_objects

# Get warning message and all objects
close_objects, all_objects = check_proximity(depth_array, results, 20)

print("All detected objects:")
for obj in all_objects:
    print(f"- {obj}")

if close_objects:
    print(f"\nWarning: {', '.join(set(close_objects))} {'is' if len(set(close_objects)) == 1 else 'are'} very close!")
else:
    print("\nAll objects are at a safe distance.")

# Visualize results
img = results.render()[0]
cv2.imshow('Object Detection', img[:, :, ::-1])
cv2.waitKey(0)
cv2.destroyAllWindows()
