import cv2
import torch
from PIL import Image
import numpy as np
from near import depth_estimator, model as near_model, normalize_depth, smooth_depth, check_proximity, object_thresholds
from vision import model as vision_model, get_compact_directions
import math
import warnings
import pyttsx3
import threading

warnings.filterwarnings("ignore")

# Initialize text-to-speech engine
engine = pyttsx3.init()

def speak(text):
    threading.Thread(target=engine.say, args=(text,)).start()
    engine.runAndWait()

# Main video processing loop
video_path = '/Users/reetvikchatterjee/Desktop/VisionHelp/testcouch.mp4'  # Replace with your video path
cap = cv2.VideoCapture(video_path)

def get_threshold(object_name, default_threshold=20):
    return object_thresholds.get(object_name, default_threshold)

# Frame skip settings
FRAME_SKIP = 3  # Process every 3rd frame
frame_count = 0

# Detection confidence threshold
DETECTION_THRESHOLD = 0.7  # 70% confidence threshold

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % FRAME_SKIP != 0:
        continue  # Skip this frame

    # Process frame for nearby object detection
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    depth_map = depth_estimator(image)["depth"]
    depth_array = np.array(depth_map)
    depth_array = normalize_depth(depth_array)
    depth_array = smooth_depth(depth_array)

    results = near_model(frame)
    close_objects, all_objects = check_proximity(depth_array, results, get_threshold)

    # Display results on frame
    for obj in results.xyxy[0]:
        x1, y1, x2, y2, conf, cls = obj.tolist()
        if conf < DETECTION_THRESHOLD:
            continue  # Skip objects below the confidence threshold
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cls = int(cls)
        object_name = results.names[cls]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{object_name}: {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    if close_objects:
        warning_message = f"Warning: {', '.join(set(close_objects))} nearby!"
        cv2.putText(frame, warning_message, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        print(warning_message)  # Print the warning message
        speak(warning_message)  # Speak the warning message

    cv2.imshow('Video', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        # Process current frame for object directions
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        results = vision_model(image)
        detections = results.pandas().xyxy[0]
        image_width, image_height = image.size
        object_positions = {}

        for _, obj in detections.iterrows():
            if obj['confidence'] < DETECTION_THRESHOLD:
                continue  # Skip objects below the confidence threshold
            obj_name = obj['name']
            xmin, ymin, xmax, ymax = obj[['xmin', 'ymin', 'xmax', 'ymax']]
            center_x = (xmin + xmax) / 2
            center_y = (ymin + ymax) / 2
            if obj_name in object_positions:
                object_positions[obj_name].append((center_x, center_y))
            else:
                object_positions[obj_name] = [(center_x, center_y)]

        detected_objects = ", ".join(object_positions.keys())
        print("\nDetected objects:", detected_objects)
        speak(f"Detected objects are {detected_objects}")

        while True:
            user_input = input("What would you like to find? (or 'back' to return to video) ").strip().lower()
            if user_input == 'back':
                break
            if user_input in object_positions:
                start_x, start_y = image_width / 2, image_height
                nearest_obj = min(object_positions[user_input], key=lambda pos: math.sqrt((pos[0] - start_x)**2 + (pos[1] - start_y)**2))
                directions = get_compact_directions(start_x, start_y, nearest_obj[0], nearest_obj[1])
                print(f"Directions to the nearest {user_input}:")
                print(directions)
                speak(f"Directions to the nearest {user_input}. {directions}")
            else:
                not_found_message = f"{user_input.capitalize()} not found."
                print(not_found_message)
                speak(not_found_message)

cap.release()
cv2.destroyAllWindows()
