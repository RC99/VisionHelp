import cv2
import torch
from PIL import Image
import numpy as np
from near import depth_estimator, model as near_model, normalize_depth, smooth_depth, check_proximity, object_thresholds
from vision import model as vision_model, get_compact_directions
import math
import warnings
import pyttsx3
import queue
from recognition import recognize_faces

warnings.filterwarnings("ignore")

# Initialize text-to-speech engine
engine = pyttsx3.init()
speech_queue = queue.Queue()

def speak(text):
    """Queue speech requests instead of calling runAndWait() in multiple threads."""
    speech_queue.put(text)

# Main video processing loop
video_path = '/Users/reetvikchatterjee/Desktop/VisionHelp/testfr.mp4'  # Replace with your video path
cap = cv2.VideoCapture(video_path)

def get_threshold(object_name, default_threshold=20):
    return object_thresholds.get(object_name, default_threshold)

FRAME_SKIP = 3  # Process every 3rd frame
frame_count = 0
DETECTION_THRESHOLD = 0.7  # 70% confidence threshold

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trained_model.yml")

relationships = {
    1: {"name": "Akshay Kumar", "relationships": "Friend"}
}

v_pressed = False

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
        print(warning_message)
        speak(warning_message)

    # Process frame for face recognition
    recognize = False
    if v_pressed:
        recognize = True
        v_pressed = False  # Reset v_pressed after use
    frame = recognize_faces(frame, speak, recognize)

    cv2.imshow('Video', frame)

    # Process speech queue in the main loop
    while not speech_queue.empty():
        try:
            text_to_speak = speech_queue.get_nowait()
            engine.say(text_to_speak)
            engine.runAndWait()
        except Exception as e:
            print(f"Speech Error: {e}")

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        results = vision_model(image)
        detections = results.pandas().xyxy[0]
        image_width, image_height = image.size
        object_positions = {}

        for _, obj in detections.iterrows():
            if obj['confidence'] < DETECTION_THRESHOLD:
                continue
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
    elif key == ord('v'):
        v_pressed = True
    elif key != ord('v'):
        pass

cap.release()
cv2.destroyAllWindows()
