import torch
from PIL import Image, ImageDraw, ImageFont

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Load an image
image_path = '/Users/reetvikchatterjee/Desktop/living-room-article-chair-22.jpg'
image = Image.open(image_path)

# Perform inference
results = model(image)

# Extract detected objects and their coordinates
detections = results.pandas().xyxy[0]  # Pandas DataFrame of detections

# Analyze positions and provide directions
image_width, image_height = image.size
object_directions = {}

# Create a draw object for annotating the image
draw = ImageDraw.Draw(image)
font = ImageFont.load_default()  # Use a default font

for _, obj in detections.iterrows():
    obj_name = obj['name']
    xmin, ymin, xmax, ymax = obj[['xmin', 'ymin', 'xmax', 'ymax']]
    center_x = (xmin + xmax) / 2
    center_y = (ymin + ymax) / 2

    # Determine horizontal position
    if center_x < image_width / 3:
        horizontal = "to your left"
    elif center_x > 2 * image_width / 3:
        horizontal = "to your right"
    else:
        horizontal = "straight ahead"

    # Determine vertical position
    if center_y < image_height / 2:
        vertical = "and slightly upward"
    else:
        vertical = "and slightly downward"

    direction = f"{horizontal} {vertical}"
    if obj_name in object_directions:
        object_directions[obj_name].append(direction)
    else:
        object_directions[obj_name] = [direction]

    # Annotate the image
    draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=3)
    draw.text((xmin, ymin), f"{obj_name}: {direction}", fill="white", font=font)

# Display the annotated image
image.show()

# Respond to user input
user_input = input("What would you like to find? ").strip().lower()

if user_input in object_directions:
    print(f"Directions to {user_input}:")
    print("\n".join(object_directions[user_input]))
else:
    detected_objects = ", ".join(object_directions.keys())
    if detected_objects:
        print(f"{user_input.capitalize()} not found. Detected objects: {detected_objects}.")
    else:
        print("No objects were detected in the image.")
