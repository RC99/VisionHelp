import torch
from PIL import Image, ImageDraw, ImageFont
import math

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
object_positions = {}

# Create a draw object for annotating the image
draw = ImageDraw.Draw(image)
font = ImageFont.load_default()  # Use a default font

for _, obj in detections.iterrows():
    obj_name = obj['name']
    xmin, ymin, xmax, ymax = obj[['xmin', 'ymin', 'xmax', 'ymax']]
    center_x = (xmin + xmax) / 2
    center_y = (ymin + ymax) / 2

    if obj_name in object_positions:
        object_positions[obj_name].append((center_x, center_y))
    else:
        object_positions[obj_name] = [(center_x, center_y)]

    # Annotate the image
    draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=3)
    draw.text((xmin, ymin), obj_name, fill="white", font=font)

# Display the annotated image
image.show()

def get_compact_directions(start_x, start_y, end_x, end_y):
    dx = end_x - start_x
    dy = start_y - end_y  # Reverse y-axis for intuitive directions
    distance = math.sqrt(dx**2 + dy**2)
    
    step_size = 50  # pixels per step
    total_steps = int(distance / step_size)
    
    if total_steps == 0:
        return "The object is right in front of you."
    
    # Determine primary direction
    if abs(dx) > abs(dy):
        primary_direction = "right" if dx > 0 else "left"
        primary_steps = int(abs(dx) / step_size)
        secondary_direction = "straight" if dy > 0 else "back"
        secondary_steps = int(abs(dy) / step_size)
    else:
        primary_direction = "straight" if dy > 0 else "back"
        primary_steps = int(abs(dy) / step_size)
        secondary_direction = "right" if dx > 0 else "left"
        secondary_steps = int(abs(dx) / step_size)
    
    directions = f"Take {primary_steps} steps {primary_direction}"
    if secondary_steps > 0:
        directions += f", then {secondary_steps} steps {secondary_direction}"
    
    # Add uphill/downhill information
    angle = math.degrees(math.atan2(dy, dx))
    if abs(angle) > 15 and abs(angle) < 75:
        if dy > 0:
            directions += ". You'll be moving slightly uphill"
        else:
            directions += ". You'll be moving slightly downhill"
    
    return directions

# Respond to user input
while True:
    user_input = input("What would you like to find? (or 'exit' to quit) ").strip().lower()

    if user_input == 'exit':
        break

    if user_input in object_positions:
        print(f"Directions to the nearest {user_input}:")
        
        # Find the nearest object of the requested type
        start_x, start_y = image_width / 2, image_height  # Starting from bottom center
        nearest_obj = min(object_positions[user_input], key=lambda pos: math.sqrt((pos[0] - start_x)**2 + (pos[1] - start_y)**2))
        
        directions = get_compact_directions(start_x, start_y, nearest_obj[0], nearest_obj[1])
        print(directions)
    else:
        detected_objects = ", ".join(object_positions.keys())
        if detected_objects:
            print(f"{user_input.capitalize()} not found. Detected objects: {detected_objects}.")
        else:
            print("No objects were detected in the image.")
