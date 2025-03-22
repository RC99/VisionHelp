import cv2
import pyttsx3
import requests
from geopy.geocoders import Nominatim
import re

def get_current_location():
    geolocator = Nominatim(user_agent="my_app")
    location = geolocator.geocode("Los Angeles")
    return location.latitude, location.longitude

def get_destination_coordinates(destination_name):
    geolocator = Nominatim(user_agent="my_app")
    location = geolocator.geocode(destination_name)
    return location.latitude, location.longitude

def get_directions(start_lat, start_lon, end_lat, end_lon):
    api_key = ""
    url = f"https://maps.googleapis.com/maps/api/directions/json?origin={start_lat},{start_lon}&destination={end_lat},{end_lon}&key={api_key}"
    response = requests.get(url)
    data = response.json()
    return data['routes'][0]['legs'][0]['steps']

def speak_directions(directions):
    engine = pyttsx3.init()
    for step in directions:
        instruction = step['html_instructions']
        instruction = re.sub('<.*?>', '', instruction)
        engine.say(instruction)
        engine.runAndWait()

# Main loop
cap = cv2.VideoCapture('testcouch.mp4')
current_lat, current_lon = get_current_location()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break


    key = cv2.waitKey(1) & 0xFF
    if key == ord('n'):  
        destination_name = input("Enter destination name: ")
        destination_lat, destination_lon = get_destination_coordinates(destination_name)
        
        directions = get_directions(current_lat, current_lon, destination_lat, destination_lon)
        speak_directions(directions)

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
