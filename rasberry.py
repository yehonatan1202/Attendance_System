import cv2
from client import *
import RPi.GPIO as GPIO
from mfrc522 import SimpleMFRC522


# Load the pre-trained Haar Cascade face detection model
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# Init RFID Reader
reader = SimpleMFRC522()
# Camera

def scan_card():
    try:
        id = reader.read_id()
        print(id)
    finally:
        GPIO.cleanup()
    return id

def take_face_photo(name):
    print('photo')
    loop = True
    # Start capturing video from the default webcam
    cap = cv2.VideoCapture(0)
    print('loop')
    # Loop indefinitely
    while loop:
        # Read a frame from the video stream
        ret, frame = cap.read()
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect faces in the grayscale frame
        faces_cords = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)
        # list of croped faces in the frame
        faces = []
        # Draw rectangles around each face
        found = False
        for (x, y, w, h) in faces_cords:
            face = frame[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (100, 100), interpolation = cv2.INTER_AREA)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            found = True
        # Display the resulting frame
        cv2.putText(frame, name, (0, 30), cv2.FONT_HERSHEY_SIMPLEX,1,(255, 0, 0), 2, cv2.LINE_AA)
        cv2.imshow('Face Detection', frame)
        # Exit the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        if found:
            return face_resized
#             cv2.imwrite('photo.jpg', face)
#             return cv2.imread('photo.jpg')


def scan(id):
    #scan and send id
    name = client.send_string(str(id))
    if name == False:
        print('id not found')
        return False
    print(name)
    photo = take_face_photo(name)
    res = client.send_photo(photo)
    while(res == 2):
        photo = take_face_photo(name)
        res = client.send_photo(photo)
    if res == 1:
        return True
    else:
        return False

while True:
    print('place RFID device')
    id = scan_card()
    client = Client('192.168.0.249', 6969)
    client.connect()
    print('connect')
    print(scan(id))
    client.close_connection()
    # Release the video capture object and close all windows
    cv2.destroyAllWindows()
