import cv2
from client import *
import RPi.GPIO as GPIO
from mfrc522 import SimpleMFRC522
# Load the pre-trained Haar Cascade face detection model
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
reader = SimpleMFRC522()
print(cv2.__version__)
def scan_card():
    try:
        id = reader.read_id()
        print(id)
    finally:
        GPIO.cleanup()
    return id

def take_face_photo():
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
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
        # list of croped faces in the frame
        faces = []
        # Draw rectangles around each face
        # Display the resulting frame
        cv2.imshow('Face Detection', frame)
        for (x, y, w, h) in faces_cords:
            face = frame[y:y+h, x:x+w]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            return face
#             cv2.imwrite('photo.jpg', face)
#             return cv2.imread('photo.jpg')
        # Exit the loop if the 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture object and close all windows
    cap.release()
    cv2.destroyAllWindows()

def scan():
	#scan and send id
	id = scan_card()
	if client.send_string(str(id)) == False:
		print('id not found')
		return False
	
	photo = take_face_photo()
	tries = 10
	while(client.send_photo(photo) == False):
		if tries == 0:
			return False
		photo = take_face_photo()
		tries -= 1
	return True


client = Client('192.168.0.249', 6969)
client.connect()
print('connect')

while True:
	print(scan())