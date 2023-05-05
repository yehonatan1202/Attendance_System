import cv2
from utils import load_model, get_vector, compare, preprocesss, load_vector_class, create_embedding_model, create_distance_model
# Load the pre-trained Haar Cascade face detection model
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# vector_class = load_vector_class('10 people')
model = load_model('siamesemodelv5.h5')
# Start capturing video from the default webcam
cap = cv2.VideoCapture(0)

# Loop indefinitely
while True:
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
    for (x, y, w, h) in faces_cords:
        face = frame[y:y+h, x:x+w]
        cv2.imwrite('face.jpg', face)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        res = compare(
            model, 'face.jpg', 'C:\\Users\\admin\\Downloads\\faces\\0\\Colin_Powell_0001.jpg')
        if res != -1:
            frame = cv2.putText(frame, f"Face: {res}", (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (255, 0, 0), 2, cv2.LINE_AA)
    # Display the resulting frame
    cv2.imshow('Face Detection', frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
