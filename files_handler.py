import os
import cv2


# Set paths
path = 'C:\\Users\\admin\\Downloads\\lfw'
new_path = 'C:\\Users\\admin\\Downloads\\new_faces'
cropped_path = 'C:\\Users\\admin\\Downloads\\new_faces_cropped'

# List to store folders with more than 3 photos
list = []

# Cascade classifier for face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Get a list of every folder with more than 3 photos and sort it
def get_folders_with_more_than_3_photos():
    for folder in os.listdir(path):
        sum = 0
        for file in os.listdir(os.path.join(path, folder)):
            sum += 1
        if (sum > 3):
            list.append((folder, sum))
    list.sort(key=lambda x: x[1])
    print(len(list))

# Moves all of the folders and photos from the list
def move_folders_and_photos():
    if not os.path.exists(new_path):
        os.makedirs(new_path)

    for folder in list:
        if not os.path.exists(os.path.join(new_path, folder[0])):
            os.makedirs(os.path.join(new_path, folder[0]))
        for file in os.listdir(os.path.join(path, folder[0])):
            os.rename(os.path.join(path, folder[0], file), os.path.join(
                new_path, folder[0], file))

# Crop all of the photos around the face and saves them
def crop_photos_around_face():
    if not os.path.exists(cropped_path):
        os.makedirs(cropped_path)

    for folder in os.listdir(new_path):
        if not os.path.exists(os.path.join(cropped_path, folder)):
            os.makedirs(os.path.join(cropped_path, folder))
        for file in os.listdir(os.path.join(new_path, folder)):
            img = cv2.imread(os.path.join(new_path, folder, file))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces_cords = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(32, 32), flags=cv2.CASCADE_SCALE_IMAGE)
            if len(faces_cords) > 0 and len(faces_cords) <= 1:
                for (x, y, w, h) in faces_cords:
                    cv2.imwrite(os.path.join(cropped_path, folder, file),img[y:y+h, x:x+w])

# Counts the number of folders with more than 9 photos
def count_folders_with_more_than_9_photos():
    count = 0
    for folder in os.listdir(cropped_path):
        sum = 0
        for file in os.listdir(os.path.join(cropped_path, folder)):
            sum += 1
        if sum > 9:
            count += 1
    print(count)