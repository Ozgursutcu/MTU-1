import cv2
import os
import tensorflow as tf
import numpy as np
from keras.utils import img_to_array, load_img
import firebase_admin
from firebase_admin import credentials
from firebase_admin import storage


# Initialize Firebase
cred = credentials.Certificate("C:/Users/ozgur/Desktop/new/admin_firebase.json")
firebase_admin.initialize_app(cred, {
    'storageBucket': 'mtuodevi.appspot.com'
})

bucket = storage.bucket()


font = cv2.FONT_HERSHEY_COMPLEX

# Load the model
model = tf.keras.models.load_model("C:/Users/ozgur/Desktop/new/fruits_new_model.h5")
print(model.summary())

# Load the categories
source_folder = "C:/Users/ozgur/Desktop/new/z_test/"
categories = os.listdir(source_folder)
categories.sort()
print(categories)
numofClasses = len(categories)
print(numofClasses)

# Load and prepare image
def prepareImage(image):
    image = cv2.resize(image, (100, 100))
    imgResult = img_to_array(image)
    imgResult = np.expand_dims(imgResult, axis=0)
    imgResult = imgResult / 255.
    return imgResult

# Capture video from camera
cap = cv2.VideoCapture(1)
ret, frame = cap.read()

# Set up video writer
output_file = "captured_video.mp4"
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = 20 # Video FPS (frames per second)
out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frame_width, frame_height))

# Start capturing and recording
start_time = cv2.getTickCount()  # Başlangıç zamanını al

fruit_boxes = []  # Meyve bölgelerini tutmak için liste

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Save the frame to video
    out.write(frame)

    # Show the captured frame with text
    imageForModel = prepareImage(frame)
    resultArray = model.predict(imageForModel, verbose=1)
    answers = np.argmax(resultArray, axis=1)
    predicted_class = categories[answers[0]]
    cv2.putText(frame, predicted_class, (0, 50), font, 1, (209, 19, 77), 1)
    cv2.imshow('image', frame)

    # Add the text overlay to the frame
    frame_with_text = frame.copy()
    cv2.putText(frame_with_text, predicted_class, (0, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (209, 19, 77), 1)

    # Perform object detection on the frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, threshold = cv2.threshold(blur, 50, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    fruit_boxes = []  # Her karede meyve bölgelerini sıfırla

    # Draw bounding boxes around the fruits
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 1000:
            x, y, w, h = cv2.boundingRect(contour)

            cv2.rectangle(frame_with_text, (x, y), (x+w, y+h), (0, 255, 0), 2)
            fruit_boxes.append((x, y, x+w, y+h))  # Meyve bölgelerini listeye ekle

            # Crop the region of interest (fruit) from the frame
            fruit = frame[y:y+h, x:x+w]

            # Perform fruit classification on the cropped image
            imageForModel = prepareImage(fruit)
            resultArray = model.predict(imageForModel, verbose=1)
            answers = np.argmax(resultArray, axis=1)
            predicted_class = categories[answers[0]]

            # Draw bounding box and label for the fruit
            cv2.rectangle(frame_with_text, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame_with_text, predicted_class, (x, y-10), cv2.FONT_HERSHEY_COMPLEX, 0.9, (0, 255, 0), 2)

# Count the number of fruits
    fruit_count = len(fruit_boxes)

    # Show the fruit count on the frame
    cv2.putText(frame_with_text, "Fruits: " + str(fruit_count), (0, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (209, 19, 77), 1)
    # Save the frame with text and bounding boxes to video
    out.write(frame_with_text)

    # Exit the loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Exit the loop if 10 seconds have passed
    elapsed_time = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
    if elapsed_time >= 10:
        break

# Release the capture and video writer
cap.release()
out.release()
cv2.destroyAllWindows()

print("Video saved as", output_file)

# Upload video to Firebase Storageqq
video_path = "captured_video.mp4"
destination_blob_name = "videos/captured_video.mp4"


blob = bucket.blob(destination_blob_name)
blob.upload_from_filename(video_path)

print("Video uploaded to Firebase Storage")