import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Function to load and preprocess images
def load_and_preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0  # Normalize pixel values to be between 0 and 1
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Step 1: Load the images and their corresponding labels
dataset_path = "Images"
known_people = {}

for person_folder in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person_folder)

    if os.path.isdir(person_path):
        known_faces = []
        known_names = [person_folder]  # Use folder name as the label

        for filename in os.listdir(person_path):
            if filename.endswith(".jpg"):
                image_path = os.path.join(person_path, filename)
                known_faces.append(load_and_preprocess_image(image_path))

        known_people[person_folder] = {'faces': known_faces, 'names': known_names}

# Determine the number of classes (individuals) in your dataset
num_classes = len(known_people)

# Load the pre-trained MobileNetV2 model for feature extraction
base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add a global average pooling layer and a dense layer for classification
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='softmax')  # num_classes is the number of individuals you want to recognize
])

# Load the trained face recognition model
model.load_weights('face_recognition_model.h5')

# Initialize the camera
cap = cv2.VideoCapture(0)

# Set camera resolution
cap.set(3, 1280)  # Width
cap.set(4, 720)   # Height

while True:
    # Capture frames from the camera
    ret, frame = cap.read()

    # Find faces in the frame
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Extract and preprocess the face region
        face_region = frame[y:y+h, x:x+w]
        face_image = cv2.resize(face_region, (224, 224))
        face_image = face_image / 255.0  # Normalize pixel values to be between 0 and 1
        face_image = np.expand_dims(face_image, axis=0)  # Add batch dimension

        # Predict the person using the trained model
        predictions = model.predict(face_image)
        predicted_class = np.argmax(predictions)

        # Display the results on the frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_DUPLEX

        if predicted_class < len(known_people):
            name = list(known_people.keys())[predicted_class]
            text = f"{name} ({predictions[0][predicted_class] * 100:.2f}%)"
        else:
            text = "Unknown"

        cv2.putText(frame, text, (x + 6, y - 6), font, 0.5, (255, 255, 255), 1)

    # Display the frame
    cv2.imshow("Face Recognition", frame)

    # Press 'q' to exit the program
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera
cap.release()
cv2.destroyAllWindows()
