import os
import numpy as np
import cv2
import mediapipe as mp
from itertools import product
from my_functions import image_process, draw_landmarks, keypoint_extraction
import keyboard

# Define the actions (signs) that will be recorded and stored in the dataset
# actions = np.array(['no', 'yes', 'i love you', 'hello'])
actions = np.array(["i love you","hello",' name',"what","thank you","is","no",'your',"yes","my","m","n","a","e",])

# Define the number of sequences and frames to be recorded for each action
sequences = 30
frames = 10

# Set the path where the dataset will be stored
PATH = os.path.join('data')

# Create directories for each action, sequence, and frame in the dataset
for action, sequence in product(actions, range(sequences)):
    try:
        os.makedirs(os.path.join(PATH, action, str(sequence)))
    except:
        pass

# Access the camera and check if the camera is opened successfully
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot access camera.")
    exit()

# Create a MediaPipe Holistic object for hand tracking and landmark extraction
with mp.solutions.holistic.Holistic(min_detection_confidence=0.75, min_tracking_confidence=0.75) as holistic:
    # Loop through each action, sequence, and frame to record data
    for action, sequence in product(actions, range(sequences)):
        # Wait for the spacebar key press to start recording for each sequence
        print(f"Get ready to record '{action}' sequence {sequence}. Press 'Space' when you are ready.")
        while True:
            if keyboard.is_pressed(' '):
                break
            ret, image = cap.read()
            if not ret:
                print("Failed to grab frame")
                continue

            # Make the image writable
            image.flags.writeable = True

            # Process the image and draw landmarks
            results = image_process(image, holistic)
            draw_landmarks(image, results)

            # Provide feedback to the user
            image_copy = np.copy(image)  # Ensure the copy is writable
            cv2.putText(image_copy, 'Pause.', (20, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            cv2.putText(image_copy, 'Press "Space" when you are ready.', (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            cv2.imshow('Camera', image_copy)
            cv2.waitKey(1)

            # Check if the 'Camera' window was closed and break the loop
            if cv2.getWindowProperty('Camera', cv2.WND_PROP_VISIBLE) < 1:
                break

        # Countdown before starting the recording
        for i in range(3, 0, -1):
            ret, image = cap.read()
            if not ret:
                print("Failed to grab frame")
                continue
            image.flags.writeable = True  # Make the image writable
            image_copy = np.copy(image)  # Ensure the copy is writable
            cv2.putText(image_copy, f'Starting in {i}...', (20, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow('Camera', image_copy)
            cv2.waitKey(1000)

        # Record frames for the current sequence
        for frame in range(frames):
            ret, image = cap.read()
            if not ret:
                print("Failed to grab frame")
                continue
            image.flags.writeable = True  # Ensure the image is writable
            results = image_process(image, holistic)
            draw_landmarks(image, results)

            # Provide feedback during recording
            image_copy = np.copy(image)  # Ensure the copy is writable
            cv2.putText(image_copy, f'Recording "{action}" Sequence {sequence} Frame {frame}', (20, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

            cv2.imshow('Camera', image_copy)
            cv2.waitKey(1)

            # Check if the 'Camera' window was closed and break the loop
            if cv2.getWindowProperty('Camera', cv2.WND_PROP_VISIBLE) < 1:
                break

            # Ensure results contain valid landmarks
            if results.left_hand_landmarks or results.right_hand_landmarks:
                keypoints = keypoint_extraction(results)
                frame_path = os.path.join(PATH, action, str(sequence), str(frame))
                np.save(frame_path, keypoints)

    # Release the camera and close any remaining windows
    cap.release()
    cv2.destroyAllWindows()
