import cv2
import torch
import numpy as np


# Open the video file
video_path = "./data/20240219T1611_d2.0_90ml_250fps.mp4"  # Replace with your video file path
cap = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

i = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = frame[150:1400, 400:600]

    filename = f"./data/images/img{i:06}.tiff"
    cv2.imwrite(filename, frame)
    print(i)
    i = i + 1


# Release resources
cap.release()
cv2.destroyAllWindows()
