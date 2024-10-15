







import cv2
import os
import numpy as np


input_video = '/Users/zhongz2/Desktop/learn/mp4/DEA-C01-test1.mp4'
output_folder = os.path.splitext(input_video)[0]
os.makedirs(output_folder, exist_ok=True)

# Open the video file
video_capture = cv2.VideoCapture(input_video)
success, frame = video_capture.read()
old_frame = None
count = 0
diff = 0

# Read each frame and save it as an image
while success:
    if count == 0 or diff > 1e8:
        image_path = os.path.join(output_folder, f"frame_{count:09d}.jpg")  # Adjust the format as per your requirement
        cv2.imwrite(image_path, frame)  # Save the frame as an image
        old_frame = frame
    success, frame = video_capture.read()  # Read next frame
    count += 1
    diff = np.sum(np.abs(frame - old_frame))
    print(count, diff)

# Release the video capture object
video_capture.release()











