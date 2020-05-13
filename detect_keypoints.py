import cv2
import numpy as np
from model import Model_3

model = Model_3()
model.load_weights('model_3_best.hdf5')
print ("Model Loaded...")

# Get a reference to webcam 
video_capture = cv2.VideoCapture(0)
print ("Camera Started...")

while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()

    # Convert the image from BGR color (which OpenCV uses) to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Resize the image array to 1,96,96,1
    gray = cv2.resize(gray, (96,96))
    gray = np.array(gray)
    gray_scale = gray.reshape((1,96,96,1))

    # Find all the keypoints in the current frame
    keypoints = model.predict(gray_scale).T

    # Display the results
    for i in range(0,len(keypoints),2):
        # Draw a box around the face
        center_x = keypoints[i] * 90 + 360
        center_y = keypoints[i+1] * 150 + 260
        
        #center_x = keypoints[i] * 48 + 48
        #center_y = keypoints[i+1] * 48 + 48
        
        cv2.circle(frame, (center_x,center_y), 1, color=(255,0,0), thickness=-1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
