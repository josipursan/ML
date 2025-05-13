import tensorflow as tf
import tensorflow_hub as tf_hub
import cv2
from matplotlib import pyplot as plt
import numpy as np

import kagglehub    # This is instead of tensorflow_hub as it seems that everything from tensorflow_hub has moved to kaggle

# Load the model
""" modelPath = kagglehub.model_download("google/movenet/tensorFlow2/multipose-lightning")
print("modelPath : {}\n".format(modelPath))

interpreter = tf.lite.Interpreter(model_path=modelPath)
interpreter.allocate_tensors()
 """

model = tf_hub.load('https://tfhub.dev/google/movenet/multipose/lightning/1')
movenet = model.signatures['serving_default']

# Let's try accessing our webcam
cap = cv2.VideoCapture(0)   # Connects to webcam - the 2 in parentheses refers to which capture device we are using (because one computer can have multiple simultanously connected capture devices)
while cap.isOpened():   # Infinite loop until exit conditions are met
    ret, frame = cap.read() # cap.read() is used to stream, or capture, feed from the camera

    cv2.imshow("Multiperson HPE", frame)    # This simply shows the picture using cv2 lib

    if cv2.waitKey(10) & 0xFF==ord('q'):    # Exit conditions - how to exit the capture cleanly - 10 refers to period used to check if q is pressed (10 ms)
        break
cap.release()   # Exit cleanup
cv2.destroyAllWindows() # Exit cleanup