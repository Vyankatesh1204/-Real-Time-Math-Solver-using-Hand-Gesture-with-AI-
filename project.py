import cvzone
import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import google.generativeai as genai
from PIL import Image
import time

# Configure the Generative AI model
genai.configure(api_key="AIzaSyCpkQVWrn0N18qpHqOy7T1u3mhA_532i7M")  # Replace 'YOUR_API_KEY' with your actual API key
model = genai.GenerativeModel('gemini-1.5-flash')

# Initialize the webcam to capture video
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Initialize the HandDetector class with the given parameters
detector = HandDetector(staticMode=False, maxHands=2, modelComplexity=1, detectionCon=0.7, minTrackCon=0.5)

def getHandInfo(img):
    # Find hands in the current frame without drawing landmarks and outlines
    hands, img = detector.findHands(img, draw=False, flipType=True)

    # Check if any hands are detected
    if hands:
        # Information for the first hand detected
        hand1 = hands[0]  # Get the first hand detected
        lmList = hand1["lmList"]  # List of 21 landmarks for the first hand

        # Count the number of fingers up for the first hand
        fingers = detector.fingersUp(hand1)
        print(fingers)
        return fingers, lmList
    else:
        return None

def draw(info, prev_pos, canvas):
    fingers, lmList = info
    current_pos = None
    if fingers == [0, 1, 0, 0, 0]:
        current_pos = lmList[8][0:2]  # Tip of index finger
        if prev_pos is None:
            prev_pos = current_pos
        cv2.line(canvas, current_pos, prev_pos, (255, 0, 255), 10)
    elif fingers == [1, 1, 1, 1, 1]:  # Open palm clears the canvas
        canvas = np.zeros_like(canvas)

    return current_pos, canvas

def take_photo(fingers, img):

    if fingers == [1, 0, 0, 0, 0]:
        timestamp = int(time.time())
        filename = f"C:/Users/venka/OneDrive/Desktop/final year project/photo_{timestamp}.png"
        cv2.imwrite(filename, img)
        print(f"Photo saved as {filename}")

def sendToAI(model, canvas, fingers):
    if fingers == [1, 1, 1, 1, 0]:  # Trigger for AI with thumb down
        pil_image = Image.fromarray(canvas)
        response = model.generate_content(["Solve this math problem", pil_image])
        print(response.text)

prev_pos = None
canvas = None
image_combined = None

# Continuously get frames from the webcam
while True:
    # Capture each frame from the webcam
    success, img = cap.read()
    img = cv2.flip(img, 1)  # Flip the image horizontally

    if canvas is None:
        canvas = np.zeros_like(img)

    info = getHandInfo(img)

    if info:
        fingers, lmList = info
        print(fingers)
        prev_pos, canvas = draw(info, prev_pos, canvas)
        take_photo(fingers, img)
        sendToAI(model, canvas, fingers)

    # Combine the webcam feed with the drawing canvas
    image_combined = cv2.addWeighted(img, 0.6, canvas, 0.4, 0)

    # Display the image in a window
    cv2.imshow("Image", img)
    cv2.imshow("Canvas", canvas)
    cv2.imshow("Combined", image_combined)

    # Keep the window open and update it for each frame; wait for 1 millisecond between frames
    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit the application
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
