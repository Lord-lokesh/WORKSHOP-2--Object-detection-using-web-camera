# YOLOv8 Object Detection Using Laptop Camera

name : lokesh m

reg :212224230142

# Aim
To access the laptop camera, capture an image, and detect objects using YOLOv8.

# Requirements

Anaconda

Jupyter Notebook

Laptop Camera

# Steps

Open Jupyter Notebook using Anaconda.

Access your laptop camera using OpenCV.

Wait for 5 seconds and capture an image.

Display the captured image.

Apply YOLOv8 object detection.

Display the detected image with bounding boxes and labels.

# Algorithm
Laptop Camera
      ↓

Capture Image
      ↓

Display Image
      ↓

YOLOv8 Detection
      ↓

Display Detected Objects

# Student Task

Capture your own image using the laptop camera.

Detect the objects present in the image.

Display the final output.

Perform the experiment with 3 different scenes.
# Submission

Jupyter Notebook (.ipynb)

Original captured images

YOLOv8 output images

Screenshot of the final result

# GitHub Reference
https://github.com/ultralytics/ultralytics

Platform: Anaconda + Jupyter Notebook only.

# Program

```
import cv2

# Open the web camera
cap = cv2.VideoCapture(0)

# Background subtractor for foreground-object detection
bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=500,
    varThreshold=50,
    detectShadows=True
)

while True:

    # Read frame from webcam
    ret, frame = cap.read()

    if not ret:
        print("Cannot read frame from web camera")
        break

    # Create foreground mask
    mask = bg_subtractor.apply(frame)

    # Remove small noise
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (5, 5)
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_DILATE,
        kernel
    )

    # Find contours of detected objects
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # Process each detected contour
    for contour in contours:

        # Calculate contour area
        area = cv2.contourArea(contour)

        # Ignore very small regions
        if area > 2500:

            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)

            # Draw bounding box
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            # Display detection text
            cv2.putText(
                frame,
                "Object Detected",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

    # Display camera frame
    cv2.imshow(
        "Workshop 2 - Object Detection",
        frame
    )

    # Press 'q' to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()

```
<img width="803" height="601" alt="Screenshot 2026-09-02 125002" src="https://github.com/user-attachments/assets/7b8d829e-cdc7-4ea2-8b19-e68c6df5f866" />


# Result
The laptop camera was successfully accessed using OpenCV, and an image was captured after 5 seconds. The captured image was then processed using the YOLOv8 object detection model. YOLOv8 successfully detected the object present in the image and displayed it with a bounding box and label. Thus, the experiment successfully demonstrated object detection using a laptop camera and YOLOv8.

