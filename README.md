# House Intrusion Detection System (Streamlit App)

[View App](https://a-house-intrusion-detection-app.streamlit.app)

This Streamlit app uses a YOLOv8 model to detect people, animals, and vehicles from webcam snapshots. It can alert on detection via display and optionally by email.

## Objective

To monitor for intrusions (human or animal) using YOLOv8 with a user-friendly webcam interface.

## Workflow

- Take photo using webcam
- Use YOLOv8 model to detect objects
- If a person is detected, optionally send an email alert (disabled in code)
- Display annotated image and detected labels

## Features

- Person/animal/vehicle detection
- Visual display of bounding boxes
- Chime/alert placeholder for real-time response
- Optional email alert system (commented out)

## Dependencies

- `streamlit`
- `opencv-python`
- `ultralytics`
- `dotenv` (for email config)

## Output

- Detected objects in webcam frame
- Labeled images with bounding boxes
# House Intrusion Detection System (Streamlit App)

This Streamlit app uses a YOLOv8 model to detect people, animals, and vehicles from webcam snapshots. It can alert on detection via display and optionally by email.

## Objective

To monitor for intrusions (human or animal) using YOLOv8 with a user-friendly webcam interface.

## Workflow

- Take photo using webcam
- Use YOLOv8 model to detect objects
- If a person is detected, optionally send an email alert (disabled in code)
- Display annotated image and detected labels

## Features

- Person/animal/vehicle detection
- Visual display of bounding boxes
- Chime/alert placeholder for real-time response
- Optional email alert system (commented out)

## Dependencies

- `streamlit`
- `opencv-python`
- `ultralytics`
- `dotenv` (for email config)

## Output

- Detected objects in webcam frame
- Labeled images with bounding boxes

