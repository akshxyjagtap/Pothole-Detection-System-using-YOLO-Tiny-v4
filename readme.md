[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://lbesson.mit-license.org/)
# Pothole Detection System using YOLOv4 Tiny

An intelligent Pothole Detection system using YOLOv4 Tiny, capable of identifying and categorizing potholes in real-time video streams. The severity level of each pothole (Low, Medium, High) is assessed based on its area in the frame. This system provides a valuable tool for monitoring road conditions and prioritizing maintenance efforts. The repository includes Python code, YOLO model files, and sample data for easy implementation and testing.
## Overview

The pothole detection system is designed to:

- Process a video stream or file input.
- Detect potholes in the video frames using YOLOv4 Tiny.
- Draw bounding boxes around detected potholes.
- Save images of detected potholes along with their coordinates.
- Display the frames per second (FPS) processed.

## Requirements

- Python
- OpenCV (cv2)
- Geocoder
- Ensure the necessary YOLOv4 Tiny model weights (`yolov4_tiny.weights`) and label names (`obj.names`) are available.)

## Usage

1. Ensure that the necessary libraries are installed.
2. Configure the paths to the YOLOv4 Tiny weights, configurations, and label names (`obj.names`) in the Python script.
3. Set the input video source (file path or camera) in the script.
4. Run the Python script `pothole_detection.py`.
5. Press 'q' to stop the video stream and terminate the detection process.

### `test.mp4`

This [video file](https://github.com/akshxyjagtap/Pothole-Detection-System-using-YOLO-Tiny-v4/blob/6b62af71427b198d772750a8daf432ebde423bb5/test.mp4) contains footage of roads with potholes, used for testing the pothole detection system.



### `result.avi` and `pothole_coordinate/`

These are the output files:

- [video file](https://github.com/akshxyjagtap/Pothole-Detection-System-using-YOLO-Tiny-v4/blob/6b62af71427b198d772750a8daf432ebde423bb5/result.avi) contains the resulting video with detected potholes highlighted.
- `pothole_coordinate/` is a folder that saves the location coordinates of detected potholes.

## Files and Directory Structure

- `pothole_detection.py`: Main Python script for pothole detection.
- `obj.names`: Text file containing label names for the YOLOv4 Tiny model.
- `yolov4_tiny.weights`: YOLOv4 Tiny pre-trained weights file.
- `yolov4_tiny.cfg`: YOLOv4 Tiny configuration file.
- `result.mp4`: Output video file with detected potholes.

## Parameters and Customization

- Adjust confidence and NMS thresholds for detection accuracy (`Conf_threshold`, `NMS_threshold`).
- Modify the output directory (`result_path`) for saving detected pothole images and coordinates.




## Notes

- Ensure CUDA and GPU support for faster processing if using GPU-enabled OpenCV.
- Fine-tune detection parameters for better accuracy based on specific scenarios and video quality.

## Contribution

Contributions to optimize, improve accuracy, or add new functionalities are welcome. Feel free to open issues or pull requests.


