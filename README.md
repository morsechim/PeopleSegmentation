# People Segmentation and Annotation

This program performs real-time object detection and annotation on a video stream. It utilizes YOLOv5 model for object detection and a custom annotation module for adding labels and masks to the detected objects.

## Requirements

- Python 3.x
- OpenCV (`cv2`)
- PyTorch
- NumPy
- Ultralytics
- `supervision` module (assumed to be available)

## Installation

1. Clone this repository:

    ```bash
    git clone https://github.com/morsechim/PeopleSegmentation.git
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Make sure to have the pre-trained YOLOv5 model available at `./weights/yolov8n-seg.pt`.

## Usage

1. Run the script `main.py`:

    ```bash
    python main.py
    ```

2. Provide the path to the input video file when prompted.

3. The annotated video will be saved in the `data` directory as `output.mp4`.

4. Press `q` to exit the program.

## Customization

You can customize the following parameters in the script:

- `conf_threshold`: Confidence threshold for object detection.
- `selected_classes`: Classes to filter detections.
- `model_path`: Path to the pre-trained YOLOv8 Segment model.
- Output video settings like codec, frame rate, etc.

## Acknowledgments

- This project utilizes the YOLOv5 implementation from the Ultralytics repository.
- The annotation module is adapted from the `supervision` library.

## License

This project is licensed under the [MIT License](LICENSE).