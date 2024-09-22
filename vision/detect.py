import cv2
import numpy as np
import json
from ultralytics import YOLO

model_path = 'vision/weights/yolov8n_float32.tflite'

class ObjectDetection:
    def __init__(self, model_path, camera_matrix, object_height):
        """
        Initialize the ObjectDetection class with the given parameters.

        :param model_path: Path to the YOLO model in TFLite format.
        :param camera_matrix: Camera intrinsic matrix.
        :param object_height: Known height of the object in meters.
        """
        self.model = YOLO(model_path)  # Load the YOLO model
        self.camera_matrix = camera_matrix
        self.object_height = object_height

    def detect_objects(self, frame):
        """
        Detect objects in a given frame.

        :param frame: The input image frame.
        :return: List of bounding boxes and classes of detected objects.
        """
        results = self.model(frame)
        detected_objects = []

        for result in results:
            for box in result.boxes.xyxy:
                x_min, y_min, x_max, y_max = map(int, box)
                obj_class = result.names[int(box.cls[0])]
                detected_objects.append({
                    'class': obj_class,
                    'bounding_box': [x_min, y_min, x_max, y_max]
                })
        
        return detected_objects

    def calculate_distance(self, bounding_box):
        """
        Calculate the distance from the camera to the detected object using the known object height.

        :param bounding_box: Bounding box of the detected object [x_min, y_min, x_max, y_max].
        :return: Estimated distance from the camera to the object in meters.
        """
        # Calculate pixel height of the object
        x_min, y_min, x_max, y_max = bounding_box
        pixel_height = y_max - y_min

        # Calculate focal length from the camera matrix
        focal_length = self.camera_matrix[1, 1]

        # Calculate distance using the formula: distance = (object_height * focal_length) / pixel_height
        distance = (self.object_height * focal_length) / pixel_height
        return distance

    def get_center(self, bounding_box):
        """
        Calculate the center of the bounding box.

        :param bounding_box: Bounding box of the detected object [x_min, y_min, x_max, y_max].
        :return: Center coordinates (x, y) of the bounding box.
        """
        x_min, y_min, x_max, y_max = bounding_box
        center_x = x_min + (x_max - x_min) // 2
        center_y = y_min + (y_max - y_min) // 2
        return (center_x, center_y)

    def draw_bounding_box(self, frame, bounding_box, obj_class):
        """
        Draw a bounding box around the detected object.

        :param frame: The image frame.
        :param bounding_box: Bounding box of the detected object [x_min, y_min, x_max, y_max].
        :param obj_class: The class name of the detected object.
        """
        x_min, y_min, x_max, y_max = bounding_box
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(frame, obj_class, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    def save_to_json(self, filename, object_data):
        """
        Save the detected object data to a JSON file.

        :param filename: Name of the JSON file to save data.
        :param object_data: Dictionary containing structured information about detected objects.
                            The structure is as follows:
                            {
                                "objects": [
                                    {
                                        "class": "object_class_name",
                                        "distance": estimated_distance_in_meters,
                                        "center": (center_x, center_y)
                                    },
                                    ...
                                ]
                            }
        """
        with open(filename, 'w') as json_file:
            json.dump(object_data, json_file, indent=4)

    def process_frame(self, frame):
        """
        Process a frame to detect objects, calculate distances, and save the results.

        :param frame: The input image frame.
        :return: A structured dictionary containing the detected objects and their information.
        """
        detected_objects = self.detect_objects(frame)
        object_data = {"objects": []}

        for obj in detected_objects:
            bounding_box = obj['bounding_box']
            obj_class = obj['class']
            distance = self.calculate_distance(bounding_box)
            center = self.get_center(bounding_box)

            # Append object data to the list
            object_data["objects"].append({
                "class": obj_class,
                "distance": distance,
                "center": center
            })

            # Draw bounding box on the frame
            self.draw_bounding_box(frame, bounding_box, obj_class)

        return object_data