import cv2

class Camera:
    def __init__(self, camera_id=0, width=640, height=480, fps=60):
        """
        Initialize the Camera object.
        
        :param camera_id: ID of the external camera (default is 0 for the first camera).
        :param width: Desired width of the camera feed in pixels.
        :param height: Desired height of the camera feed in pixels.
        :param fps: Frames per second for the camera feed.
        """
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None

    def start(self):
        """
        Start the video capture from the camera using OpenCV.
        
        Opens the camera feed and sets the properties such as width, height, and FPS.
        Raises an exception if the camera cannot be opened.
        """
        self.cap = cv2.VideoCapture(self.camera_id)

        if not self.cap.isOpened():
            raise Exception(f"Error: Unable to open the camera with ID {self.camera_id}")

        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

    def get_frame(self):
        """
        Capture a single frame from the camera.
        
        :return: The captured frame as a NumPy array.
        :raises: Exception if the frame cannot be captured.
        """
        if self.cap is None or not self.cap.isOpened():
            raise Exception("Error: Camera is not opened")

        # Read a frame from the camera
        ret, frame = self.cap.read()

        if not ret:
            raise Exception("Error: Failed to capture frame from the camera")

        return frame

    def stop(self):
        """
        Stop the video capture and release the camera resource.
        
        This method ensures the camera feed is properly closed, freeing up system resources.
        """
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()

    def is_opened(self):
        """
        Check if the camera is currently opened.
        
        :return: True if the camera is opened, False otherwise.
        """
        return self.cap.isOpened() if self.cap else False

    def set_resolution(self, width, height):
        """
        Set the resolution of the camera feed.
        
        :param width: New width for the camera feed.
        :param height: New height for the camera feed.
        """
        self.width = width
        self.height = height

        if self.cap is not None and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

    def set_fps(self, fps):
        """
        Set the frames per second (FPS) for the camera feed.
        
        :param fps: New frames per second for the camera feed.
        """
        self.fps = fps

        if self.cap is not None and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
