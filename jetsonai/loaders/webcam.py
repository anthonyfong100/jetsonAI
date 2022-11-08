import cv2

WINDOW_NAME = "Jetson Feed"


class WebCamLoader:
    def __init__(self) -> None:
        self.cam = cv2.VideoCapture(0)
        self.window_name = WINDOW_NAME

    def __enter__(self):
        cv2.namedWindow(WINDOW_NAME)
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.cam.release()
        cv2.destroyAllWindows()
        print("shutting down all video feeds")

    def iter(self):
        while True:
            res, frame = self.cam.read()
            key = cv2.waitKey(1)
            yield res, frame
