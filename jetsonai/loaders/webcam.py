import asyncio
from jetsonai.constants import WAIT_DURATION_MS
import cv2
from typing import Callable
from jetsonai.utils import gstreamer_pipeline
WINDOW_NAME = "Jetson Feed"


class WebCamLoader:
    def __init__(self,g_streamer: bool = False) -> None:
        if g_streamer:
            self.cam = cv2.VideoCapture(-1)
        else:
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
            _ = cv2.waitKey(WAIT_DURATION_MS)
            yield res, frame

    async def iter_append_q(
        self, sleep_duration_seconds: float = 0.01, async_callback_func: Callable = None
    ):
        while True:
            res, frame = self.cam.read()
            if async_callback_func:
                await async_callback_func(frame)
                await asyncio.sleep(sleep_duration_seconds)
                continue
