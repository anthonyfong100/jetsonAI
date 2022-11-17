"""
Orchestrator which stores the async queues and calls the displays method
"""
import cv2
import asyncio
from jetsonai.triton_client import TritonClientApi
from jetsonai.constants import (
    INFERENCE_Q_MAX_SIZE,
    DISPLAY_Q_MAX_SIZE,
    IMG_CAPTURE_SLEEP_DURATION_SEC,
)
from jetsonai.loaders import WebCamLoader
from jetsonai.annotator import draw_box_labels
from jetsonai.constants import YOLOV5_INPUT_HEIGHT, YOLOV5_INPUT_WIDTH, WAIT_DURATION_MS
from typing import List


class PipelineOrchestrator:
    def __init__(
        self,
        loader: WebCamLoader,
        triton_clients: List[TritonClientApi],
    ) -> None:
        self.triton_clients = triton_clients
        self.loader = loader
        self.inference_queue = asyncio.Queue(maxsize=INFERENCE_Q_MAX_SIZE)
        self.display_queue = asyncio.Queue(maxsize=DISPLAY_Q_MAX_SIZE)

    async def __insert_to_q(self, frame):
        print(f"inserting to q {self.inference_queue.qsize()}")
        await self.inference_queue.put(frame)

    async def __infer(self, triton_client: TritonClientApi, agent_id: int):
        while True:

            frame = await self.inference_queue.get()
            results = await triton_client.async_infer(frame)
            print(f"agent {agent_id} returned with {results}")
            img_with_boxes = draw_box_labels(
                frame, results, (YOLOV5_INPUT_HEIGHT, YOLOV5_INPUT_WIDTH)
            )
            await self.display_queue.put(img_with_boxes)
            self.inference_queue.task_done()

    async def __display(self):
        while True:
            annotated_img = await self.display_queue.get()
            cv2.imshow(self.loader.window_name, annotated_img)
            _ = cv2.waitKey(WAIT_DURATION_MS)
            self.display_queue.task_done()

    def run_pipeline(self):
        loop = asyncio.get_event_loop()
        frame_producer = self.loader.iter_append_q(
            sleep_duration_seconds=IMG_CAPTURE_SLEEP_DURATION_SEC,
            async_callback_func=self.__insert_to_q,
        )
        inference_consumers = [
            self.__infer(triton_client, agent_id)
            for agent_id, triton_client in enumerate(self.triton_clients)
        ]
        display_consumer = self.__display()
        pipeline = asyncio.gather(
            frame_producer, display_consumer, *inference_consumers
        )
        loop.run_until_complete(pipeline)
        loop.close()
