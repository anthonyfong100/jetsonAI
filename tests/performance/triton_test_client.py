from jetsonai.triton_client import TritonClientApi
from locust import User, task
from jetsonai.triton.model.enums import ClientType
import tritonclient.http as httpclient
from jetsonai.loaders import LocalFileLoader
import time

HOST = "172.20.238.9:30800"
MODEL_NAME = "densenet_onnx"
SCALING = "INCEPTION"
MODEL_VERSION = ""
IMAGE_FILENAME = "tests/data/car.jpeg"
NUM_CLASS_RESULT = 1


class TritonTestClientApi(User):
    def __init__(self, environment) -> None:
        self.environment = environment
        triton_client = httpclient.InferenceServerClient(
            url=HOST, verbose=False, concurrency=100
        )
        self.api_client = TritonClientApi(
            triton_client,
            ClientType.http,
            MODEL_NAME,
            MODEL_VERSION,
            SCALING,
            NUM_CLASS_RESULT,
        )
        self.loader = LocalFileLoader(IMAGE_FILENAME)
        self._request_event = environment.events.request

    @task
    def infer(self):
        for image in self.loader.iter():
            request_meta = {
                "request_type": "http",
                "name": "infer",
                "start_time": time.time(),
                "response_length": 0,
                "response": None,
                "context": {},
                "exception": None,
            }
            start_perf_counter = time.perf_counter()
            # time.sleep(0.1)
            request_meta["response"] = self.api_client.infer(image)
            request_meta["response_time"] = (
                time.perf_counter() - start_perf_counter
            ) * 1000
            self._request_event.fire(**request_meta)
            return request_meta["response"]
