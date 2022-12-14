import random
from typing import Tuple

BASE_DIR = "models"
DENSENET_ONNX_LABEL_PATH = f"{BASE_DIR}/densenet_onnx/densenet_labels.txt"
YOLOV5_LABEL_PATH = f"{BASE_DIR}/yolov5/yolov5_labels.txt"


def get_random_color() -> Tuple[int, int, int]:
    return tuple(random.sample(range(0, 255), 3))


class LabelLoader:
    def __init__(self) -> None:
        self.densenet_onnx_map = self.__load_labels_from_file(DENSENET_ONNX_LABEL_PATH)
        self.yolov5_map = self.__load_labels_from_file(YOLOV5_LABEL_PATH)
        self.yolov5_color_map = {
            class_ix: get_random_color() for class_ix in range(len(self.yolov5_map))
        }

    def __load_labels_from_file(self, path: int):
        with open(path, "r") as file:
            class_file_contents = file.readlines()
            class_map = {
                line_num: class_name.rstrip()
                for line_num, class_name in enumerate(class_file_contents)
            }
        return class_map


label_manager = LabelLoader()
