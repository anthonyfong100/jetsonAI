import os
from typing import List
import cv2


class LocalFileLoader:
    def __init__(self, file_path: str):
        self.file_paths = self.__get_all_files(file_path)

    def __get_all_files(self, file_path: str) -> List[str]:
        if os.path.isdir(file_path):
            return sorted(
                [
                    os.path.join(file_path, f)
                    for f in os.listdir(file_path)
                    if os.path.isfile(os.path.join(file_path, f))
                ]
            )
        return [file_path]

    def iter(self) -> "cv2.Mat":
        for file_path in self.file_paths:
            yield cv2.imread(file_path)
