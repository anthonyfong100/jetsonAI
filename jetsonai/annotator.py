import cv2
import numpy as np
from typing import Sequence


def box_label(
    image: cv2.Mat,
    box: Sequence[int],
    label="",
    color=(128, 128, 128),
    txt_color=(255, 255, 255),
    line_width=2,
) -> np.array:
    # Add one xyxy box to image with label
    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    cv2.rectangle(image, p1, p2, color, thickness=line_width, lineType=cv2.LINE_AA)
    if label:
        tf = max(line_width - 1, 1)  # font thickness
        width, height = cv2.getTextSize(
            label, 0, fontScale=line_width / 3, thickness=tf
        )[
            0
        ]  # text width, height
        outside = p1[1] - height >= 3
        p2 = p1[0] + width, p1[1] - height - 3 if outside else p1[1] + height + 3
        cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            image,
            label,
            (p1[0], p1[1] - 2 if outside else p1[1] + height + 2),
            0,
            line_width / 3,
            txt_color,
            thickness=tf,
            lineType=cv2.LINE_AA,
        )
    return np.asarray(image)
