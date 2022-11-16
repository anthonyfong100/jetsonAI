import cv2
import numpy as np
from typing import List, Tuple
from jetsonai.triton.model import ObjectDetectionResult
from jetsonai.loaders.labels import label_manager
import torch


def clip_boxes(boxes, shape):
    # Clip boxes (xyxy) to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[0].clamp_(0, shape[1])  # x1
        boxes[1].clamp_(0, shape[0])  # y1
        boxes[2].clamp_(0, shape[1])  # x2
        boxes[3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[[0, 2]] = boxes[[0, 2]].clip(0, shape[1])  # x1, x2
        boxes[[1, 3]] = boxes[[1, 3]].clip(0, shape[0])  # y1, y2


def scale_box(
    img1_shape: Tuple[int, int],
    box: List[int],
    img0_shape: Tuple[int, int],
    ratio_pad=None,
):
    # Rescale box (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(
            img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1]
        )  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (
            img1_shape[0] - img0_shape[0] * gain
        ) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    box[[0, 2]] -= pad[0]  # x padding
    box[[1, 3]] -= pad[1]  # y padding
    box[:4] /= gain
    clip_boxes(box, img0_shape)
    return box


def draw_box_label(
    image: "cv2.Mat",
    prediction: ObjectDetectionResult,
    prediction_iamge_shape: Tuple[int, int],
    line_width=2,
    text_color=(255, 255, 255),
) -> np.array:

    box = np.array([prediction.x1, prediction.y1, prediction.x2, prediction.y2])
    box_scaled = scale_box(prediction_iamge_shape, box, image.shape)
    class_color = label_manager.yolov5_color_map[prediction.class_id]
    # Add one xyxy box to image with label
    p1 = (int(box_scaled[0]), int(box_scaled[1]))
    p2 = (int(box_scaled[2]), int(box_scaled[3]))
    cv2.rectangle(
        image, p1, p2, class_color, thickness=line_width, lineType=cv2.LINE_AA
    )
    tf = max(line_width - 1, 1)  # font thickness
    width, height = cv2.getTextSize(
        prediction.class_name, 0, fontScale=line_width / 3, thickness=tf
    )[
        0
    ]  # text width, height
    outside = p1[1] - height >= 3
    p2 = p1[0] + width, p1[1] - height - 3 if outside else p1[1] + height + 3
    cv2.rectangle(image, p1, p2, class_color, -1, cv2.LINE_AA)  # filled
    cv2.putText(
        image,
        prediction.class_name,
        (p1[0], p1[1] - 2 if outside else p1[1] + height + 2),
        0,
        line_width / 3,
        text_color,
        thickness=tf,
        lineType=cv2.LINE_AA,
    )
    return np.asarray(image)


def draw_box_labels(
    image: "cv2.Mat",
    predictions: List[ObjectDetectionResult],
    line_width=2,
) -> np.array:
    for pred in predictions:
        image = draw_box_label(image, pred, line_width)
    return image
