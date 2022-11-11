import numpy as np
import torch
import time
import torchvision
import cv2
from typing import List, Dict
from jetsonai.utils import xywh2xyxy, letterbox
from jetsonai.triton.model.model import InputConfig
from tritonclient.utils import triton_to_np_dtype
from jetsonai.triton.model import ModelResponse
from jetsonai.loaders.labels import label_manager
import tritonclient.grpc.model_config_pb2 as mc
from PIL import Image


def __set_image_color(img: Image, channels: int) -> Image:
    return img.convert("L") if channels == 1 else img.convert("RGB")


def __resize_image(img, width: int, height: int, data_type):
    resized_img = img.resize((width, height), Image.BILINEAR)
    resized = np.array(resized_img)
    if resized.ndim == 2:
        resized = resized[:, :, np.newaxis]
    npdtype = triton_to_np_dtype(data_type)
    typed = resized.astype(npdtype)
    return typed


def __normalize_image(img, scaling_schema: str, data_type: str, channels: int):
    npdtype = triton_to_np_dtype(data_type)
    if scaling_schema == "INCEPTION":
        scaled = (img / 127.5) - 1
    elif scaling_schema == "VGG":
        if channels == 1:
            scaled = img - np.asarray((128,), dtype=npdtype)
        else:
            scaled = img - np.asarray((123, 117, 104), dtype=npdtype)
    else:
        scaled = img
    return scaled


def __reorder_channels(img, format: int):
    return np.transpose(img, (2, 0, 1)) if format == mc.ModelInput.FORMAT_NCHW else img


def preprocess_densenet(
    img, input_config: InputConfig, normalize_schema: str, metadata_datatype: str
):
    image = __set_image_color(img, input_config.channels)
    image = __resize_image(
        image, input_config.width, input_config.height, metadata_datatype
    )
    image = __normalize_image(
        image, normalize_schema, metadata_datatype, input_config.channels
    )
    return __reorder_channels(image, input_config.format)


def preprocess_yolov5(
    image: cv2.Mat, input_config: InputConfig, normalize_schema, metadata_type
):
    im = letterbox(image, input_config.width, stride=32, auto=False)[0]  # padded resize
    print(f"post letterbox shape{im.shape}")
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)  # contiguous
    print(f"im shape{im.shape}")
    # im = torch.from_numpy(im)
    # im = im.float()  # uint8 to fp16/32
    im = im / 255.0  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim
    npdtype = triton_to_np_dtype(metadata_type)
    return im.astype(npdtype)


def __top_k_ix(arr: np.array, top_k: int):
    return np.argsort(arr)[-top_k:][::-1]


def __postprocess_densenet(results: np.array, top_k: int = 1) -> List[ModelResponse]:
    top_class_index = __top_k_ix(results, top_k)
    processed_results: List[ModelResponse] = []
    densenet_classes_map = label_manager.densenet_onnx_map
    for index in top_class_index:
        processed_results.append(
            ModelResponse(
                class_id=index,
                confidence=results[index],
                class_name=densenet_classes_map[index],
            )
        )
    return processed_results


def box_iou(box1, box2, eps=1e-7):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1.unsqueeze(1).chunk(2, 2), box2.unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / ((a2 - a1).prod(2) + (b2 - b1).prod(2) - inter + eps)


def non_max_suppression(
    prediction,
    conf_thres=0.25,
    iou_thres=0.45,
    classes=None,
    agnostic=False,
    multi_label=False,
    labels=(),
    max_det=300,
    nm=0,  # number of masks
):
    """Non-Maximum Suppression (NMS) on inference results to reject overlapping detections
    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """
    print(prediction.shape)

    if isinstance(
        prediction, (list, tuple)
    ):  # YOLOv5 model in validation model, output = (inference_out, loss_out)
        prediction = prediction[0]  # select only inference output

    device = prediction.device
    mps = "mps" in device.type  # Apple MPS
    if mps:  # MPS not fully supported yet, convert tensors to CPU before NMS
        prediction = prediction.cpu()
    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - nm - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Checks
    assert (
        0 <= conf_thres <= 1
    ), f"Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0"
    assert (
        0 <= iou_thres <= 1
    ), f"Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0"

    # Settings
    # min_wh = 2  # (pixels) minimum box width and height
    max_wh = 7680  # (pixels) maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 0.5 + 0.05 * bs  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    mi = 5 + nc  # mask start index
    output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            lb = labels[xi]
            v = torch.zeros((len(lb), nc + nm + 5), device=x.device)
            v[:, :4] = lb[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
            x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box/Mask
        box = xywh2xyxy(
            x[:, :4]
        )  # center_x, center_y, width, height) to (x1, y1, x2, y2)
        mask = x[:, mi:]  # zero columns if no masks

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:mi] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, 5 + j, None], j[:, None].float(), mask[i]), 1)
        else:  # best class only
            conf, j = x[:, 5:mi].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float(), mask), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence
        else:
            x = x[x[:, 4].argsort(descending=True)]  # sort by confidence

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3e3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * scores[None]  # box weights
            x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(
                1, keepdim=True
            )  # merged boxes
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        output[xi] = x[i]
        if mps:
            output[xi] = output[xi].to(device)
        if (time.time() - t) > time_limit:
            print(f"WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded")
            break  # time limit exceeded

    return output


def __postprocess_yolov5(outputs: np.array, top_k: int):
    outputs = torch.tensor(outputs)
    pred = non_max_suppression(outputs)
    return pred


def get_preprocesser_func(model_name: str):
    preprocesser_map = {
        "densenet_onnx": preprocess_densenet,
        "yolov5": preprocess_yolov5,
    }
    return preprocesser_map[model_name]


def get_postprocess_func(model_name: str):
    preprocesser_map = {
        "densenet_onnx": __postprocess_densenet,
        "yolov5": __postprocess_yolov5,
    }
    return preprocesser_map[model_name]
