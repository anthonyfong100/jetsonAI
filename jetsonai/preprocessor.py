import numpy as np
import cv2
from typing import List, Dict
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
    image: Image, input_config: InputConfig, normalize_schema, metadata_type
):

    image_arr = np.copy(image)
    ih, iw = input_config.width, input_config.width
    h, w, _ = image_arr.shape

    scale = min(iw / w, ih / h)
    nw, nh = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image_arr, (nw, nh))
    npdtype = triton_to_np_dtype(metadata_type)

    image_padded = np.full(shape=[ih, iw, 3], fill_value=128.0, dtype=npdtype)
    dw, dh = (iw - nw) // 2, (ih - nh) // 2
    image_padded[dh : nh + dh, dw : nw + dw, :] = image_resized
    image_padded = image_padded / 255.0
    image_reordered = __reorder_channels(image_padded, mc.ModelInput.FORMAT_NCHW)
    image_padded = image_reordered[np.newaxis, ...].astype(npdtype)
    return image_padded


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


def get_preprocesser_func(model_name: str):
    preprocesser_map = {
        "densenet_onnx": preprocess_densenet,
        "yolov5": preprocess_yolov5,
    }
    return preprocesser_map[model_name]


def get_postprocess_func(model_name: str):
    preprocesser_map = {
        "densenet_onnx": __postprocess_densenet,
        # "yolov5": __postprocess_yolov5,
    }
    return preprocesser_map[model_name]
