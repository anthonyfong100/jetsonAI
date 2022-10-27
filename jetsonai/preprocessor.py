import numpy as np
from jetsonai.triton.model.model import InputConfig
from tritonclient.utils import triton_to_np_dtype
import tritonclient.grpc.model_config_pb2 as mc
from PIL import Image

def __set_image_color(img:Image, channels:int) ->Image:
    return img.convert("L") if channels == 1 else img.convert("RGB") 

def __resize_image(img, width:int , height:int, data_type):
    resized_img = img.resize((width, height), Image.BILINEAR)
    resized = np.array(resized_img)
    if resized.ndim == 2:
        resized = resized[:, :, np.newaxis]
    npdtype = triton_to_np_dtype(data_type)
    typed = resized.astype(npdtype)
    return typed

def __normalize_image(img, scaling_schema:str, data_type:str, channels:int):
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

def __reorder_channels(img, format:int):
    return np.transpose(img, (2, 0, 1)) if format == mc.ModelInput.FORMAT_NCHW else img

def preprocess(img, input_config:InputConfig, normalize_schema:str,metadata_datatype:str):
    image = __set_image_color(img, input_config.channels)
    image = __resize_image(image, input_config.width, input_config.height, metadata_datatype)
    image = __normalize_image(image, normalize_schema, metadata_datatype, input_config.channels)
    return __reorder_channels(image,input_config.format)