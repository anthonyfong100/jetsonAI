from typing import List, Dict
from pydantic import BaseModel, validator, root_validator
from jetsonai.triton.model.block import ModelMetaDataBlock, ModelConfigBlock
import tritonclient.grpc.model_config_pb2 as model_config
from typing import Optional

FP32_CONSTANT = "FP32"


class InputConfig(ModelConfigBlock):
    format: int
    height: int
    width: int
    channels: int
    has_input_batch_dim: bool
    input_metadata_shape: List[int]

    @root_validator(pre=True)
    def generate_input_config(cls, values):
        FORMAT_ENUM_TO_INT = dict(model_config.ModelInput.Format.items())
        values["format"] = FORMAT_ENUM_TO_INT[values["format"]]

        # assert (
        #     values["format"] == model_config.ModelInput.FORMAT_NCHW
        #     or values["format"] == model_config.ModelInput.FORMAT_NHWC
        # ), (
        #     f"Unexpected input format received {model_config.ModelInput.Format.Name(values['format'])} "
        #     f"expecting { model_config.ModelInput.Format.Name(model_config.ModelInput.FORMAT_NCHW)} "
        #     f"or {model_config.ModelInput.Format.Name(model_config.ModelInput.FORMAT_NHWC)}"
        # )

        # generate height width channels
        input_metadata_shape, has_input_batch_dim = (
            values["input_metadata_shape"],
            values["has_input_batch_dim"],
        )
        if values["format"] == model_config.ModelInput.FORMAT_NHWC:
            values["height"] = input_metadata_shape[1 if has_input_batch_dim else 0]
            values["width"] = input_metadata_shape[2 if has_input_batch_dim else 1]
            values["channels"] = input_metadata_shape[3 if has_input_batch_dim else 2]
        else:
            values["channels"] = input_metadata_shape[1 if has_input_batch_dim else 0]
            values["height"] = input_metadata_shape[2 if has_input_batch_dim else 1]
            values["width"] = input_metadata_shape[3 if has_input_batch_dim else 2]
        return values


class ModelConfig(BaseModel):
    name: str
    platform: str
    backend: str
    max_batch_size: int
    input: List[ModelConfigBlock]
    output: List[ModelConfigBlock]
    batch_input: List[ModelConfigBlock]
    batch_output: List[ModelConfigBlock]
    default_model_filename: str
    input_metadata_shape: List[int]
    input_config: InputConfig = None  # computed in validator

    @root_validator(pre=True)
    def generate_input_config(cls, values):
        input_config: ModelConfigBlock = ModelConfigBlock.parse_obj(values["input"][0])
        input_config_dict: dict = input_config.dict()
        values["input_config"] = InputConfig(
            input_metadata_shape=values["input_metadata_shape"],
            has_input_batch_dim=values["max_batch_size"] > 0,
            **input_config_dict,
        )
        return values

    @validator("input", "output")
    def check_input_output_len(cls, field):
        assert (
            len(field) > 0
        ), f"Inputs and outputs should have a length of more than one, received {len(field)}"
        return field


class ModelMetadata(BaseModel):
    name: str
    versions: List[str]
    platform: str
    max_batch_size: int
    inputs: List[ModelMetaDataBlock]
    outputs: List[ModelMetaDataBlock]

    @validator("inputs", "outputs")
    def check_single_input(cls, field):
        assert (
            len(field) > 0
        ), f"Inputs and outputs should have a length of one, received {len(field)}"
        return field

    @validator("outputs")
    def check_fp_32(cls, outputs: List[ModelMetaDataBlock]):
        for output in outputs:
            assert (
                output.datatype == FP32_CONSTANT
            ), f"output should have datatype FP32,{output.name} has received {output.datatype}"
        return outputs


class ClassificationResult(BaseModel):
    confidence: float
    class_id: int
    class_name: str


class ObjectDetectionResult(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int
    class_name: str
