from typing import List, Any
from pydantic import BaseModel, validator
from src.triton.model.block import ModelMetaDataBlock, ModelConfigBlock

FP32_CONSTANT = "FP32"


class ModelConfig(BaseModel):
    name: str
    platform: int
    backend: int
    max_batch_size: int
    input: List[ModelConfigBlock]
    output: List[ModelConfigBlock]
    batch_input: List[ModelConfigBlock]
    batch_output: List[ModelConfigBlock]
    default_model_filename: str


class ModelMetadata(BaseModel):
    name: str
    versions: List[str]
    platform: str
    inputs: List[ModelMetaDataBlock]
    outputs: List[ModelMetaDataBlock]

    @validator("inputs", "outputs")
    def check_single_input(cls, field):
        assert (
            len(field) == 1
        ), f"Inputs and outputs should have a length of one, received {len(field)}"
        return field

    @validator("outputs")
    def check_fp_32(cls, outputs: List[ModelMetaDataBlock]):
        for output in outputs:
            assert (
                output.datatype == FP32_CONSTANT
            ), f"output should have datatype FP32,{output.name} has received {output.datatype}"
        return outputs
