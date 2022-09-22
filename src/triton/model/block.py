from typing import List, Optional
from pydantic import BaseModel


class ModelMetaDataBlock(BaseModel):
    name: str
    datatype: str
    shape: List[int]


class ModelConfigBlock(BaseModel):
    class Reshape(BaseModel):
        shape: List[int]

    name: str
    datatype: str
    format: str
    dims: List[int]
    reshape: Reshape
    is_shape_tensor: bool
    allow_ragged_batch: bool
    label_filename: Optional[str]
