from pydantic import BaseModel


class PreprocessConfig(BaseModel):
    format: int
    dtype: str
    channel: int
    height: int
    width: int
    scaling_type: str
