import tritonclient.http as httpclient
from typing import Tuple
from jetsonai.triton.model.model import ModelConfig, ModelMetadata
from jetsonai.triton.model.enums import ClientType,get_client_module
from PIL import Image
from jetsonai.preprocessor import preprocess


class TritonClientApi:
    def __init__(self, api_client,client_type:ClientType, model_name: str, model_version: str, normalize_schema:str) -> None:
        self.triton_client = api_client
        self.client_type = client_type
        self.model_config, self.model_metadata = self.__get_model_metadata(model_name, model_version)
        self.normalize_schema = normalize_schema

    def __get_model_metadata(self, model_name: str, model_version: str)-> Tuple[ModelConfig, ModelMetadata]:
        """
        Check the configuration of a model to make sure it meets the
        requirements for an image classification network (as expected by
        this client)
        """
        model_metadata_dict = self.triton_client.get_model_metadata(
            model_name=model_name, model_version=model_version
        )
        model_config_dict = self.triton_client.get_model_config(
            model_name=model_name, model_version=model_version
        )
        model_config_dict["input_metadata_shape"] = model_metadata_dict["inputs"][0]["shape"]
        model_metadata_dict["max_batch_size"] = model_config_dict["max_batch_size"]
        model_config:ModelConfig = ModelConfig.parse_obj(model_config_dict)
        model_metadata:ModelMetadata = ModelMetadata.parse_obj(model_metadata_dict)

        return model_config, model_metadata

    def infer(self,image:Image):
        metadata_data_type = self.model_metadata.inputs[0].datatype
        img_preprocessed = preprocess(image, self.model_config.input_config,self.normalize_schema,metadata_data_type)
        client_module = get_client_module(self.client_type)
        triton_input = [client_module.InferInput(self.model_config.input_config.name, img_preprocessed.shape,metadata_data_type)]
        triton_input[0].set_data_from_numpy(img_preprocessed)
        return self.triton_client.infer(self.model_config.name,triton_input)
