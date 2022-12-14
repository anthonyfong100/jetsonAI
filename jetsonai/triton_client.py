import tritonclient.http as httpclient
import tritonclient.grpc as grpcclient
import asyncio
import numpy as np
from typing import Tuple, Union, List
from jetsonai.triton.model.model import (
    ModelConfig,
    ModelMetadata,
    ClassificationResult,
    ObjectDetectionResult,
)
from jetsonai.triton.model.enums import ClientType, get_client_module
from PIL import Image
from jetsonai.preprocessor import get_preprocesser_func, get_postprocess_func

outputResponseType = Union[httpclient.InferResult, grpcclient.InferResult]
outputResultType = Union[ClassificationResult, ObjectDetectionResult]


class TritonClientApi:
    def __init__(
        self,
        api_client,
        client_type: ClientType,
        model_name: str,
        model_version: str,
        normalize_schema: str,
        classes: int,
    ) -> None:
        self.triton_client = api_client
        self.client_type = client_type
        self.model_config, self.model_metadata = self.__get_model_metadata(
            model_name, model_version
        )
        self.normalize_schema = normalize_schema
        self.classes = classes
        self.preprocessor_func, self.postprocess_func = get_preprocesser_func(
            model_name
        ), get_postprocess_func(model_name)

    def __get_model_metadata(
        self, model_name: str, model_version: str
    ) -> Tuple[ModelConfig, ModelMetadata]:
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
        model_config_dict["input_metadata_shape"] = model_metadata_dict["inputs"][0][
            "shape"
        ]
        model_metadata_dict["max_batch_size"] = model_config_dict["max_batch_size"]
        model_config: ModelConfig = ModelConfig.parse_obj(model_config_dict)
        model_metadata: ModelMetadata = ModelMetadata.parse_obj(model_metadata_dict)

        return model_config, model_metadata

    def __generate_input_output(self, image: Image):
        metadata_data_type = self.model_metadata.inputs[0].datatype
        output_name = self.model_metadata.outputs[0].name
        img_preprocessed = self.preprocessor_func(
            image,
            self.model_config.input_config,
            self.normalize_schema,
            metadata_data_type,
        )
        client_module = get_client_module(self.client_type)
        triton_input = [
            client_module.InferInput(
                self.model_config.input_config.name,
                img_preprocessed.shape,
                metadata_data_type,
            )
        ]
        triton_input[0].set_data_from_numpy(img_preprocessed)
        outputs = [client_module.InferRequestedOutput(output_name)]
        return triton_input, outputs, output_name

    def __async_infer_promise(self, image: Image) -> httpclient.InferAsyncRequest:
        triton_input, outputs, output_name = self.__generate_input_output(image)
        result_promise = self.triton_client.async_infer(
            self.model_config.name, triton_input, outputs=outputs
        )
        return result_promise, output_name

    async def async_infer(self, image: Image):
        result_promise, output_name = self.__async_infer_promise(image)
        event_loop = asyncio._get_running_loop()
        res = await event_loop.run_in_executor(None, result_promise.get_result)
        return self.__parse_inference(res, output_name)

    def infer(self, image: Image) -> List[outputResultType]:
        triton_input, outputs, output_name = self.__generate_input_output(image)
        results = self.triton_client.infer(
            self.model_config.name, triton_input, outputs=outputs
        )
        return self.__parse_inference(results, output_name)

    def __parse_inference(
        self,
        results: outputResponseType,
        output_name: str,
        supports_batching: bool = False,
    ) -> List[outputResultType]:
        output_array = results.as_numpy(output_name)
        return self.postprocess_func(output_array, self.classes)
