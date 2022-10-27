from jetsonai.triton.model.model import ModelConfig, ModelMetadata


class TritonClientApi:
    def __init__(self, api_client, model_name: str, model_version: str) -> None:
        self.triton_client = api_client
        self.model_metadata = self.__get_model_metadata(model_name, model_version)

    def __get_model_metadata(self, model_name: str, model_version: str):
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

        print(model_config.input_config)
        input_metadata = model_metadata.inputs[0]
        input_config = model_config.input[0]
        output_metadata = model_metadata.outputs[0]

        # Output is expected to be a vector. But allow any number of
        # dimensions as long as all but 1 is size 1 (e.g. { 10 }, { 1, 10
        # }, { 10, 1, 1 } are all ok). Ignore the batch dimension if there
        # is one.
        output_batch_dim = model_config.max_batch_size > 0
        non_one_cnt = 0
        for dim in output_metadata.shape:
            if output_batch_dim:
                output_batch_dim = False
            elif dim > 1:
                non_one_cnt += 1
                if non_one_cnt > 1:
                    raise Exception("expecting model output to be a vector")

        # Model input must have 3 dims, either CHW or HWC (not counting
        # the batch dimension), either CHW or HWC
        input_batch_dim = model_config.max_batch_size > 0
        expected_input_dims = 3 + (1 if input_batch_dim else 0)
        if len(input_metadata.shape) != expected_input_dims:
            raise Exception(
                "expecting input to have {} dimensions, model '{}' input has {}".format(
                    expected_input_dims, model_metadata.name, len(input_metadata.shape)
                )
            )
        return (
            model_config.max_batch_size,
            input_metadata.name,
            output_metadata.name,
            input_config.format,
            input_metadata.datatype,
        )

    def infer():
        pass
