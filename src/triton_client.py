class TritonClientApi:
    def __init__(self,api_client, model_name:str , model_version:str)->None:
      self.api_client = api_client
      self.model_metadata = self.__get_model_metadata(model_name ,model_version)
    
    def __get_model_metadata(self, model_name:str, model_version:str):
      pass

    def infer():
      pass