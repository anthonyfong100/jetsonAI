import tritonclient.grpc as grpcclient
import tritonclient.http as httpclient
from enum import Enum

class ClientType(Enum):
  http = "http"
  grpc = "grpc"

def get_client_module(client_type:ClientType):
  client_type_mapping = {
    ClientType.http : httpclient,
    ClientType.grpc : grpcclient,
  }
  return client_type_mapping[client_type]
