# download inception model
mkdir -p models/densenet_onnx/1
wget -O models/densenet_onnx/1/model.onnx https://contentmamluswest001.blob.core.windows.net/content/14b2744cf8d6418c87ffddc3f3127242/9502630827244d60a1214f250e3bbca7/08aed7327d694b8dbaee2c97b8d0fcba/densenet121-1.2.onnx

# download labels
wget -O models/densenet_onnx/config.pbtxt https://raw.githubusercontent.com/triton-inference-server/server/master/docs/examples/model_repository/densenet_onnx/config.pbtxt
wget -O models/densenet_onnx/densenet_labels.txt https://raw.githubusercontent.com/triton-inference-server/server/master/docs/examples/model_repository/densenet_onnx/densenet_labels.txt

# clean up
rm -rf tmp/

