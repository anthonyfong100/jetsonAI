# move model 
mkdir -p models/simple/1

# download labels
wget -O models/simple/config.pbtxt  https://raw.githubusercontent.com/triton-inference-server/server/main/docs/examples/model_repository/simple/config.pbtxt
wget -O models/simple/1/model.graphdef  https://raw.githubusercontent.com/triton-inference-server/server/main/docs/examples/model_repository/simple/1/model.graphdef

# clean up
rm -rf tmp/