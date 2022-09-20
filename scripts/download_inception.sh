# download inception model
mkdir -p tmp
wget -O tmp/inception_v3_frozen.pb.tar.gz \
     https://storage.googleapis.com/download.tensorflow.org/models/inception_v3_2016_08_28_frozen.pb.tar.gz
tar -xzf tmp/inception_v3_frozen.pb.tar.gz -C tmp/

# move model 
mkdir -p models/inception_graphdef/1
mv tmp/inception_v3_2016_08_28_frozen.pb models/inception_graphdef/1/model.graphdef

# download labels
wget -O models/inception_graphdef/config.pbtxt  https://raw.githubusercontent.com/triton-inference-server/server/main/docs/examples/model_repository/inception_graphdef/config.pbtxt
wget -O models/inception_graphdef/inception_labels.txt  https://raw.githubusercontent.com/triton-inference-server/server/main/docs/examples/model_repository/inception_graphdef/inception_labels.txt

# clean up
rm -rf tmp/