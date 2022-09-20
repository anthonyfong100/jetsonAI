mkdir -p tmp
wget -O tmp/inception_v3_frozen.pb.tar.gz \
     https://storage.googleapis.com/download.tensorflow.org/models/inception_v3_2016_08_28_frozen.pb.tar.gz
tar -xzf tmp/inception_v3_frozen.pb.tar.gz -C tmp/
mv tmp/inception_v3_2016_08_28_frozen.pb models/inception_graphdef/1/model.graphdef
