# Jetson Cluster hosting Triton server on K3s
The documentation of this repo builds upon this [blog post](https://thenewstack.io/tutorial-edge-ai-with-triton-inference-server-kubernetes-jetson-mate/)

### Set up Jetson with Python
```
sudo apt-get update
sudo apt-get upgrade
sudo apt-get install git cmake python3-dev nano
sudo apt-get install libhdf5-serial-dev hdf5-tools libhdf5-dev zlib1g-dev zip libjpeg8-dev
sudo apt-get install python3-pip
sudo pip3 install -U pip testresources setuptools

sudo groupadd docker
sudo usermod -aG docker ${USER}
su -s ${USER}
```

### Setting up the jetson nanos
Follow this [link](https://gilberttanner.com/blog/jetson-nano-getting-started/) to setup up all jetson nanos 

### Setting up k3s in master node
Only run this on one of the jetson nano nodes
```
mkdir  -p ~/.kube
curl -sfL https://get.k3s.io | \
 K3S_TOKEN=jetsonmate \
 K3S_KUBECONFIG_MODE="644" \
 INSTALL_K3S_EXEC="--docker --disable servicelb --disable traefik" \
 K3S_KUBECONFIG_OUTPUT="$HOME/.kube/config" \
 sh -
```

### Setting up k3s in worker node
When setting up the worker nodes, run the following:

```
curl -sfL https://get.k3s.io | \
 K3S_TOKEN="token-value" \
 K3S_URL="https://url:6443" \
 INSTALL_K3S_EXEC="--docker"
```

The ip of master can be obtained by typing ipconfig on the master node. The k3s token can be obtained via `cat /var/lib/rancher/k3s/server/node-token`

### Setting up docker in worker node
```
sudo groupadd docker
sudo usermod -aG docker ${USER}
sudo chmod 666 /var/run/docker.sock
```
### Quick start
Run `make dev` on master node to spin up all the daemon sets


### Testing local setup
Run `make simple` after running the k3s cluster
Run `make test` after running the k3s cluster

### Access to GPU from k3s node
Run `kubectl apply -k tests/k3s` to check the nvidia output
