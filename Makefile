.PHONY: download_models
download_models:
	./scripts/download_inception.sh

.PHONY: clean
clean:
	rm -rf models/* tmp/ .venv

.PHONY: monitoring
monitoring:
	kubectl apply -k k3s/monitoring

.PHONY: dev
dev:
	kubectl apply -k k3s/triton

.PHONY: docker
docker:
	docker run -p 8000:8000 -p 8001:8001 -p 8002:8002 jetsonai --min-supported-compute-capability=5.3 --model-repository=/opt/triton/models --backend-config=tensorflow,version=2 --strict-model-config=false --log-verbose 3 --log-info 1 --log-warning 1 --log-error 1

.PHONY: monitoring-stop
monitoring-stop:
	kubectl delete -k k3s/monitoring

.PHONY: dev-stop
dev-stop:
	kubectl delete -k k3s/triton

.PHONY: test
test:
	# TF_FORCE_GPU_ALLOW_GROWTH=true OPENBLAS_CORETYPE=ARMV8 python3 jetsonai/image_client.py -u 172.20.238.9:30800 -m densenet_onnx -s INCEPTION -c 3 tests/data/car.jpeg
	TF_FORCE_GPU_ALLOW_GROWTH=true OPENBLAS_CORETYPE=ARMV8 python3 jetsonai/image_client.py -u localhost:8000 -m densenet_onnx -s INCEPTION -c 3 tests/data/car.jpeg

.PHONY: densenet
densenet:
	# TF_FORCE_GPU_ALLOW_GROWTH=true OPENBLAS_CORETYPE=ARMV8 python3 client.py -u 172.20.238.9:30800 -m densenet_onnx -s INCEPTION -c 3 tests/data/car.jpeg
	TF_FORCE_GPU_ALLOW_GROWTH=true OPENBLAS_CORETYPE=ARMV8 python3 client.py -u localhost:8000 -m densenet_onnx -s INCEPTION -c 3 tests/data/car.jpg

.PHONY: yolov5
yolov5:
	#TF_FORCE_GPU_ALLOW_GROWTH=true OPENBLAS_CORETYPE=ARMV8 python3 client.py -u localhost:8000 -m yolov5 -c 1 tests/data/bus.jpg
	TF_FORCE_GPU_ALLOW_GROWTH=true OPENBLAS_CORETYPE=ARMV8 python3 client.py -u localhost:30800 -m yolov5 -c 1 tests/data/bus.jpg --gstream


.PHONY: simple
simple:
	# OPENBLAS_CORETYPE=ARMV8 python3 tests/simple_test.py -u 172.20.238.9:30800
	# OPENBLAS_CORETYPE=ARMV8 python3 tests/simple_test.py -u localhost:8000
	
.PHONY: release
release:
	docker build -t anthonyfong/jetson-triton . && docker push anthonyfong/jetson-triton

.PHONY: perf-test
perf-test:
	OPENBLAS_CORETYPE=ARMV8 locust --config .locust.conf
