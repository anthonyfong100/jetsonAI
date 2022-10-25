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
dev: monitoring
	kubectl apply -k k3s/triton

.PHONY: monitoring-stop
monitoring-stop:
	kubectl delete -k k3s/monitoring

.PHONY: dev-stop
dev-stop:
	kubectl delete -k k3s/triton

.PHONY: test
test:
	TF_FORCE_GPU_ALLOW_GROWTH=true OPENBLAS_CORETYPE=ARMV8 python3 src/image_client.py -u 172.20.238.9:30800 -m densenet_onnx -s INCEPTION -c 3 tests/data/car.jpeg

.PHONY: client
client:
	TF_FORCE_GPU_ALLOW_GROWTH=true OPENBLAS_CORETYPE=ARMV8 python3 client.py -u 172.20.238.9:30800 -m densenet_onnx -s INCEPTION -c 3 tests/data/car.jpeg

.PHONY: simple
simple:
	OPENBLAS_CORETYPE=ARMV8 python3 tests/simple_test.py -u 172.20.238.9:30800
	
.PHONY: release
release:
	docker build -t anthonyfong/jetson-triton . && docker push anthonyfong/jetson-triton