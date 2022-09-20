.PHONY: download_models
download_models:
	./scripts/download_inception.sh

.PHONY: clean
clean:
	rm -rf models/* tmp/ .venv

.PHONY: dev
dev:
	kubectl apply -k k3s/base

.PHONY: dev-stop
dev-stop:
	kubectl delete -k k3s/base

.PHONY: test
test:
	TF_FORCE_GPU_ALLOW_GROWTH=true OPENBLAS_CORETYPE=ARMV8 python src/image_client.py -u 172.20.238.9:30800 -m inception_graphdef -s INCEPTION -x 1 -c 1 tests/data/car.jpeg

.PHONY: release-image
release-image:
	docker build -t anthonyfong/jetson-triton . && docker push anthonyfong/jetson-triton