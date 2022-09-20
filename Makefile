.PHONY: download_models
download_models:
	./scripts/download_inception.sh

.PHONY: clean
clean:
	rm -rf models/* tmp/ .venv

.PHONY: dev
dev:
	kubectl apply -k k3s/base



.PHONY: test
test:
	OPENBLAS_CORETYPE=ARMV8 python src/image_client.py -u 172.20.238.9:30800 -m inception_graphdef -s INCEPTION -x 1 -c 1 tests/data/car.jpeg
