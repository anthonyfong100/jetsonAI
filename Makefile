.PHONY: download_models
download_models:
	./scripts/download_inception.sh

.PHONY: clean
clean:
	rm -rf models/* tmp/
