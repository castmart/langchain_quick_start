PHONY: create-venv
create-venv:
	python3 -m venv venv

PHONY: install-dependencies
install-dependencies:
	echo "Make sure you are installing in the virtual environment!!"
	pip install -r requirements.txt

PHONY: run-ollama
run-ollama:
	docker-compose up -d