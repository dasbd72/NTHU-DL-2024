LINT_TARGETS := models utils

.Phony: lint
lint:
	isort --profile black $(LINT_TARGETS)
	black --line-length 79 $(LINT_TARGETS)
	flake8 --ignore E501,W503 $(LINT_TARGETS)
