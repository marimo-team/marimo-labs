.PHONY: help
# show this help
help:
	@# prints help for rules preceded with comment
	@# https://stackoverflow.com/a/35730928
	@awk '/^#/{c=substr($$0,3);next}c&&/^[[:alpha:]][[:alnum:]_-]+:/{print substr($$1,1,index($$1,":")),c}1{c=0}' Makefile | column -s: -t

.PHONY: py
py:
	pip install -e .

.PHONY: check
check:
	./scripts/pyfix.sh

.PHONY: check-test
# run all checks and tests
check-test: check test

.PHONY: test
# run all checks and tests
test:
	pytest

.PHONY: wheel
# build wheel
wheel:
	python -m build
