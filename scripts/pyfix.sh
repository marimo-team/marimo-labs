#!/bin/sh

echo "[fix: ruff]"
ruff check marimo_labs/ --fix
echo "[fix: black]"
black marimo_labs/
black tests/
echo "[check: typecheck]"
mypy marimo_labs/
