import pytest

@pytest.fixture(autouse=True)
def marimo_temp_dir(monkeypatch, tmp_path):
    """tmp_path is unique to each test function.
    It will be cleared automatically according to pytest docs:
    https://docs.pytest.org/en/6.2.x/reference.html#tmp-path
    """
    monkeypatch.setenv("MARIMO_TEMP_DIR", str(tmp_path))
    return tmp_path
