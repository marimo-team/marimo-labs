import os
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageCms

from marimo_labs.huggingface import _processing_utils as processing_utils
import media_data


class TestTempFileManagement:
    def test_save_b64_to_cache(self, marimo_temp_dir):
        base64_file_1 = media_data.BASE64_IMAGE
        base64_file_2 = media_data.BASE64_AUDIO["data"]

        f = processing_utils.save_base64_to_cache(
            base64_file_1, cache_dir=marimo_temp_dir
        )
        try:  # Delete if already exists from before this test
            os.remove(f)
        except OSError:
            pass

        f = processing_utils.save_base64_to_cache(
            base64_file_1, cache_dir=marimo_temp_dir
        )
        assert (
            len([f for f in marimo_temp_dir.glob("**/*") if f.is_file()]) == 1
        )

        f = processing_utils.save_base64_to_cache(
            base64_file_1, cache_dir=marimo_temp_dir
        )
        assert (
            len([f for f in marimo_temp_dir.glob("**/*") if f.is_file()]) == 1
        )

        f = processing_utils.save_base64_to_cache(
            base64_file_2, cache_dir=marimo_temp_dir
        )
        assert (
            len([f for f in marimo_temp_dir.glob("**/*") if f.is_file()]) == 2
        )
