import io
import base64
import tiktoken

from PIL import Image
from math import ceil


def count_txt_tokens(string: str, llm_version: str) -> int:
    enc = tiktoken.encoding_for_model(llm_version)
    num_tokens = len(enc.encode(string))
    return num_tokens


def resize(width, height):
    if width > 1024 or height > 1024:
        if width > height:
            height = int(height * 1024 / width)
            width = 1024
        else:
            width = int(width * 1024 / height)
            height = 1024
    return width, height


def count_img_tokens(img_b64: str) -> int:
    if img_b64.startswith("data:image/jpeg;base64,"):
        img_b64 = img_b64[len("data:image/jpeg;base64,") :]
    img_data = base64.b64decode(img_b64)
    img = Image.open(io.BytesIO(img_data))
    width, height = img.size
    width, height = resize(width, height)
    h = ceil(height / 512)
    w = ceil(width / 512)
    num_tokens = 85 + 170 * h * w

    return num_tokens
