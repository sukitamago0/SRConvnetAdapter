import os


def make_sample_key(path: str) -> str:
    p = os.path.splitext(str(path))[0].replace("\\", "/").strip("/")
    return p.replace("/", "__")

