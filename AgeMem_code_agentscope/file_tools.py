import json
import re
from typing import Any
from pathlib import Path
import os


def append_json_array_to_jsonl(json_array, filename):
    """
    Append each object in the JSON array to a JSONL file.
    The file will be created automatically if it does not exist.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "a", encoding="utf-8") as f:
        for item in json_array:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def append_json_to_jsonl(file_path: str, obj: Any, ensure_ascii: bool = False) -> None:
    """
        Append a JSON object to the specified JSONL file.
        If the file does not exist, it will be created automatically.

        Args:
            file_path: Path to the JSONL file.
            obj: The JSON object to write (usually a dict).
            ensure_ascii: Whether to escape non-ASCII characters. Defaults to False (preserve Chinese characters).
        """
    path = Path(file_path)

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=ensure_ascii)
        f.write("\n")