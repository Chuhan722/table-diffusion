"""为同一套第 6B-1 五段管线选择冻结协议模块。"""

from __future__ import annotations

import importlib
import os
from types import ModuleType


SELECTOR_ENV = "ISSUE53_STAGE6B1_PROTOCOL"
_MODULES = {
    "stage6b1a": "issue53_stage6b1_protocol",
    "stage6b1b": "issue53_stage6b1b_protocol",
    "stage6b1b_batched": "issue53_stage6b1b_batched_protocol",
}


def load_protocol() -> ModuleType:
    selector = os.environ.get(SELECTOR_ENV, "stage6b1a")
    if selector not in _MODULES:
        raise RuntimeError(
            f"{SELECTOR_ENV} 只允许 stage6b1a、stage6b1b 或 "
            "stage6b1b_batched"
        )
    prefix = "scripts." if __package__ else ""
    return importlib.import_module(prefix + _MODULES[selector])
