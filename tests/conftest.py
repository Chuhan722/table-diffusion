import sys
from pathlib import Path

# 让 tests 零安装即可 import resevo 与 tests/reference 下的对拍参考实现
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests" / "reference"))
