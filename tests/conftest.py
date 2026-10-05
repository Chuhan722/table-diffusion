"""测试会话级环境准备。

cuBLAS 在 torch.use_deterministic_algorithms(True) 下要求进程级
CUBLAS_WORKSPACE_CONFIG（须在 CUDA 上下文初始化前设置），否则相关
CUDA 测试直接 RuntimeError。conftest 在任何测试模块导入 torch 之前
执行，因此在这里 setdefault 是唯一可靠的进程内注入点；已有外部值
时不覆盖。
"""

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
