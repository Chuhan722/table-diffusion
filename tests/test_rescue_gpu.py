"""救援打分 GPU 路径测试，与 CPU 打分全指标对拍，无卡环境自动跳过。"""
import numpy as np
import pytest

cp = pytest.importorskip("cupy")
try:
    if cp.cuda.runtime.getDeviceCount() < 1:
        pytest.skip("无可用 GPU", allow_module_level=True)
    _ = cp.zeros(2).sum()
except Exception:
    pytest.skip("GPU 运行时不可用", allow_module_level=True)

from test_batchkernel import _scene  # noqa: E402
from test_rescue import BUDGET  # noqa: E402

from resevo.dataset import target_from_specs  # noqa: E402
from resevo.pairing import build_rescue_menu  # noqa: E402


def test_rescue_gpu_menu_matches_cpu():
    """GPU 打分的救援菜单与 CPU 版同抽行同配对同菜单，增益同式。"""
    registry, ids, weights, _ = _scene(6)
    y = target_from_specs(registry.specs)
    cpu = build_rescue_menu(
        registry, ids, y, weights, np.random.default_rng(17), 12, BUDGET,
    )
    gpu = build_rescue_menu(
        registry, ids, y, weights, np.random.default_rng(17), 12, BUDGET,
        use_gpu=True,
    )
    np.testing.assert_array_equal(cpu[1], gpu[1])  # 抽中行小编号
    np.testing.assert_array_equal(cpu[2], gpu[2])  # 全表行下标
    assert len(cpu[4]) == len(gpu[4])
    for sc, sg in zip(cpu[4], gpu[4]):
        assert sc.rows == sg.rows, "配对结果必须一致"
        assert sc.outcomes == sg.outcomes, "菜单动作必须一致"
