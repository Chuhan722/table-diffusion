"""
主循环接受判据接线测试（Issue #33 阶段 1）

锚定三件事：
1. acceptance_rule=None（默认）与改动前的主循环逐轨迹等价——钉死 loss/accept
   轨迹与表哈希，值取自改动前的提交 158787b 实测。
2. 显式规则 'A0'/'A1' 确实接线到主循环（改变 eps 必须改变轨迹）。
3. 参数校验 fail-closed：拼错的规则名、非法 eps 立即报错。

注意：本文件只测"接线"，判据本身的语义在 test_acceptance.py 里测。
"""
import hashlib

import numpy as np
import pytest

from table_diffevo.evolution import run_evolution
from table_diffevo.schema import Schema, AttributeBlock


def make_toy_schema():
    return Schema([
        AttributeBlock(name="age", type="numeric", description="年龄",
                       range=[18, 100]),
        AttributeBlock(name="edu", type="categorical", description="学历",
                       values=["low", "mid", "high"]),
        AttributeBlock(name="job", type="categorical", description="职业",
                       values=["a", "b", "c"]),
    ])


def make_toy_queries():
    return [
        {"conditions": [{"attribute": "edu", "operator": "==", "value": "high"}]},
        {"conditions": [{"attribute": "job", "operator": "==", "value": "a"}]},
        {"conditions": [{"attribute": "age", "operator": ">=", "value": 50}]},
    ]


TOY_TARGET = np.array([30, 40, 50])


def run_toy(**kwargs):
    kwargs.setdefault("n_records", 100)
    kwargs.setdefault("n_rounds", 20)
    kwargs.setdefault("seed", 0)
    kwargs.setdefault("log_every", 1000)
    return run_evolution(
        TOY_TARGET, make_toy_queries(), make_toy_schema(), **kwargs
    )


def table_sha256(df) -> str:
    return hashlib.sha256(df.to_csv(index=False).encode()).hexdigest()


# ---------------------------------------------------------------------------
# 平局带判别力夹具
#
# TOY 夹具测不出平局带：实测其 20 次尝试里 sign(ΔL1) 与 sign(ΔQ) 从不分歧，
# 唯一的 ΔL1==0 那次 ΔQ 也为 0（两条判据都拒绝），故 A1 的接受序列与 eps_L1
# 无关——拿它测 eps_L1 灵敏度会得到永真的断言。
#
# 下面这个夹具用异质量级的 target（前 9 个查询计数量级远大于后 9 个）制造
# L1（等权绝对）与 Q（平方，被大计数支配）的分歧：实测 40 次尝试中有 10 次
# sign(ΔL1) != sign(ΔQ)，含 ΔL1=+1 个步长而 ΔQ<0 的情形——正是平局带裁决的对象。
# ---------------------------------------------------------------------------
HETERO_N_RECORDS = 100


def make_hetero_queries():
    qs = []
    for v in ["low", "mid", "high"]:
        qs.append({"conditions": [{"attribute": "edu", "operator": "==", "value": v}]})
    for v in ["a", "b", "c"]:
        qs.append({"conditions": [{"attribute": "job", "operator": "==", "value": v}]})
    for t in [30, 50, 70]:
        qs.append({"conditions": [{"attribute": "age", "operator": ">=", "value": t}]})
    for v in ["low", "mid", "high"]:
        for j in ["a", "b", "c"]:
            qs.append({
                "conditions": [
                    {"attribute": "edu", "operator": "==", "value": v},
                    {"attribute": "job", "operator": "==", "value": j},
                ]
            })
    return qs


def make_hetero_target(n_queries):
    rng = np.random.default_rng(7)
    return np.concatenate([
        rng.integers(5, 60, size=9),
        rng.integers(0, 15, size=n_queries - 9),
    ]).astype(float)


def run_hetero(**kwargs):
    queries = make_hetero_queries()
    kwargs.setdefault("n_records", HETERO_N_RECORDS)
    kwargs.setdefault("n_rounds", 40)
    kwargs.setdefault("seed", 0)
    kwargs.setdefault("log_every", 1000)
    return run_evolution(
        make_hetero_target(len(queries)), queries, make_hetero_schema(), **kwargs
    )


def make_hetero_schema():
    return make_toy_schema()


def hetero_step() -> float:
    """ΔL1 的最小非零变化量（步长）= 1/(m * N)"""
    return 1.0 / (len(make_hetero_queries()) * HETERO_N_RECORDS)


# 改动前（提交 158787b）用同一配置实测的默认路径轨迹。这些值是回归锚点：
# 一旦默认路径的判据被无意改动，下面的断言会立刻失败。
PRE_CHANGE_SHA256 = (
    "0b70bd124a737c50c4f4cb8f9f0a7b83bce269141cc35c46f3411148bb6af386"
)
PRE_CHANGE_LOSS = [
    223.0, 198.0, 196.0, 148.5, 90.5, 79.0, 57.5, 46.5, 30.5, 16.5,
    9.0, 3.0, 2.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
]
PRE_CHANGE_ACCEPT = [
    True, True, True, True, True, True, True, True, True, True,
    True, True, True, False, False, False, False, False, True, False,
]


class TestDefaultPathUnchanged:
    """默认路径必须与改动前逐轨迹等价"""

    def test_default_is_none(self):
        """不传 acceptance_rule 时默认为 None，即主循环历史判据"""
        _, diag = run_toy()
        assert diag["params"]["acceptance_rule"] is None

    def test_default_trajectory_matches_pre_change(self):
        """默认路径的 loss/accept 轨迹与改动前实测值逐项一致"""
        best_S, diag = run_toy()
        assert [float(x) for x in diag["loss_history"]] == PRE_CHANGE_LOSS
        assert [bool(x) for x in diag["accept_history"]] == PRE_CHANGE_ACCEPT
        assert diag["rounds_run"] == 20
        assert float(diag["best_loss"]) == 1.0

    def test_default_table_hash_matches_pre_change(self):
        """默认路径产出的表与改动前逐位一致（哈希锚定）"""
        best_S, _ = run_toy()
        assert table_sha256(best_S) == PRE_CHANGE_SHA256

    def test_default_accepts_tie_non_strict(self):
        """默认路径保留非严格口径：平局与 tol 内的微小恶化仍被接受。

        这正是它与严格判据 A0/A1 的分界，也是历史 baseline 依赖的口径。
        """
        # 放宽 tol 会让更多"轻微恶化"的候选被接受，接受数必须单调不减
        _, diag_tight = run_toy(tol=1e-9)
        _, diag_loose = run_toy(tol=50.0)
        assert sum(diag_loose["accept_history"]) >= sum(
            diag_tight["accept_history"]
        )


class TestExplicitRulesWired:
    """显式规则确实接线到主循环"""

    def test_a0_wired_via_eps_q(self):
        """eps_Q 极大时 A0 拒绝一切候选——证明 A0 真的参与了判定"""
        _, diag = run_toy(acceptance_rule="A0", eps_Q=1e9)
        assert sum(diag["accept_history"]) == 0

    def test_a1_wired_via_eps_l1(self):
        """A1 接线：eps_L1 极大时全部落入平局带，由 Q 裁决"""
        _, diag = run_toy(acceptance_rule="A1", eps_L1=1e9)
        # 平局带内仅 delta_Q < -eps_Q 才接受，故接受数由 Q 决定而非 L1
        assert sum(diag["accept_history"]) > 0
        _, diag_blocked = run_toy(
            acceptance_rule="A1", eps_L1=1e9, eps_Q=1e9
        )
        assert sum(diag_blocked["accept_history"]) == 0

    def test_rule_recorded_in_params(self):
        """规则名与 eps 落到 params，便于复现"""
        _, diag = run_toy(acceptance_rule="A1", eps_L1=1e-5, eps_Q=0.25)
        assert diag["params"]["acceptance_rule"] == "A1"
        assert diag["params"]["eps_L1"] == 1e-5
        assert diag["params"]["eps_Q"] == 0.25

    def test_strict_rejects_tie_unlike_default(self):
        """严格口径拒绝平局，默认口径接受平局。

        用一个必然出现平局的构造：target 已被完美命中时，任何不改变计数的
        候选都是 delta_Q == 0。严格判据必须拒绝。
        """
        _, diag_strict = run_toy(acceptance_rule="A0", eps_Q=0.0, n_rounds=40)
        _, diag_default = run_toy(n_rounds=40)
        # 收敛后默认路径继续接受平局，严格路径不再接受
        assert sum(diag_strict["accept_history"]) <= sum(
            diag_default["accept_history"]
        )


class TestDeltaDiagnostics:
    """ΔL1 / ΔQ 无条件记录"""

    def test_delta_histories_present_on_default_path(self):
        """默认路径也记录 ΔL1/ΔQ——用于给 eps_L1 定标，不参与判定"""
        _, diag = run_toy()
        assert "delta_L1_history" in diag
        assert "delta_Q_history" in diag
        assert len(diag["delta_L1_history"]) == diag["rounds_run"]
        assert len(diag["delta_Q_history"]) == diag["rounds_run"]

    def test_delta_shapes_match_attempts(self):
        """每轮记录的 Δ 数与该轮候选尝试数一致"""
        _, diag = run_toy(max_retries=3)
        for dl, dq, n_attempt in zip(
            diag["delta_L1_history"],
            diag["delta_Q_history"],
            diag["proposal_attempts_history"],
        ):
            assert len(dl) == n_attempt
            assert len(dq) == n_attempt

    def test_delta_q_matches_loss_difference(self):
        """记录的 ΔQ 与主循环自己的 loss 差一致（不重复评价、不产生分歧）。

        loss_history[t] 是进入第 t 轮时的 loss（在候选评价之前 append），
        因此被接受的第 t 轮满足 ΔQ == loss_history[t+1] - loss_history[t]。
        """
        _, diag = run_toy()
        loss_hist = diag["loss_history"]
        accept_hist = diag["accept_history"]
        checked = 0
        for t in range(len(loss_hist) - 1):
            if accept_hist[t] and len(diag["delta_Q_history"][t]) == 1:
                observed = float(loss_hist[t + 1]) - float(loss_hist[t])
                assert diag["delta_Q_history"][t][0] == pytest.approx(
                    observed, abs=1e-9
                )
                checked += 1
        # 断言确实比对过若干轮，避免循环空转让本测试变成永真
        assert checked > 0


class TestValidationFailClosed:
    """参数校验：拒绝而非静默回落"""

    @pytest.mark.parametrize("bad", ["A2", "a0", "", "A0 ", "None", "Q"])
    def test_unknown_rule_raises(self, bad):
        """未支持的规则名必须报错，不得静默走默认路径"""
        with pytest.raises(ValueError, match="acceptance_rule 必须是"):
            run_toy(acceptance_rule=bad)

    @pytest.mark.parametrize("bad", [-1e-9, -1.0, float("nan"), float("inf")])
    def test_invalid_eps_l1_raises(self, bad):
        with pytest.raises(ValueError, match="eps_L1 必须是非负有限实数"):
            run_toy(acceptance_rule="A1", eps_L1=bad)

    @pytest.mark.parametrize("bad", [-1e-9, float("nan"), float("-inf")])
    def test_invalid_eps_q_raises(self, bad):
        with pytest.raises(ValueError, match="eps_Q 必须是非负有限实数"):
            run_toy(acceptance_rule="A0", eps_Q=bad)

    def test_bool_eps_rejected(self):
        """bool 是 int 的子类，必须显式拒绝，否则 True 会被当成 1.0"""
        with pytest.raises(ValueError, match="eps_L1 必须是非负有限实数"):
            run_toy(acceptance_rule="A1", eps_L1=True)

    def test_eps_validated_even_on_default_path(self):
        """默认路径也校验 eps——非法值不能因为"反正没用到"而放过"""
        with pytest.raises(ValueError, match="eps_L1 必须是非负有限实数"):
            run_toy(eps_L1=-1.0)

    def test_zero_eps_accepted(self):
        """0 是合法值（严格改善口径的默认）"""
        _, diag = run_toy(acceptance_rule="A0", eps_L1=0.0, eps_Q=0.0)
        assert diag["params"]["eps_Q"] == 0.0


class TestEpsL1TieBand:
    """eps_L1 平局带的语义与两端退化

    ΔL1 的步长 = 1/(m * n_records)：normalized_l1 的分子是整数计数之差，故 ΔL1
    只能以此为单位跳变。对照实验取 eps_L1=0（平局带 = 「ΔL1 恰好为 0」），依据
    就是这个步长——0 与最小非零变化之间隔着一整个台阶，没有需要容差吸收的浮点
    毛刺。下面前两条钉住这个算术依据，后两条钉住取非零值时的两端退化。
    """

    def test_step_is_min_nonzero_delta_l1(self):
        """实测锚定步长：最小非零 |ΔL1| 恰为 1/(m*N)，且都是它的整数倍"""
        _, diag = run_hetero(acceptance_rule=None)
        magnitudes = {
            abs(d)
            for round_deltas in diag["delta_L1_history"]
            for d in round_deltas
            if d != 0.0
        }
        assert magnitudes, "夹具没产生任何非零 ΔL1，本测试会变成永真"
        assert min(magnitudes) == pytest.approx(hetero_step(), rel=1e-9)
        for mag in magnitudes:
            ratio = mag / hetero_step()
            assert ratio == pytest.approx(round(ratio), abs=1e-6)

    def test_exact_ties_occur_and_are_decided_by_q(self):
        """eps_L1=0 时 Q 那一路不是死代码：确有 ΔL1 恰好为 0 的候选转交 Q。

        这条是 eps_L1=0 的正当性依据——若精确平局从不发生，A1 就等同于纯 L1
        判据，「L1 主判 + Q 平局判」这个设计在本夹具上便无从体现。
        """
        _, diag = run_hetero(acceptance_rule="A1", eps_L1=0.0)
        exact_ties = [
            d
            for round_deltas in diag["delta_L1_history"]
            for d in round_deltas
            if d == 0.0
        ]
        assert exact_ties, "夹具没产生精确平局，Q 平局判分支未被覆盖"

    def test_sub_step_band_degenerates_to_zero(self):
        """窄于一个步长的平局带等价于 eps_L1=0。

        整数计数下 |ΔL1| 不可能落在 (0, 一个步长) 区间内，故半宽 0.5 个步长的
        带子永远接不住任何东西。这也是「eps_L1 不能跨数据集直接沿用」的机制：
        同一个绝对值在 m*N 更小的数据集上可能整个落进这一端。
        """
        best_half, diag_half = run_hetero(
            acceptance_rule="A1", eps_L1=0.5 * hetero_step()
        )
        best_zero, diag_zero = run_hetero(acceptance_rule="A1", eps_L1=0.0)
        assert table_sha256(best_half) == table_sha256(best_zero)
        assert diag_half["accept_history"] == diag_zero["accept_history"]

    def test_huge_band_degenerates_to_a0(self):
        """宽于 L1 总量的平局带让全部候选转交 Q → A1 在算术上等于 A0。

        预注册的 eps_L1=0.02 在 nltcs 上就落在这一端（覆盖 100% 候选），
        这是它必须被替换的原因。
        """
        best_big, diag_big = run_hetero(acceptance_rule="A1", eps_L1=1e6)
        best_a0, diag_a0 = run_hetero(acceptance_rule="A0")
        assert table_sha256(best_big) == table_sha256(best_a0)
        assert diag_big["accept_history"] == diag_a0["accept_history"]

    def test_nonzero_band_changes_trajectory(self):
        """平局带确实参与判定：0 与一个步长的轨迹不同。

        这条在 TOY 夹具上会永真（那里 sign(ΔL1) 与 sign(ΔQ) 从不分歧），
        故必须用 hetero 夹具。
        """
        best_0, diag_0 = run_hetero(acceptance_rule="A1", eps_L1=0.0)
        best_1, diag_1 = run_hetero(
            acceptance_rule="A1", eps_L1=1.0 * hetero_step()
        )
        assert table_sha256(best_0) != table_sha256(best_1)
        assert diag_0["accept_history"] != diag_1["accept_history"]

    def test_eps_l1_and_step_recorded_in_params(self):
        """params 同时留档 eps_L1 与 ΔL1 步长，事后才判得出带宽的松紧"""
        _, diag = run_hetero(acceptance_rule="A1", eps_L1=0.0)
        params = diag["params"]
        assert params["eps_L1"] == 0.0
        assert params["delta_L1_step"] == pytest.approx(hetero_step())
