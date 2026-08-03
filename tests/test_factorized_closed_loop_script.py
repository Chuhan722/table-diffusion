"""因子 Gibbs 标准闭环实验脚本的离线评价与配对汇总测试。"""

import pandas as pd
import pytest

import scripts.compare_factorized_gibbs_closed_loop as experiment
from table_diffevo.marginals import load_marginals
from table_diffevo.queries import load_queries


def _inputs():
    marginals = load_marginals(str(experiment.MARGINALS_PATH))
    queries = load_queries(str(experiment.QUERY_PATH))
    reference = pd.read_csv(experiment.REAL_DATA_PATH)
    return reference, queries, marginals


def test_reference_self_comparison_has_zero_offline_error():
    reference, queries, marginals = _inputs()
    domains = experiment._discretization_domains(marginals)
    measured = experiment._measured_cell_keys(
        queries, marginals, order=3
    )

    metrics = experiment._offline_metrics(
        reference,
        reference.copy(),
        marginals,
        domains,
        measured,
    )

    assert len(measured) == 5
    assert metrics["unmeasured_3way"]["n_queries"] == 5051
    assert metrics["unmeasured_4way"]["n_queries"] == 30450
    assert metrics["unmeasured_3way"]["mean"] == 0.0
    assert metrics["unmeasured_4way"]["mean"] == 0.0
    for name in ("raw_joint", "binned_joint"):
        assert metrics[name]["tvd"] == 0.0
        assert metrics[name]["missing_reference_mass"] == 0.0
        assert metrics[name]["novel_synthetic_mass"] == 0.0


@pytest.mark.parametrize(
    "attribute,value,message",
    [
        ("age", 101, "分箱范围外"),
        ("region", "unknown", "schema 外取值"),
    ],
)
def test_discretization_rejects_values_outside_public_domains(
    attribute, value, message
):
    reference, _, marginals = _inputs()
    invalid = reference.iloc[[0]].copy()
    invalid.loc[invalid.index[0], attribute] = value

    with pytest.raises(ValueError, match=message):
        experiment._discretize(invalid, marginals)


def test_paired_summary_respects_metric_direction():
    baseline = [{"metric": 3.0}, {"metric": 5.0}, {"metric": 7.0}]
    candidate = [{"metric": 2.0}, {"metric": 5.0}, {"metric": 9.0}]

    lower = experiment._paired(
        candidate, baseline, "metric", lower_is_better=True
    )
    higher = experiment._paired(
        candidate, baseline, "metric", lower_is_better=False
    )

    assert lower["wins"] == 1
    assert lower["ties"] == 1
    assert lower["losses"] == 1
    assert higher["wins"] == 1
    assert higher["ties"] == 1
    assert higher["losses"] == 1
    assert lower["mean_difference"] == pytest.approx(1.0 / 3.0)
    assert lower["difference_95pct_t_interval"][0] < 0.0
    assert lower["difference_95pct_t_interval"][1] > 0.0


def test_paired_summary_serializes_constant_nonzero_difference():
    baseline = [{"metric": 1.0}, {"metric": 2.0}, {"metric": 3.0}]
    candidate = [{"metric": 2.0}, {"metric": 3.0}, {"metric": 4.0}]

    result = experiment._paired(
        candidate, baseline, "metric", lower_is_better=True
    )

    assert result["mean_difference"] == 1.0
    assert result["difference_95pct_t_interval"] == [1.0, 1.0]
    assert result["paired_t"] is None
    assert result["paired_p"] == 0.0
    assert result["zero_variance_nonzero_difference"] is True


def test_aggregate_rejects_empty_rows():
    with pytest.raises(ValueError, match="没有可汇总记录"):
        experiment._aggregate([], "metric")


def test_output_directory_must_be_new_or_empty(tmp_path):
    new_directory = tmp_path / "new"
    assert experiment._prepare_output_directory(new_directory) == new_directory
    assert new_directory.is_dir()

    empty_directory = tmp_path / "empty"
    empty_directory.mkdir()
    assert experiment._prepare_output_directory(empty_directory) == empty_directory

    (empty_directory / "old.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="不是空目录"):
        experiment._prepare_output_directory(empty_directory)

    existing_file = tmp_path / "file"
    existing_file.write_text("old", encoding="utf-8")
    with pytest.raises(FileExistsError, match="不是空目录"):
        experiment._prepare_output_directory(existing_file)
