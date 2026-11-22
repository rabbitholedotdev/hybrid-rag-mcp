from eval.bootstrap_ci import metric_value


def test_metric_value_plana_e_aninhada():
    x = {"primary_recall": 1.0, "faithfulness": {"score": 0.7}, "numbers": {"rate": 0.2}}
    assert metric_value(x, "primary_recall") == 1.0
    assert metric_value(x, "faithfulness") == 0.7
    assert metric_value(x, "numbers_hallucination_rate") == 0.2
