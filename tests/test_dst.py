import pytest

from urfd_cascade.dst import (
    combine,
    combine_dempster,
    combine_yager,
    decide,
    kinematic_evidence,
    pignistic,
    semantic_evidence,
)


def test_dempster_agreement_reinforces_confidence():
    # Both sources strongly agree on anomaly -> low conflict, high fused mass.
    m1 = {"A": 0.9, "N": 0.05, "AN": 0.05}
    m2 = {"A": 0.85, "N": 0.05, "AN": 0.10}
    fused, k = combine_dempster(m1, m2)
    assert k < 0.1
    assert fused["A"] > 0.9
    assert decide(pignistic(fused)) == "A"


def test_dempster_masses_sum_to_one():
    m1 = {"A": 0.6, "N": 0.1, "AN": 0.3}
    m2 = {"A": 0.2, "N": 0.5, "AN": 0.3}
    fused, _ = combine_dempster(m1, m2)
    assert fused["A"] + fused["N"] + fused["AN"] == pytest.approx(1.0)


def test_dempster_raises_on_total_conflict():
    m1 = {"A": 1.0, "N": 0.0, "AN": 0.0}
    m2 = {"A": 0.0, "N": 1.0, "AN": 0.0}
    with pytest.raises(ValueError):
        combine_dempster(m1, m2)


def test_yager_redirects_conflict_to_ignorance():
    # Sports scenario from the paper's Table: kinematics says fall,
    # semantics says normal play -> high conflict, Yager should push
    # mass into AN (ignorance) rather than force a confident decision.
    m_y = {"A": 0.7, "N": 0.1, "AN": 0.2}
    m_v = {"A": 0.1, "N": 0.8, "AN": 0.1}
    fused, k = combine(m_y, m_v, yager_threshold=0.5)
    assert k > 0.5  # this scenario is indeed high-conflict
    assert fused["AN"] > m_y["AN"]  # ignorance grew, no artificial consensus


def test_combine_uses_dempster_below_threshold():
    m1 = {"A": 0.9, "N": 0.05, "AN": 0.05}
    m2 = {"A": 0.85, "N": 0.05, "AN": 0.10}
    fused, k = combine(m1, m2, yager_threshold=0.6)
    expected, _ = combine_dempster(m1, m2)
    assert fused == pytest.approx(expected)


def test_pignistic_splits_ignorance_evenly():
    m = {"A": 0.3, "N": 0.3, "AN": 0.4}
    betp = pignistic(m)
    assert betp["A"] == pytest.approx(0.5)
    assert betp["N"] == pytest.approx(0.5)
    assert betp["A"] + betp["N"] == pytest.approx(1.0)


def test_decide_ties_favour_anomaly_for_safety():
    assert decide({"A": 0.5, "N": 0.5}) == "A"


def test_kinematic_evidence_stable_state_favours_normal():
    # V(x) well below threshold gamma, positive MoS -> stable, m_N high.
    m = kinematic_evidence(lyapunov_v=0.1, stability_gamma=1.0, mos=0.5)
    assert m["N"] > m["A"]
    assert m["A"] + m["N"] + m["AN"] == pytest.approx(1.0)


def test_kinematic_evidence_unstable_state_favours_anomaly():
    # V(x) far above gamma, negative MoS -> unstable, m_A high.
    m = kinematic_evidence(lyapunov_v=5.0, stability_gamma=1.0, mos=-0.3)
    assert m["A"] > m["N"]


def test_kinematic_evidence_occlusion_increases_ignorance():
    # Neither condition strongly triggers -> most mass goes to AN.
    m = kinematic_evidence(lyapunov_v=1.0, stability_gamma=1.0, mos=0.0)
    assert m["AN"] > 0.5


def test_semantic_evidence_high_entropy_increases_ignorance():
    confident = semantic_evidence(p_anomaly=0.9, entropy=0.0)
    uncertain = semantic_evidence(p_anomaly=0.9, entropy=0.9)
    assert uncertain["AN"] > confident["AN"]


def test_semantic_evidence_masses_sum_to_one():
    m = semantic_evidence(p_anomaly=0.3, entropy=0.4)
    assert m["A"] + m["N"] + m["AN"] == pytest.approx(1.0)
