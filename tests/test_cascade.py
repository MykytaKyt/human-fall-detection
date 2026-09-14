from urfd_cascade.cascade import routing_risk, run_cascade
from urfd_cascade.vlm import SemanticJudgement


class FakeVLM:
    """Stand-in for VLMClient in tests -- no network/LM Studio needed."""

    def __init__(self, p_anomaly: float, entropy: float):
        self._judgement = SemanticJudgement(p_anomaly, entropy, "test")

    def judge_image(self, image_path: str) -> SemanticJudgement:
        return self._judgement


def test_routing_risk_sign():
    # High error probability, low cost -> positive risk -> should call VLM.
    assert routing_risk(p_error=0.8, cost_vlm=0.1) > 0
    # Low error probability, high cost -> negative risk -> should not call.
    assert routing_risk(p_error=0.1, cost_vlm=0.8) < 0


def test_cascade_confident_kinematics_skips_vlm():
    # Very stable state (large positive MoS -> m_N close to 0 is wrong;
    # exp(-mos) shrinks as mos grows, so m_N -> 0 too. What actually
    # yields low ignorance is a *small*, clearly-positive MoS with V(x)
    # far below gamma: m_A=0, m_N=exp(-mos) close to 1, m_AN close to 0.
    result = run_cascade(lyapunov_v=0.0, stability_gamma=1.0, mos=0.01,
                          image_path="unused.png", vlm=FakeVLM(0.9, 0.0))
    assert result.routed_to_vlm is False
    assert result.decision == "N"


def test_cascade_no_vlm_available_uses_fast_path():
    # No image/client provided at all -> must not attempt to route.
    result = run_cascade(lyapunov_v=5.0, stability_gamma=1.0, mos=-0.5)
    assert result.routed_to_vlm is False
    assert result.m_semantic is None


def test_cascade_ambiguous_kinematics_routes_to_vlm_and_fuses():
    # Ambiguous kinematic state (near threshold) -> high ignorance -> risk>0
    # -> VLM is consulted and evidence is fused.
    result = run_cascade(lyapunov_v=1.0, stability_gamma=1.0, mos=0.0,
                          image_path="frame.png", vlm=FakeVLM(0.85, 0.1))
    assert result.routed_to_vlm is True
    assert result.m_semantic is not None
    assert result.conflict_k is not None
    assert result.decision in ("A", "N")


def test_cascade_vlm_disagreement_produces_conflict():
    # Kinematics leans anomaly, VLM strongly says normal (sports scenario).
    result = run_cascade(lyapunov_v=3.0, stability_gamma=1.0, mos=-0.2,
                          image_path="frame.png", vlm=FakeVLM(p_anomaly=0.05, entropy=0.0))
    assert result.routed_to_vlm is True
    assert result.conflict_k > 0.0
