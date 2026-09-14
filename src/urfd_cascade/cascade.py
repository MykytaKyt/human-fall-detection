"""Full two-cascade pipeline: kinematic filter -> dynamic router -> (optional)
semantic VLM -> Dempster-Shafer fusion -> decision. Wires together
ddl_features, dst, and vlm into the architecture described in the paper.
"""
from __future__ import annotations

from dataclasses import dataclass

from .dst import combine, decide, kinematic_evidence, pignistic, semantic_evidence
from .vlm import VLMClient


@dataclass
class CascadeResult:
    decision: str  # "A" (anomaly) or "N" (normal)
    routed_to_vlm: bool
    risk: float
    m_kinematic: dict
    m_semantic: dict | None
    conflict_k: float | None
    betp: dict


def routing_risk(p_error: float, cost_vlm: float, lam: float = 0.5) -> float:
    """Dynamic-routing risk function (Eq. eq:risk):
    R(x, VLM) = (1-lambda)*P(Error|YOLO,x) - lambda*Cost(VLM).
    Call the VLM iff R > 0.
    """
    return (1.0 - lam) * p_error - lam * cost_vlm


def run_cascade(
    lyapunov_v: float,
    stability_gamma: float,
    mos: float,
    image_path: str | None = None,
    vlm: VLMClient | None = None,
    cost_vlm: float = 0.1,
    lam: float = 0.5,
) -> CascadeResult:
    """Run the full cascade for one frame.

    `lyapunov_v`, `stability_gamma`, `mos` come from the kinematic filter
    (Sec. 3.1 of the paper). If `image_path`/`vlm` are given and the
    router decides the risk is high enough, the frame is sent to the VLM
    and evidence is fused via Dempster-Shafer; otherwise the kinematic
    evidence alone (via the pignistic transform) determines the decision
    -- this is the "fast path, no VLM/cloud call" branch of Fig. 1.
    """
    m_y = kinematic_evidence(lyapunov_v, stability_gamma, mos)

    # P(Error | YOLO, x): the calibrated uncertainty of the first-level
    # classifier is exactly the ignorance mass m_Y(Theta) -- no separate
    # calibration model is introduced here (ponytail: this *is* the
    # Gatekeeper-style calibration signal described in the paper; a
    # learned calibrator is future work, not required by the equations).
    risk = routing_risk(p_error=m_y["AN"], cost_vlm=cost_vlm, lam=lam)
    route = risk > 0 and image_path is not None and vlm is not None

    if not route:
        betp = pignistic(m_y)
        return CascadeResult(decide(betp), False, risk, m_y, None, None, betp)

    judgement = vlm.judge_image(image_path)
    m_v = semantic_evidence(judgement.p_anomaly, judgement.entropy)
    fused, k = combine(m_y, m_v)
    betp = pignistic(fused)
    return CascadeResult(decide(betp), True, risk, m_y, m_v, k, betp)
