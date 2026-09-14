"""Dempster-Shafer evidence fusion for the binary frame of discernment
Theta = {A, N} (A = anomaly/fall, N = normal), as used in the paper's
DST cascade. Belief masses are dicts {'A': m(A), 'N': m(N), 'AN': m(Theta)}
summing to 1; m(empty)=0 is implied (never represented).

ponytail: hardcoded to the binary frame instead of a generic power-set
engine (subsets, intersections, etc.) -- the paper only ever uses
Theta={theta_A, theta_N}, so a generic DST library would be unused
generality. If a multi-class frame is ever needed (paper's own "future
work"), replace this module with one, e.g. pyds.
"""
from __future__ import annotations

BBA = dict[str, float]


def combine_dempster(m1: BBA, m2: BBA) -> tuple[BBA, float]:
    """Dempster's rule of combination (Eq. eq:dempster, eq:conflict).

    Returns (combined_mass, conflict_K). Raises ValueError if K == 1
    (total conflict; caller should fall back to combine_yager, which
    handles this case by design -- see Sec. 5.4 of the paper).
    """
    k = m1["A"] * m2["N"] + m1["N"] * m2["A"]
    if k >= 1.0:
        raise ValueError("total conflict (K=1): use combine_yager instead")
    norm = 1.0 / (1.0 - k)
    a = norm * (m1["A"] * m2["A"] + m1["A"] * m2["AN"] + m1["AN"] * m2["A"])
    n = norm * (m1["N"] * m2["N"] + m1["N"] * m2["AN"] + m1["AN"] * m2["N"])
    an = norm * (m1["AN"] * m2["AN"])
    return {"A": a, "N": n, "AN": an}, k


def combine_yager(m1: BBA, m2: BBA, k: float) -> BBA:
    """Yager's rule (Eq. eq:yager): redirects all conflict mass K into
    Theta (ignorance) instead of Dempster's normalisation, used when K
    exceeds a critical threshold to avoid Dempster's counter-intuitive
    behaviour under high conflict.
    """
    a = m1["A"] * m2["A"] + m1["A"] * m2["AN"] + m1["AN"] * m2["A"]
    n = m1["N"] * m2["N"] + m1["N"] * m2["AN"] + m1["AN"] * m2["N"]
    an = m1["AN"] * m2["AN"] + k
    return {"A": a, "N": n, "AN": an}


def combine(m1: BBA, m2: BBA, yager_threshold: float = 0.6) -> tuple[BBA, float]:
    """Combine evidence, switching to Yager's rule when conflict is high
    (as described in the paper: "if K exceeds a critical threshold").
    Returns (combined_mass, conflict_K).
    """
    k = m1["A"] * m2["N"] + m1["N"] * m2["A"]
    if k >= yager_threshold:
        return combine_yager(m1, m2, k), k
    return combine_dempster(m1, m2)


def pignistic(m: BBA) -> dict[str, float]:
    """Pignistic transform (Eq. eq:pignistic) for the binary frame:
    BetP(A) = m(A) + m(Theta)/2, BetP(N) = m(N) + m(Theta)/2.
    """
    half_an = m["AN"] / 2.0
    return {"A": m["A"] + half_an, "N": m["N"] + half_an}


def decide(betp: dict[str, float]) -> str:
    """Final decision (Eq. eq:decision): argmax over BetP."""
    return "A" if betp["A"] >= betp["N"] else "N"


def kinematic_evidence(
    lyapunov_v: float,
    stability_gamma: float,
    mos: float,
    lambda1: float = 1.0,
    lambda2: float = 1.0,
) -> BBA:
    """Kinematic BBA m_Y from the LTC/DDL stage (paper's kinematic-evidence
    equations, just above Eq. eq:dempster). `lyapunov_v` is V(x) from the
    LTC stability manifold check; `mos` is the Margin of Stability.

    ponytail: mass not assigned to A or N is put in AN (ignorance), which
    is exactly what the paper describes for occlusion ("redirects mass to
    Theta") -- no separate occlusion branch needed, it falls out of the
    normalisation below.
    """
    import math

    m_a = 1.0 - math.exp(-lambda1 * max(0.0, lyapunov_v - stability_gamma))
    m_n = math.exp(-lambda2 * mos) if mos > 0 else 0.0
    m_a, m_n = min(m_a, 1.0), min(m_n, 1.0 - m_a)
    return {"A": m_a, "N": m_n, "AN": 1.0 - m_a - m_n}


def semantic_evidence(p_anomaly: float, entropy: float) -> BBA:
    """Semantic BBA m_V from the VLM stage: m_V(theta_i) = p_VLM(theta_i)
    * (1 - f_alpha(H)) (paper's semantic-evidence equation). `entropy` is
    normalised to [0, 1] (0 = fully confident, 1 = maximum uncertainty);
    this plays the role of f_alpha(H) directly (f_alpha = identity).

    ponytail: f_alpha(H) = identity on normalised entropy is the simplest
    choice consistent with "uncertainty grows with entropy"; the paper
    does not fix a specific f_alpha, so no extra calibration curve is
    invented here.
    """
    confidence = max(0.0, 1.0 - entropy)
    m_a = p_anomaly * confidence
    m_n = (1.0 - p_anomaly) * confidence
    return {"A": m_a, "N": m_n, "AN": 1.0 - m_a - m_n}
