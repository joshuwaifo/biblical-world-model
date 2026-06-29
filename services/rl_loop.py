"""
Test-time RL Loop — Alberta Plan four-component agent.

Reference: arxiv 2208.11173 (Sutton, Bowling, Pilarski — "The Alberta Plan
for Canadian AI Research").  Figure 2: Perception → Reactive Policies →
Value Functions → Transition Model.

Mapping to this system
──────────────────────
Perception          embedder.encode()   — any stream → Biblical coordinate (dim from config)
Reactive Policy     oracle.next_action  — GNN attention as learned policy
                    + program_library.match()  — SOAR program retrieval shortcut
Value Function      value_fn.ValueFunction — 9-param TD(λ), updates every step
Transition Model    _imagine_next()     — frozen encoder + EMA interpolation
                    used for k-step lookahead in search_control()

Temporal uniformity (Alberta Plan principle): ALL four components run on
EVERY time step.  There are no "offline" training phases.  The BibleGAT
weights never change; the value function and program library grow.

Test-time compute (ARC Prize / SOAR inspiration):
  On each step, search_control() imagines k candidate next states using the
  frozen transition model and picks the action predicted to yield highest V.
  This is O(k) forward passes through the frozen embedder — cheap, effective.

NDEA search guidance:
  The program library IS the neuro-symbolic component: learned (value fn)
  heuristics + symbolic (program) sequences.  Synthesis happens when the
  RL agent discovers a high-reward trajectory; retrieval replaces planning
  overhead on familiar terrain.

Global singletons persist across navigate() calls in the same server process.
On first import the loop is cold.  It warms up as requests arrive.
"""

from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING

from config import W2V_VECTOR_SIZE
from services.value_fn      import ValueFunction
from services.program_library import ProgramLibrary, SYNTHESIS_THRESHOLD

if TYPE_CHECKING:
    from services import oracle as _oracle_t

# ---------------------------------------------------------------------------
# Global singletons — warm across requests
# ---------------------------------------------------------------------------
_vf      = ValueFunction(dim=W2V_VECTOR_SIZE, gamma=0.95, eta=0.05, lmbda=0.8)
_library = ProgramLibrary()

# trajectory buffer for the current navigate() call
# reset at start of each call via begin_episode()
_traj_states:    list[np.ndarray] = []
_traj_steps:     list[dict]       = []
_traj_td_errors: list[float]      = []


# ---------------------------------------------------------------------------
# Episode management
# ---------------------------------------------------------------------------

def begin_episode() -> None:
    """Call at the start of each navigate() call.  Clears per-episode buffers."""
    global _traj_states, _traj_steps, _traj_td_errors
    _vf.reset_trace()
    _traj_states    = []
    _traj_steps     = []
    _traj_td_errors = []


def end_episode(dominant_arc: str = "unknown") -> dict | None:
    """
    Call after the navigation loop ends.  Attempts SOAR-style program synthesis
    if the episode was sufficiently successful.

    Returns the new/updated ProgramEntry as a dict, or None if below threshold.
    """
    if len(_traj_states) < 2:
        return None

    alignments = [s.get("alignment", 0.0) for s in _traj_steps]
    mean_reward = float(np.mean(alignments)) if alignments else 0.0

    if mean_reward < SYNTHESIS_THRESHOLD:
        return None

    mean_state = np.mean(_traj_states, axis=0).astype(np.float32)
    entry = _library.synthesise(
        trajectory=_traj_steps,
        mean_state=mean_state,
        arc=dominant_arc,
        mean_reward=mean_reward,
    )
    if entry is None:
        return None
    return {"name": entry.name, "arc": entry.arc,
            "reward": round(entry.reward, 4), "n_uses": entry.n_uses}


# ---------------------------------------------------------------------------
# Policy selection (Reactive Policy component)
# ---------------------------------------------------------------------------

def select_action(
    current_vec:  np.ndarray,
    goal_vec:     np.ndarray,
    tool_list:    list[str],
    tool_vecs:    list[np.ndarray],
    step_num:     int,
    prior_content: list[dict],
) -> tuple[str, float, bool]:
    """
    Select next tool via the Alberta Plan reactive-policy hierarchy:

      1. Check program library — if a known successful program matches the
         current state, return its first remaining action (shortcut).
      2. Run oracle.next_action (GNN attention policy).
      3. Search control: for each candidate tool, imagine the next state
         using the frozen transition model and pick the one with highest V.
         (Only when multiple tools are available and step > 1.)

    Returns
    -------
    (tool_name, action_confidence, library_hit)
      library_hit = True when the program library provided the selection
    """
    from services import oracle

    # ── 1. Program library retrieval ──────────────────────────────────────
    lib_match = _library.match(current_vec)
    if lib_match is not None and step_num <= len(lib_match.sequence):
        action = lib_match.sequence[step_num - 1]
        tname  = action.get("tool", "web_search")
        if tname in tool_list:
            return tname, 0.95, True   # high confidence — proven program

    # ── 2. Oracle attention policy ────────────────────────────────────────
    if not tool_list:
        return "web_search", 0.0, False

    if len(tool_vecs) > 1:
        action_probs = oracle.next_action(current_vec, tool_vecs)
    else:
        action_probs = [1.0]

    # ── 3. Search control: lookahead V over candidate next states ─────────
    #    Only run when value function has enough signal (n_updates > 5)
    if _vf.n_updates > 5 and len(tool_vecs) > 1 and step_num > 1:
        scored = []
        for i, (tname, tvec) in enumerate(zip(tool_list, tool_vecs)):
            # Imagine next state: interpolation toward tool embedding
            imagined = _imagine_next(current_vec, tvec, alpha=0.6)
            # Composite: oracle prob * V(imagined_next)
            v_next = _vf.predict(imagined)
            # Normalise V to [0,1] roughly: sigmoid(v_next)
            v_gate = float(1.0 / (1.0 + np.exp(-v_next)))
            scored.append((action_probs[i] * v_gate, i))
        best_idx = max(scored, key=lambda x: x[0])[1]
    else:
        best_idx = int(np.argmax(action_probs))

    return tool_list[best_idx], float(action_probs[best_idx]), False


# ---------------------------------------------------------------------------
# TD(λ) update (Value Function component)
# ---------------------------------------------------------------------------

def update_value(
    s_t:       np.ndarray,
    reward:    float,
    s_t1:      np.ndarray,
    done:      bool = False,
    step_info: dict | None = None,
) -> float:
    """
    Update the value function via TD(λ) and buffer the step for synthesis.

    The reward signal is the oracle alignment score for this step — the same
    signal Silver & Sutton identify as "grounded" reward: produced by the
    environment (the Biblical manifold), not a human.

    Returns
    -------
    td_error: float
    """
    delta = _vf.update(s_t, reward, s_t1, done=done)
    _traj_states.append(s_t.copy())
    _traj_td_errors.append(delta)
    if step_info:
        _traj_steps.append(step_info)
    return delta


# ---------------------------------------------------------------------------
# Transition model (for search control imaginations)
# ---------------------------------------------------------------------------

def _imagine_next(s: np.ndarray, action_vec: np.ndarray,
                  alpha: float = 0.6) -> np.ndarray:
    """
    Predict the next state by interpolating current position toward action
    representation.  This is the frozen transition model — no parameters.

    T(s, a) ≈ normalise((1 - α)·s + α·embed(tool_description))
    """
    mixed = (1 - alpha) * s.astype(np.float32) + alpha * action_vec.astype(np.float32)
    norm  = np.linalg.norm(mixed)
    return mixed / (norm + 1e-9)


# ---------------------------------------------------------------------------
# Typological program search — manifold-grounded arc identification
# ---------------------------------------------------------------------------

def score_programs_for(
    vec:   np.ndarray,
    kind:  str   = "typological",
    gamma: float = 0.9,
    alpha: float = 0.6,
) -> list[tuple[str, float, str]]:
    """
    Search the program space for the best-matching typological arc.

    Two RL mechanisms combined — the same composite used in select_action():
      composite(program) = oracle_attention_weight × trajectory_return

    oracle_attention_weight : next_action(vec, pattern_triggers)
        The GNN's learned attention from the current position toward each arc's
        centroid region.  This IS the manifold's emergent guidance signal —
        trained over 38,927 verses to know which regions support which content.

    trajectory_return : Σ_i gamma^i · alignment_score(imagine_next(s_i, verse_i))
        Walk through the arc's key verse sequence.  At each step, interpolate
        toward the next canonical verse (frozen transition model) and score
        oracle reconstruction quality.  The arc whose narrative trajectory
        fits the query position best wins.

    Together: "which arc does the GNN attend to, confirmed by simulating the
    full narrative walk from the query position?"

    Returns
    -------
    [(arc_name, composite_score, program_name)] sorted descending by composite.
    Returns [] if oracle is not initialised or no typological programs exist.
    """
    from services import oracle

    programs = [e for e in _library._entries if e.kind == kind]
    if not programs or oracle._model is None:
        return []

    # Oracle attention prior over all pattern triggers
    triggers = [p.trigger for p in programs]
    try:
        attn_weights = oracle.next_action(vec, triggers)
    except Exception:
        attn_weights = [1.0 / len(programs)] * len(programs)

    # Trajectory simulation: imagine walking each arc's verse sequence
    results = []
    for prog, attn in zip(programs, attn_weights):
        traj_return = _evaluate_trajectory(vec, prog.sequence, oracle, gamma, alpha)
        composite   = float(attn) * traj_return
        results.append((prog.arc, composite, prog.name))

    return sorted(results, key=lambda x: x[1], reverse=True)


def _evaluate_trajectory(
    start_vec:     np.ndarray,
    sequence:      list[dict],
    oracle_module,
    gamma:         float = 0.9,
    alpha:         float = 0.6,
) -> float:
    """
    Simulate walking through a verse-trajectory and accumulate
    gamma-discounted oracle alignment rewards.

    At each step:
      1. Interpolate current position toward the arc's next canonical verse
         (frozen transition model — same as _imagine_next in tool selection).
      2. Score the imagined position via oracle reconstruction quality.
      3. Accumulate with discount factor gamma.

    The sequence of imagined states traces the arc's narrative through the
    manifold.  Arcs whose regions the query content fits naturally into will
    produce higher cumulative returns than arcs whose regions are alien to it.
    """
    verse_steps = [s for s in sequence if s.get("type") == "verse_trajectory"]
    if not verse_steps:
        return 0.0

    s     = start_vec.astype(np.float32)
    total = 0.0

    for i, step in enumerate(verse_steps):
        verse_emb = np.asarray(step["embedding"], dtype=np.float32)
        s_next    = _imagine_next(s, verse_emb, alpha=alpha)
        reward    = oracle_module.alignment_score(s_next)
        total    += reward * (gamma ** i)
        s         = s_next

    return total


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def diagnostics() -> dict:
    """Return a snapshot of the RL state for API exposure."""
    vf_state = _vf.state_dict()
    return {
        "value_function": {
            "n_updates":     vf_state["n_updates"],
            "mean_td_error": round(vf_state["mean_td_error"], 6),
            "weights_norm":  round(float(np.linalg.norm(vf_state["w"])), 6),
            "bias":          round(vf_state["b"], 6),
        },
        "program_library": {
            "navigation": _library.summary(kind="navigation"),
            "typological": _library.summary(kind="typological"),
        },
        "episode": {
            "steps_buffered":     len(_traj_states),
            "mean_episode_td_err": round(
                float(np.mean(_traj_td_errors)) if _traj_td_errors else 0.0, 6
            ),
        },
    }
