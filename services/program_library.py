"""
Program Abstraction Library — SOAR-style test-time program synthesis.

Inspired by:
  - ARC Prize 2025 SOAR framework (Pang): "dynamically creates a program
    abstraction library to steer synthesis"
  - Alberta Plan: search control via accumulated sub-task knowledge
  - NDEA search guidance: neuro-symbolic — learned heuristics + symbolic programs

At test-time, when the navigator finds a sequence of tool calls that produces
high alignment with the Biblical manifold, that sequence is synthesised into
a reusable program and stored here.  Future queries that land in the same
Biblical coordinate region trigger the stored program instead of re-discovering
the sequence from scratch.

This is the ONLY persistent state that changes across sessions.
The BibleGAT weights remain frozen.  The program library grows.

Program entry format:
  trigger    : 8-dim L2-normalised W2V centroid — the Biblical coordinate
               region that activates this program
  arc        : the typological pattern name (e.g. "exodus_deliverance")
  sequence   : ordered list of (tool_name, query_template) pairs
  reward     : mean alignment × goal-direction score observed when used
  n_uses     : how many times this program has been triggered and used

Synthesis criterion:
  A new program is synthesised when a navigation trajectory achieves
  alignment > SYNTHESIS_THRESHOLD across ALL steps.  The trigger is the
  mean state vector across that trajectory (the "centroid of success").

Retrieval:
  cosine similarity between current state and stored trigger vectors.
  If best_sim > RETRIEVAL_THRESHOLD, the matching program is executed
  directly, bypassing oracle.next_action for tool selection.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field


SYNTHESIS_THRESHOLD  = 0.65   # min mean alignment to synthesise a new program
RETRIEVAL_THRESHOLD  = 0.88   # min cosine to trigger an existing program
MAX_LIBRARY_SIZE     = 256    # prune least-used entries beyond this


@dataclass
class ProgramEntry:
    name:       str
    arc:        str            # typological pattern ("covenant", "exodus_deliverance", ...)
    trigger:    np.ndarray     # L2-normalised centroid in W2V space
    sequence:   list[dict]     # tool steps OR verse-trajectory steps (see kind)
    reward:     float          # mean alignment × goal-cosine
    n_uses:     int   = 0
    total_rwd:  float = 0.0
    kind:       str   = "navigation"   # "navigation" | "typological"
    # sequence format by kind:
    #   "navigation"  : [{"tool": str, "query_template": str}, ...]
    #   "typological" : [{"type": "verse_trajectory", "verse_id": int,
    #                      "ref": str, "embedding": np.ndarray}, ...]

    def update_trigger(self, new_centroid: np.ndarray, alpha: float = 0.1) -> None:
        """EMA update of trigger centroid as new matching trajectories arrive."""
        merged = (1 - alpha) * self.trigger + alpha * new_centroid
        norm = np.linalg.norm(merged)
        self.trigger = merged / (norm + 1e-9)


class ProgramLibrary:
    """
    Accumulates successful navigation programs at test-time.

    Alberta Plan mapping:
      search control  ←→  library retrieval (what program to execute next)
      options/subtasks ←→  program entries (reusable tool-call sequences)
      planning        ←→  sequence of retrieved programs toward a goal
    """

    def __init__(self) -> None:
        self._entries: list[ProgramEntry] = []

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def match(self, s: np.ndarray, kind: str = "navigation") -> ProgramEntry | None:
        """
        Find the best matching program for the current Biblical coordinate.

        Only searches programs of the given kind (default "navigation") so that
        typological seeds never interfere with tool-selection retrieval.

        Returns the entry if cosine(s, trigger) > RETRIEVAL_THRESHOLD,
        else None (fall through to oracle.next_action).
        """
        candidates = [e for e in self._entries if e.kind == kind]
        if not candidates:
            return None
        s = s.astype(np.float32)
        norm = np.linalg.norm(s)
        s_n = s / (norm + 1e-9)

        best_sim, best_entry = RETRIEVAL_THRESHOLD, None
        for entry in candidates:
            sim = float(np.dot(s_n, entry.trigger))
            if sim > best_sim:
                best_sim, best_entry = sim, entry

        if best_entry is not None:
            best_entry.n_uses += 1
        return best_entry

    # ------------------------------------------------------------------
    # Synthesis
    # ------------------------------------------------------------------

    def synthesise(
        self,
        trajectory: list[dict],
        mean_state: np.ndarray,
        arc: str,
        mean_reward: float,
    ) -> ProgramEntry | None:
        """
        Attempt to synthesise a new program from a successful trajectory.

        Parameters
        ----------
        trajectory   : list of step dicts, each with "tool", "query", "alignment"
        mean_state   : mean of state vectors across trajectory (the trigger centroid)
        arc          : dominant typological pattern name for this trajectory
        mean_reward  : mean alignment × goal-cosine across trajectory

        Returns the new (or updated) ProgramEntry, or None if below threshold.
        """
        if mean_reward < SYNTHESIS_THRESHOLD:
            return None

        # Build the action sequence from the trajectory
        sequence = [
            {"tool": step["tool"], "query_template": step.get("query", "")}
            for step in trajectory
        ]

        mean_state = mean_state.astype(np.float32)
        norm = np.linalg.norm(mean_state)
        trigger = mean_state / (norm + 1e-9)

        # Check for an existing entry close enough to update
        existing = self.match(trigger)
        if existing is not None:
            existing.update_trigger(trigger)
            existing.total_rwd += mean_reward
            existing.n_uses    += 1
            existing.reward     = existing.total_rwd / existing.n_uses
            return existing

        # Synthesise a new entry
        name  = f"{arc}:{len(self._entries):04d}"
        entry = ProgramEntry(
            name=name, arc=arc, trigger=trigger,
            sequence=sequence, reward=mean_reward,
            n_uses=1, total_rwd=mean_reward,
        )
        self._entries.append(entry)
        self._prune()
        return entry

    # ------------------------------------------------------------------
    # Housekeeping
    # ------------------------------------------------------------------

    def _prune(self) -> None:
        """Remove least-used NAVIGATION entries beyond MAX_LIBRARY_SIZE.
        Typological seeds are canonical — never pruned."""
        nav = [e for e in self._entries if e.kind == "navigation"]
        typ = [e for e in self._entries if e.kind != "navigation"]
        if len(nav) > MAX_LIBRARY_SIZE:
            nav.sort(key=lambda e: e.n_uses, reverse=True)
            nav = nav[:MAX_LIBRARY_SIZE]
        self._entries = nav + typ

    def summary(self, kind: str | None = None) -> dict:
        entries = self._entries if kind is None else [e for e in self._entries if e.kind == kind]
        return {
            "n_programs":  len(entries),
            "top_programs": [
                {"name": e.name, "arc": e.arc,
                 "reward": round(e.reward, 4), "n_uses": e.n_uses}
                for e in sorted(entries, key=lambda e: e.reward, reverse=True)[:5]
            ],
        }
