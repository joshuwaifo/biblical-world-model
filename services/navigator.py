"""
Navigator — the search guidance agent.

Uses the Biblical world model geometry to guide navigation through any
information space.  Any stream of bits/bytes is a valid query; any open-source
model (Gemma, Llama, EXAONE, etc.) running locally via Ollama is a valid tool.

The loop
─────────
1. Encode the query (any modality) → 8-dim Biblical coordinates
2. Render: find the nearest Biblical concepts (what region of meaning is this?)
3. Plan: chart a geometric path toward the goal coordinates
4. For each step in the plan:
   a. Translate the current Biblical context into a tool prompt
   b. Select and invoke the best tool (web search, fetch, Ollama model)
   c. Encode the tool output → update current position
   d. Check proximity to goal; stop if close enough
5. Return: navigation trace, all retrieved content, final coordinates

The Biblical text is never shown to the user as output — it is the internal
coordinate system.  What is returned is the retrieved/synthesised information
from the tools, guided by the geometry.
"""

import time
import numpy as np
from dataclasses import dataclass, field
from typing import Any

from services import embedder as emb
from services import store
from services import oracle
from services import rl_loop

# Thresholds are read dynamically from the oracle at call time so they always
# reflect the BibleGAT's calibrated reconstruction distribution.
# oracle.alignment_floor / oracle.alignment_high are set during oracle.init().


def _match_pattern(vec: np.ndarray) -> dict:
    """Safe typology wrapper — returns fallback if typology not yet initialised."""
    try:
        from services import typology
        return typology.match_pattern(vec)
    except Exception:
        return {"name": "unknown", "confidence": 0.0, "key_passages": [],
                "arc": "", "description": "", "all_scores": {}}


@dataclass
class NavigationStep:
    step_num: int
    biblical_context: list[dict]   # nearest Bible verses at this position
    tool_used: str
    tool_input: dict
    tool_output: str
    coordinates: list[float]       # 8-dim Biblical position after this step
    cosine_to_goal: float
    alignment: float               # oracle: reconstruction likelihood of output
    novelty: float                 # oracle: flipped logit — how new is this?
    action_confidence: float       # oracle: attention weight that selected this tool
    elapsed_s: float
    # ── RL / Truth-layer fields ──────────────────────────────────────────
    outside_prior: bool = False          # alignment < ALIGNMENT_FLOOR
    manifold_interpretation: dict = field(default_factory=dict)  # typology match
    td_error: float = 0.0               # TD(λ) error at this step
    library_hit: bool = False            # True when program library fired


@dataclass
class NavigationResult:
    query_coordinates: list[float]
    goal_coordinates: list[float]
    steps: list[NavigationStep]
    final_coordinates: list[float]
    final_cosine_to_goal: float
    converged: bool
    synthesis: str                 # final answer synthesised by the model
    retrieved_content: list[dict]  # all raw tool outputs
    # ── Two-layer summaries ──────────────────────────────────────────────
    truth_layer_summary: dict = field(default_factory=dict)
    factoid_layer_summary: dict = field(default_factory=dict)
    rl_diagnostics: dict = field(default_factory=dict)


def navigate(
    query: Any,
    query_modality: str = "auto",
    goal: str = "",
    max_steps: int = 5,
    convergence_threshold: float = 0.85,
    tool_preference: list[str] | None = None,
) -> NavigationResult:
    """
    Navigate from the query toward the goal through the information space.

    Parameters
    ----------
    query                  : any content (str, bytes, image bytes, ...)
    query_modality         : "text" | "image" | "audio" | "video" | "bytes" | "auto"
    goal                   : text description of what we're looking for
    max_steps              : maximum tool invocations
    convergence_threshold  : cosine similarity to goal coordinates at which
                             we consider the search converged
    tool_preference        : ordered list of tool names to prefer

    Returns
    -------
    NavigationResult with full trace + synthesised answer
    """
    from tools import registry

    # ── 1. Encode query ──────────────────────────────────────────────────
    query_vec = emb.encode(query, query_modality)

    # ── 2. Encode goal ───────────────────────────────────────────────────
    goal_vec  = emb.encode(goal, "text") if goal else query_vec.copy()

    # ── 3. Find Biblical context at query position ────────────────────────
    start_context = _biblical_context(query_vec, k=5)

    current_vec   = query_vec.copy()
    prev_vec      = query_vec.copy()   # s_{t-1} for TD update
    steps: list[NavigationStep] = []
    all_content: list[dict] = []

    # ── 4. RL loop initialisation ─────────────────────────────────────────
    tools      = _select_tools(tool_preference, registry)
    rl_loop.begin_episode()

    for step_num in range(1, max_steps + 1):
        t0 = time.time()

        cosine_to_goal = float(np.dot(current_vec, goal_vec))

        if cosine_to_goal >= convergence_threshold and step_num > 1:
            break

        # Biblical context at current position
        bib_ctx = _biblical_context(current_vec, k=3)

        # Choose tool via Alberta Plan reactive-policy hierarchy (rl_loop)
        # then generate the query string via the existing oracle helper
        tool_name, tool_kwargs, action_confidence, library_hit = _plan_tool_call_rl(
            current_vec, goal_vec, goal, bib_ctx, step_num, tools, all_content
        )

        # Invoke tool
        try:
            tool = registry.get(tool_name)
            raw_result = tool.invoke(**tool_kwargs)
            # some tools return a list (web_search returns multiple results)
            if isinstance(raw_result, list):
                snippets = [r for r in raw_result if isinstance(r.content, str)]
                # oracle: rank snippets by alignment — prefer content consistent
                # with the Biblical world model (high alignment = high truth signal)
                if snippets:
                    snippet_vecs = [emb.encode(r.content, "text") for r in snippets]
                    alignments   = [oracle.alignment_score(v) for v in snippet_vecs]
                    novelties    = [1.0 - a for a in alignments]
                    goal_sims    = [float(np.dot(v, goal_vec)) for v in snippet_vecs]
                    scores_rank  = [a * g for a, g in zip(alignments, goal_sims)]
                    order        = sorted(range(len(snippets)),
                                          key=lambda i: scores_rank[i], reverse=True)
                    snippets     = [snippets[i] for i in order]
                    alignments   = [alignments[i] for i in order]
                    novelties    = [novelties[i] for i in order]

                output_text    = "\n\n".join(r.content for r in snippets)
                step_alignment = float(np.mean(alignments)) if alignments else 0.0
                step_novelty   = float(np.mean(novelties))  if novelties  else 1.0
                for r, a, n in zip(snippets, alignments, novelties):
                    all_content.append({
                        "step": step_num, "tool": tool_name,
                        "source": r.source, "text": r.content[:500],
                        "alignment": round(a, 4), "novelty": round(n, 4),
                    })
            else:
                output_text    = raw_result.content if isinstance(raw_result.content, str) else ""
                out_vec        = emb.encode(output_text, "text") if output_text.strip() else current_vec
                step_alignment = oracle.alignment_score(out_vec)
                step_novelty   = 1.0 - step_alignment
                all_content.append({
                    "step": step_num, "tool": tool_name,
                    "source": raw_result.source, "text": output_text[:500],
                    "alignment": round(step_alignment, 4),
                    "novelty":   round(step_novelty, 4),
                })
        except Exception as e:
            output_text    = f"[Tool error: {e}]"
            step_alignment = 0.0
            step_novelty   = 1.0
            all_content.append({"step": step_num, "tool": tool_name,
                                 "source": "error", "text": output_text,
                                 "alignment": 0.0, "novelty": 1.0})

        # Update position: encode tool output → move toward it
        if output_text.strip():
            new_vec     = emb.encode(output_text, "text")
            current_vec = _interpolate(current_vec, new_vec, alpha=0.6)

        # ── Truth-layer signals ───────────────────────────────────────────
        manifold_match = _match_pattern(current_vec)
        outside_prior  = step_alignment < oracle.alignment_floor

        # ── TD(λ) update: reward = oracle alignment (grounded, not human) ─
        is_last = (step_num == max_steps) or (
            float(np.dot(current_vec, goal_vec)) >= convergence_threshold
        )
        td_err = rl_loop.update_value(
            s_t=prev_vec,
            reward=step_alignment,
            s_t1=current_vec,
            done=is_last,
            step_info={
                "tool":      tool_name,
                "query":     tool_kwargs.get("query", tool_kwargs.get("prompt", "")),
                "alignment": step_alignment,
            },
        )
        prev_vec = current_vec.copy()

        steps.append(NavigationStep(
            step_num=step_num,
            biblical_context=bib_ctx,
            tool_used=tool_name,
            tool_input=tool_kwargs,
            tool_output=output_text[:800],
            coordinates=[round(float(x), 5) for x in current_vec],
            cosine_to_goal=round(float(np.dot(current_vec, goal_vec)), 4),
            alignment=round(step_alignment, 4),
            novelty=round(step_novelty, 4),
            action_confidence=round(action_confidence, 4),
            elapsed_s=round(time.time() - t0, 2),
            outside_prior=outside_prior,
            manifold_interpretation=manifold_match,
            td_error=round(td_err, 6),
            library_hit=library_hit,
        ))

    # ── 5. Two-layer summaries ────────────────────────────────────────────
    step_alignments = [s.alignment for s in steps]
    overall_align   = float(np.mean(step_alignments)) if step_alignments else 0.0

    if len(step_alignments) >= 2:
        rising = all(
            step_alignments[i] <= step_alignments[i + 1]
            for i in range(len(step_alignments) - 1)
        )
        all_below = all(a < oracle.alignment_floor for a in step_alignments)
        trajectory = "converging" if rising else ("exploring" if all_below else "mixed")
    else:
        trajectory = "single_step"

    dominant_pattern = _match_pattern(current_vec)

    seen, key_passages_invoked = set(), []
    for s in steps:
        for p in s.manifold_interpretation.get("key_passages", [])[:2]:
            if p.get("ref") not in seen:
                key_passages_invoked.append(p)
                seen.add(p.get("ref"))

    truth_layer_summary = {
        "overall_alignment":             round(overall_align, 4),
        "alignment_floor":               round(oracle.alignment_floor, 4),
        "alignment_high":                round(oracle.alignment_high, 4),
        "trajectory":                    trajectory,
        "dominant_pattern":              dominant_pattern,
        "key_biblical_passages_invoked": key_passages_invoked,
    }
    factoid_layer_summary = {
        "sources_consulted":  list({c.get("source", "") for c in all_content}),
        "tools_used":         list({c.get("tool", "") for c in all_content}),
        "total_snippets":     len(all_content),
        "high_novelty_steps": sum(1 for s in steps if s.outside_prior),
    }

    # ── 6. SOAR program synthesis attempt ────────────────────────────────
    arc_name   = dominant_pattern.get("name", "unknown")
    synthesised = rl_loop.end_episode(dominant_arc=arc_name)

    # ── 7. Synthesise answer via local model ─────────────────────────────
    synthesis = _synthesise(query, goal, all_content, tools,
                            dominant_pattern=dominant_pattern)

    final_cosine = float(np.dot(current_vec, goal_vec))

    return NavigationResult(
        query_coordinates=[round(float(x), 5) for x in query_vec],
        goal_coordinates=[round(float(x), 5) for x in goal_vec],
        steps=steps,
        final_coordinates=[round(float(x), 5) for x in current_vec],
        final_cosine_to_goal=round(final_cosine, 4),
        converged=final_cosine >= convergence_threshold,
        synthesis=synthesis,
        retrieved_content=all_content,
        truth_layer_summary=truth_layer_summary,
        factoid_layer_summary=factoid_layer_summary,
        rl_diagnostics={
            **rl_loop.diagnostics(),
            "synthesised_program": synthesised,
        },
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _biblical_context(vec: np.ndarray, k: int = 5) -> list[dict]:
    """Find the k nearest Bible verses to a coordinate vector."""
    neighbors = store.knn(vec, k)
    ctx = []
    for vid, sim in neighbors:
        m = store.get_meta(vid)
        ctx.append({"ref": m["ref"], "text": m["text"], "sim": round(sim, 4)})
    return ctx


def _select_tools(preference: list[str] | None, registry) -> list[str]:
    available = registry.available_names()
    if preference:
        ordered = [t for t in preference if t in available]
        rest = [t for t in available if t not in ordered]
        return ordered + rest
    # Default: web_search first, then ollama, then fetch
    preferred = ["web_search", "ollama", "fetch", "ollama_vision"]
    return [t for t in preferred if t in available] + \
           [t for t in available if t not in preferred]


def _plan_tool_call_rl(
    current_vec:   np.ndarray,
    goal_vec:      np.ndarray,
    goal:          str,
    bib_ctx:       list[dict],
    step_num:      int,
    tools:         list[str],
    prior_content: list[dict],
) -> tuple[str, dict, float, bool]:
    """
    Alberta Plan reactive-policy + query generation.
    Returns (tool_name, kwargs, action_confidence, library_hit).

    Tool selection — three-tier hierarchy (rl_loop.select_action):
      1. Program library retrieval (SOAR: proven sequence → shortcut)
      2. Oracle GNN attention (frozen policy)
      3. V-guided lookahead over imagined next states (when VF is warm)

    Query generation — unchanged: Ollama + Biblical coordinate context.
    """
    from tools import registry as tool_registry

    tool_list = [t for t in tools if t in tool_registry.all_tools()]
    if not tool_list:
        return "web_search", {"query": goal, "max_results": 5}, 0.0, False

    # Embed each tool's description into W2V Biblical space
    tool_vecs = [emb.encode(tool_registry.get(t).description, "text") for t in tool_list]

    # ── Alberta Plan: select action via rl_loop ───────────────────────────
    selected_tool, action_conf, library_hit = rl_loop.select_action(
        current_vec=current_vec,
        goal_vec=goal_vec,
        tool_list=tool_list,
        tool_vecs=tool_vecs,
        step_num=step_num,
        prior_content=prior_content,
    )

    # Override on step 1: always start with a broad web sweep
    if step_num == 1 and "web_search" in tool_list and not library_hit:
        selected_tool = "web_search"
        action_conf   = 0.0   # step-1 override, not from policy

    # ── Query generation via Ollama + Biblical context ────────────────────
    context_text = "\n".join(
        f"  [{c['ref']}] {c['text'][:100]}" for c in bib_ctx
    )
    prior_summary = "\n".join(
        f"  [{p['source']}] {p['text'][:150]}" for p in prior_content[-3:]
    ) if prior_content else "(none yet)"

    search_query = goal
    if "ollama" in tool_list and step_num > 1:
        refine_prompt = (
            f"You are navigating an information space toward a goal.\n"
            f"Goal: {goal}\n\n"
            f"Current conceptual region (nearest Biblical concepts):\n"
            f"{context_text}\n\n"
            f"Retrieved so far:\n{prior_summary}\n\n"
            f"Write ONE specific search query (just the query, no explanation) "
            f"that will best advance toward the goal from this conceptual position."
        )
        try:
            model_tool = tool_registry.get("ollama")
            result = model_tool.invoke(
                refine_prompt,
                context=(
                    "You are a precise research navigator. "
                    "Output only the search query string, nothing else."
                ),
            )
            candidate = result.content.strip().strip('"').strip("'")
            if candidate and len(candidate) < 250:
                search_query = candidate
        except Exception:
            pass

    if selected_tool == "web_search":
        return selected_tool, {"query": search_query, "max_results": 5}, action_conf, library_hit
    elif selected_tool == "fetch":
        return selected_tool, {"url": search_query}, action_conf, library_hit
    elif selected_tool in ("ollama", "ollama_vision"):
        return selected_tool, {"prompt": f"Goal: {goal}\n\nContext:\n{prior_summary}"}, action_conf, library_hit
    else:
        return "web_search", {"query": search_query, "max_results": 5}, action_conf, library_hit


def _interpolate(a: np.ndarray, b: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    """Move alpha-fraction from a toward b, then re-normalise."""
    mixed = (1 - alpha) * a + alpha * b
    norm  = np.linalg.norm(mixed)
    return mixed / (norm + 1e-9)


def _synthesise(
    query: Any,
    goal: str,
    content: list[dict],
    tools: list[str],
    dominant_pattern: dict | None = None,
) -> str:
    """Ask the local model to synthesise a final answer.

    When a dominant typological pattern is available, the prompt opens with
    the truth-layer context so the model's answer is grounded in it.
    """
    if "ollama" not in tools:
        return "(local model not available for synthesis)"
    if not content:
        return "(no content retrieved)"

    combined = "\n\n".join(
        f"[{c['source']}]\n{c['text']}" for c in content
    )[:4000]

    truth_preamble = ""
    if dominant_pattern and dominant_pattern.get("name") not in (None, "unknown"):
        passages = dominant_pattern.get("key_passages", [])[:3]
        passage_lines = "\n".join(
            f"  [{p['ref']}] {p.get('text', '')[:100]}" for p in passages
        )
        truth_preamble = (
            f"TRUTH LAYER (Biblical prior):\n"
            f"Pattern: {dominant_pattern['name']} — {dominant_pattern.get('arc', '')}\n"
            f"Key passages:\n{passage_lines}\n\n"
        )

    prompt = (
        f"{truth_preamble}"
        f"FACTOID LAYER (retrieved content):\n{combined}\n\n"
        f"Goal: {goal}\n\n"
        f"In your answer, {'explicitly state what the typological pattern reveals about this situation, then ' if truth_preamble else ''}"
        f"provide a concise, accurate factual answer.\n\nAnswer:"
    )

    try:
        from tools import registry
        model_tool = registry.get("ollama")
        result = model_tool.invoke(prompt)
        return result.content.strip()
    except Exception as e:
        return f"(synthesis failed: {e})"
