"""
agents/agent4_qa.py

Agent 4: QA + Refinement
==========================
Role   : Strict quality reviewer for educational YouTube content
Goal   : Score content on 4 dimensions, approve or reject with actionable notes
Input  : Full A2 (script) + A3 (visual plan) fields from VideoState
Writes : qa_scores, qa_status, revision_notes, qa_iteration
         On pass: also writes refined title/hook/full_script improvements

Scoring dimensions (each 0.0–1.0, threshold from config.QA_PASS_THRESHOLD):
  1. accuracy  — factual correctness, no hallucinations
  2. tone      — 15-yr-old suitability, conversational, jargon-free
  3. seo       — title/hook keyword strength, search-worthy
  4. length    — script fits 5–10 minute target (~700–1400 words)

Routing (handled by route_after_qa in state.py):
  ALL four scores >= threshold → qa_status = "pass" → A5
  ANY score < threshold        → qa_status = "fail" → A2 (retry with notes)
  qa_iteration >= MAX_RETRIES  → qa_status = "fail" → END
"""

import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from state import VideoState, QAScores
from config import (
    ANTHROPIC_API_KEY, LLM_MODEL, LLM_TEMPERATURE,
    QA_PASS_THRESHOLD, QA_MAX_RETRIES,
)


# ── Word-count constants for length scoring ────────────────────────────────────
WORDS_PER_MINUTE    = 140    # average conversational speaking pace
MIN_DURATION_MIN    = 5
MAX_DURATION_MIN    = 10
MIN_WORDS           = WORDS_PER_MINUTE * MIN_DURATION_MIN   # 700
MAX_WORDS           = WORDS_PER_MINUTE * MAX_DURATION_MIN   # 1400


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a strict quality reviewer for educational YouTube content.
Your audience is teenagers aged 13–17 and young adults aged 18–25.
You review scripts and visual plans against four quality dimensions.
You are direct, specific, and actionable — no vague feedback.
Always respond with valid JSON only — no markdown fences, no preamble.
"""

# ── Human prompt — your specification, faithfully implemented ─────────────────

HUMAN_PROMPT = """\
INPUT:
Script (Agent 2 output):
{script_context}

Visual Plan (Agent 3 output):
{visual_context}

Word count of full script: {word_count} words
Estimated duration: {estimated_duration} minutes
Target range: 5–10 minutes ({min_words}–{max_words} words)

TASK: Review and refine content for quality, clarity, and engagement.

CHECK each dimension and score 0.0–1.0:

1. ACCURACY  — Is the explanation factually correct? No hallucinations or invented facts?
2. TONE      — Is it easy to understand for a 15-year-old? Conversational? Jargon explained?
3. SEO       — Is the title compelling and search-worthy? Does the hook grab attention in 5 sec?
4. LENGTH    — Does the script fit within 5–10 minutes at normal speaking pace?
5. VISUALS   — Are visuals aligned with narration? One idea per scene? Not too complex?

RULES:
- Reject (approved: false) if ANY dimension score < {threshold}
- Be specific — every issue must name the scene number or field it refers to
- Improvements must be actionable — say exactly what to change, not just "improve this"
- If approving, still list improvements for the next iteration
- Keep tone engaging and simple in any refined_script sections
- refined_script: provide ONLY the scenes that need rewriting (not the full script)

OUTPUT FORMAT (valid JSON, no markdown):
{{
  "approved"      : true,
  "scores": {{
    "accuracy"    : 0.92,
    "tone"        : 0.88,
    "seo"         : 0.85,
    "length"      : 0.90
  }},
  "issues"        : [
    "Scene 3 narration uses the term 'vector embedding' without explaining it",
    "Title lacks a curiosity-triggering word (e.g. 'Why', 'How', 'What if')"
  ],
  "improvements"  : [
    "Scene 3: Replace 'vector embedding' with 'a map of meaning in numbers'",
    "Title: Change to 'What If Your AI Could Look Things Up? — RAG Explained'"
  ],
  "visual_issues" : [
    "Scene 5 visual is too abstract — suggest a simpler metaphor visual"
  ],
  "refined_script": {{
    "title": "improved title if needed, else empty string",
    "hook" : "improved hook if needed, else empty string",
    "scenes": []
  }}
}}
"""

# ── Prompt template ────────────────────────────────────────────────────────────

QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human",  HUMAN_PROMPT),
])


# ── Helpers ────────────────────────────────────────────────────────────────────

def _build_script_context(state: VideoState) -> str:
    """Serialize A2 script fields for the QA prompt."""
    context = {
        "video_title"       : state.get("title", ""),
        "hook"              : state.get("hook", ""),
        "narration_style"   : state.get("narration_style", ""),
        "cta"               : state.get("cta", ""),
        "duration_minutes"  : state.get("estimated_duration_min", 7),
        "full_script"       : state.get("full_script", ""),
        "scene_list"        : state.get("scene_list", []),
    }
    return json.dumps(context, indent=2)


def _build_visual_context(state: VideoState) -> str:
    """Serialize A3 visual fields for the QA prompt."""
    briefs = state.get("visual_briefs", [])
    scenes_out = []
    for b in briefs:
        scenes_out.append({
            "scene_number"  : b["scene_number"],
            "scene_title"   : b["scene_title"],
            "image_prompt"  : b.get("visual_prompt", ""),
            "video_prompt"  : b.get("video_prompt", ""),   # type: ignore
            "voiceover_text": b.get("voiceover_text", ""), # type: ignore
            "background_music": b.get("background_music",""), # type: ignore
            "b_roll"        : b.get("b_roll", ""),
            "on_screen_text": b.get("on_screen_text", ""),
        })
    context = {
        "style_guide"     : briefs[0].get("style_guide", "") if briefs else "", # type: ignore
        "voice_style"     : briefs[0].get("voice_style", "") if briefs else "",  # type: ignore
        "thumbnail_prompt": state.get("thumbnail_prompt", ""),
        "color_palette"   : state.get("color_palette", []),
        "font_suggestion" : state.get("font_suggestion", ""),
        "scenes"          : scenes_out,
    }
    return json.dumps(context, indent=2)


def _count_words(script: str) -> int:
    return len(script.split())


def _length_score(word_count: int) -> float:
    """Score the script length — 1.0 if in range, degrades outside it."""
    if MIN_WORDS <= word_count <= MAX_WORDS:
        return 1.0
    elif word_count < MIN_WORDS:
        return max(0.0, word_count / MIN_WORDS)
    else:
        # Penalise proportionally above max
        return max(0.0, 1.0 - (word_count - MAX_WORDS) / MAX_WORDS)


def _build_revision_notes(raw: dict) -> list[str]:
    """Combine issues + improvements into actionable revision notes for A2."""
    notes = []
    for issue in raw.get("issues", []):
        notes.append(f"ISSUE: {issue}")
    for improvement in raw.get("improvements", []):
        notes.append(f"FIX: {improvement}")
    for visual_issue in raw.get("visual_issues", []):
        notes.append(f"VISUAL: {visual_issue}")
    return notes


def _apply_refined_script(state: VideoState, refined: dict) -> dict:
    """
    Merge the LLM's partial refined_script suggestions back into state.
    Only overwrite fields that the LLM actually changed (non-empty).
    """
    updates = {}
    if refined.get("title"):
        updates["title"] = refined["title"]
    if refined.get("hook"):
        updates["hook"] = refined["hook"]
    # Partial scene rewrites — merge by scene_number
    if refined.get("scenes"):
        existing_briefs = state.get("visual_briefs", [])
        brief_map = {b["scene_number"]: b for b in existing_briefs}
        for scene in refined["scenes"]:
            num = scene.get("scene_number")
            if num and num in brief_map:
                brief_map[num]["visual_prompt"] = scene.get(
                    "visual_description", brief_map[num]["visual_prompt"]
                )
        updates["visual_briefs"] = list(brief_map.values())
    return updates


# ── Agent node function ────────────────────────────────────────────────────────

def agent4_qa(state: VideoState) -> VideoState:
    """
    LangGraph node — Agent 4: QA + Refinement.

    Reads  : All A2 + A3 fields from state
    Writes : qa_scores, qa_status, revision_notes, qa_iteration
             Plus refined title/hook/briefs if LLM suggests improvements
    Returns: updated VideoState
    """
    title     = state.get("title", state.get("topic", ""))
    iteration = state.get("qa_iteration", 0)
    print(f"\n[A4] QA review: '{title}'  (iteration {iteration + 1})")

    # ── Build prompt inputs ────────────────────────────────────────────────────
    script_context  = _build_script_context(state)
    visual_context  = _build_visual_context(state)
    full_script     = state.get("full_script", "")
    word_count      = _count_words(full_script)
    estimated_dur   = state.get("estimated_duration_min", 7)

    # ── Compute length score independently (no hallucination risk) ─────────────
    computed_length_score = _length_score(word_count)
    print(f"[A4] Word count: {word_count}  |  Length score: {computed_length_score:.2f}")

    # ── Invoke LLM chain ──────────────────────────────────────────────────────
    llm = ChatAnthropic(
        model=LLM_MODEL,
        temperature=0.3,   # Lower temperature: QA should be consistent
        api_key=ANTHROPIC_API_KEY,
    )
    chain  = QA_PROMPT | llm | JsonOutputParser()
    raw: dict = chain.invoke({
        "script_context"    : script_context,
        "visual_context"    : visual_context,
        "word_count"        : word_count,
        "estimated_duration": estimated_dur,
        "min_words"         : MIN_WORDS,
        "max_words"         : MAX_WORDS,
        "threshold"         : QA_PASS_THRESHOLD,
    })

    # ── Extract and validate scores ────────────────────────────────────────────
    raw_scores = raw.get("scores", {})
    qa_scores = QAScores(
        accuracy = float(raw_scores.get("accuracy", 0.5)),
        tone     = float(raw_scores.get("tone",     0.5)),
        seo      = float(raw_scores.get("seo",      0.5)),
        length   = computed_length_score,       # Override LLM with computed value
    )

    # ── Determine pass/fail ────────────────────────────────────────────────────
    all_pass    = all(v >= QA_PASS_THRESHOLD for v in qa_scores.values())
    llm_approved = bool(raw.get("approved", False))
    qa_status   = "pass" if (all_pass and llm_approved) else "fail"

    # ── Build revision notes for A2 retry ─────────────────────────────────────
    revision_notes = _build_revision_notes(raw)

    # ── Apply any partial refinements the LLM suggested ───────────────────────
    refinements = {}
    if refined := raw.get("refined_script", {}):
        refinements = _apply_refined_script(state, refined)

    # ── Log QA result ─────────────────────────────────────────────────────────
    status_icon = "✅ PASS" if qa_status == "pass" else "❌ FAIL"
    print(f"[A4] {status_icon}")
    print(f"[A4] Scores — accuracy: {qa_scores['accuracy']:.2f}  "
          f"tone: {qa_scores['tone']:.2f}  "
          f"seo: {qa_scores['seo']:.2f}  "
          f"length: {qa_scores['length']:.2f}  "
          f"(threshold: {QA_PASS_THRESHOLD})")
    if revision_notes:
        print(f"[A4] Revision notes ({len(revision_notes)}):")
        for note in revision_notes:
            print(f"       {note}")

    # ── Write into VideoState ──────────────────────────────────────────────────
    updates: VideoState = {
        "qa_scores"     : qa_scores,
        "qa_status"     : qa_status,           # type: ignore[typeddict-item]
        "revision_notes": revision_notes,
        "qa_iteration"  : iteration + 1,
        **refinements,
    }

    return {**state, **updates}


# ── Standalone test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from state import initial_state, route_after_qa
    from agents.agent1_research import agent1_research
    from agents.agent2_script    import agent2_script
    from agents.agent3_visual    import agent3_visual

    TEST_TOPIC  = "What is Retrieval Augmented Generation (RAG)?"
    TEST_DOMAIN = "AI"

    print(f"\n{'='*60}")
    print(f"PIPELINE TEST: A1 → A2 → A3 → A4")
    print(f"Topic : {TEST_TOPIC}")
    print('='*60)

    s = initial_state(TEST_TOPIC, TEST_DOMAIN)
    s = agent1_research(s)
    s = agent2_script(s)
    s = agent3_visual(s)
    s = agent4_qa(s)

    print(f"\n{'─'*60}")
    print(f"QA STATUS   : {s.get('qa_status')}")
    print(f"QA SCORES   : {s.get('qa_scores')}")
    print(f"ITERATIONS  : {s.get('qa_iteration')}")
    print(f"NEXT NODE   : {route_after_qa(s)}")
    print(f"REVISIONS   : {len(s.get('revision_notes', []))} notes")
    for note in s.get("revision_notes", []):
        print(f"  {note}")
