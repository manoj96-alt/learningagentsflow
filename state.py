"""
state.py — Single source of truth for the entire 6-agent pipeline.

Every agent reads from VideoState and writes back to it.
Nothing is passed as function arguments between agents.

LangGraph uses this TypedDict to:
  - Track which fields are populated at each node
  - Decide routing conditions (e.g. qa_status == "pass")
  - Enable retry loops without losing prior work
  - Allow the orchestrator to inspect state at any point
"""

from typing import TypedDict, Optional, Literal
from datetime import date


# ── QA scores (0.0–1.0) ───────────────────────────────────────────────────────

class QAScores(TypedDict):
    accuracy:    float   # factual correctness
    tone:        float   # 15-yr-old suitability, no jargon
    seo:         float   # title/hook keyword strength
    length:      float   # script fits 5–10 min target


# ── One visual brief per scene ────────────────────────────────────────────────

class SceneBrief(TypedDict):
    scene_number:   int
    scene_title:    str
    visual_prompt:  str    # prompt for DALL-E / Midjourney
    b_roll:         str    # suggested stock footage description
    on_screen_text: str    # caption or lower-third text


# ── Feedback from a posted video ─────────────────────────────────────────────

class VideoFeedback(TypedDict):
    views:              int
    watch_time_percent: float   # avg % of video watched
    ctr:                float   # click-through rate on thumbnail
    top_comments:       list[str]
    sentiment:          Literal["positive", "mixed", "negative"]
    growth_score:       float   # 0.0–1.0 composite performance score


# ── Master state ──────────────────────────────────────────────────────────────

class VideoState(TypedDict, total=False):
    """
    Shared state passed through all 6 LangGraph nodes.

    Fields marked (A1)–(A6) show which agent populates them.
    Fields with total=False are optional until that agent runs.
    """

    # ── Input ─────────────────────────────────────────────────────────────────
    topic:          str                             # user-provided topic
    domain:         Literal["AI", "Data Governance", "SAP MDG"]
    input_style:    Literal["direct", "subject_area", "rough_idea"]
    raw_input:      str                             # original user text

    # ── A1: Research + Simplification ────────────────────────────────────────
    refined_topic:          str                     # sharpened topic title
    simple_explanation:     str                     # plain language, 3–5 sentences
    analogy:                str                     # one vivid real-world analogy
    real_world_example:     str                     # concrete relatable example
    key_facts:              list[str]               # 3–5 bullet facts
    target_audience:        str                     # who this video is for
    research_confidence:    float                   # 0.0–1.0 self-rating

    # ── A2: Script + Scene Builder ────────────────────────────────────────────
    title:                  str                     # SEO-friendly video title
    hook:                   str                     # first 5-second line
    scene_list:             list[str]               # ordered scene titles
    full_script:            str                     # complete narration text
    narration_style:        str                     # tone descriptor
    cta:                    str                     # call-to-action at end
    estimated_duration_min: int                     # target 5–10

    # ── A3: Visual Generation Planner ─────────────────────────────────────────
    visual_briefs:          list[SceneBrief]        # one brief per scene
    thumbnail_prompt:       str                     # image gen prompt
    color_palette:          list[str]               # hex codes for visual consistency
    font_suggestion:        str                     # typography note

    # ── A4: QA + Refinement ───────────────────────────────────────────────────
    qa_scores:              QAScores
    qa_status:              Literal["pending", "pass", "fail"]
    revision_notes:         list[str]               # specific fix instructions
    qa_iteration:           int                     # retry count (max 2)

    # ── A5: Publishing ────────────────────────────────────────────────────────
    notion_page_id:         str
    publish_date:           str                     # ISO date "YYYY-MM-DD"
    publish_day:            str                     # "Tuesday"
    local_json_path:        str                     # path to saved output file
    status:                 Literal[
        "pending", "in_progress", "drafted",
        "reviewed", "scheduled", "posted"
    ]

    # ── A6: Feedback + Growth Engine ─────────────────────────────────────────
    feedback:               VideoFeedback
    feedback_summary:       str                     # plain-language analysis
    lessons_learned:        list[str]               # what worked / didn't
    next_topic_suggestions: list[str]               # fed back into next run


# ── Routing helpers ───────────────────────────────────────────────────────────

def route_after_qa(state: VideoState) -> str:
    """
    LangGraph conditional edge: called after A4 runs.

    Returns the name of the next node:
      - "agent5_publish"  if QA passed
      - "agent2_script"   if QA failed and retries remain
      - "END"             if max retries exceeded
    """
    qa_status    = state.get("qa_status", "pending")
    qa_iteration = state.get("qa_iteration", 0)
    MAX_RETRIES  = 2

    if qa_status == "pass":
        return "agent5_publish"

    if qa_iteration < MAX_RETRIES:
        return "agent2_script"   # retry: rewrite script with revision_notes

    return "END"                 # give up after max retries


def route_after_research(state: VideoState) -> str:
    """
    LangGraph conditional edge: called after A1 runs.
    Always proceeds to A2, but confidence < 0.5 could
    be extended here to trigger a human-in-the-loop pause.
    """
    confidence = state.get("research_confidence", 1.0)
    if confidence < 0.5:
        return "agent1_research"  # re-research with narrower scope
    return "agent2_script"


# ── Initial state factory ─────────────────────────────────────────────────────

def initial_state(
    topic: str,
    domain: str,
    input_style: str = "direct",
    raw_input: str = "",
) -> VideoState:
    """
    Build a clean VideoState for a new pipeline run.
    All downstream fields start unpopulated.
    """
    return VideoState(
        topic=topic,
        domain=domain,                      # type: ignore[arg-type]
        input_style=input_style,            # type: ignore[arg-type]
        raw_input=raw_input or topic,
        qa_status="pending",
        qa_iteration=0,
        status="pending",
    )
