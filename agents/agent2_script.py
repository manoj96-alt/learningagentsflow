"""
agents/agent2_script.py

Agent 2: Script + Scene Builder
=================================
Role   : Professional YouTube scriptwriter and visual storyteller
Goal   : Convert Agent 1's research into a structured scene-by-scene script
Input  : Full A1 output fields from VideoState
Writes : title, hook, scene_list, full_script, narration_style,
         cta, estimated_duration_min

Design notes:
  - Agent 1's output is injected as structured JSON context
  - On QA retry, revision_notes are appended so A2 knows what to fix
  - Each scene maps directly to a SceneBrief that A3 will expand visually
  - scene_list is a list of scene title strings (A3 uses this for visual planning)
  - full_script is the complete concatenated narration (for QA length check)
"""

import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from state import VideoState, SceneBrief
from config import ANTHROPIC_API_KEY, LLM_MODEL, LLM_TEMPERATURE


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a professional YouTube scriptwriter and visual storyteller.
You specialize in educational content for teenage audiences (ages 13–17).
Your scripts are conversational, energetic, and visually driven.
Always respond with valid JSON only — no markdown fences, no preamble.
"""

# ── Human prompt — your specification, faithfully implemented ─────────────────

HUMAN_PROMPT = """\
INPUT (Agent 1 research output):
{agent1_context}

{revision_block}

TASK: Convert the input into a structured video script with scenes and visuals.

RULES:
- Each scene should be 5–10 seconds when read aloud
- Keep narration conversational — talk TO the viewer, not AT them
- Visuals must be simple and clear — one idea per scene
- Use storytelling flow: hook → problem → solution → example → summary
- Scene 1 MUST open with the hook from the input
- Final scene MUST include a summary + call to action
- animation_style must be one of: "2D", "whiteboard", "cinematic", "infographic"
- Avoid jargon unless immediately explained in the same scene
- Total duration should match the estimated_duration_min from input

OUTPUT FORMAT (valid JSON, no markdown):
{{
  "video_title"       : "final polished YouTube title (max 60 chars)",
  "duration_minutes"  : 7,
  "narration_style"   : "one-line tone description, e.g. 'friendly and curious'",
  "cta"               : "call-to-action line for the final scene",
  "scenes": [
    {{
      "scene_number"       : 1,
      "scene_title"        : "short scene name",
      "narration"          : "full spoken narration for this scene",
      "visual_description" : "what the viewer sees — one clear image or animation",
      "animation_style"    : "2D",
      "on_screen_text"     : "caption or lower-third text (keep short)"
    }}
  ]
}}
"""


# ── Prompt template ────────────────────────────────────────────────────────────

SCRIPT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human",  HUMAN_PROMPT),
])


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_agent1_context(state: VideoState) -> str:
    """Serialize A1's output fields as clean JSON for the prompt."""
    context = {
        "topic"              : state.get("topic", ""),
        "domain"             : state.get("domain", ""),
        "refined_topic"      : state.get("refined_topic", state.get("topic", "")),
        "title_draft"        : state.get("title", ""),
        "hook_draft"         : state.get("hook", ""),
        "simple_explanation" : state.get("simple_explanation", ""),
        "analogy"            : state.get("analogy", ""),
        "real_world_example" : state.get("real_world_example", ""),
        "key_facts"          : state.get("key_facts", []),
        "story_flow"         : state.get("scene_list", []),   # A1's story_flow seed
        "target_audience"    : state.get("target_audience", "beginners aged 15–25"),
        "estimated_duration_min": state.get("estimated_duration_min", 7),
    }
    return json.dumps(context, indent=2)


def _build_revision_block(state: VideoState) -> str:
    """On QA retry, inject revision notes so A2 knows exactly what to fix."""
    notes = state.get("revision_notes", [])
    iteration = state.get("qa_iteration", 0)

    if not notes or iteration == 0:
        return ""

    lines = "\n".join(f"  - {note}" for note in notes)
    return (
        f"QA REVISION REQUIRED (attempt {iteration + 1}):\n"
        f"The previous script failed QA. Fix ALL of the following:\n"
        f"{lines}\n"
        f"Rewrite the full script addressing every issue above.\n"
    )


def _scenes_to_state(scenes: list[dict]) -> tuple[list[str], list[SceneBrief], str]:
    """
    Convert raw scene dicts into:
      - scene_list      : list of scene title strings   (for A3 visual planning)
      - visual_briefs   : list of SceneBrief dicts      (partial — A3 enriches)
      - full_script     : concatenated narration string  (for QA length check)
    """
    scene_titles: list[str]    = []
    briefs: list[SceneBrief]   = []
    narrations: list[str]      = []

    for scene in scenes:
        num   = int(scene.get("scene_number", 0))
        title = str(scene.get("scene_title", f"Scene {num}"))
        scene_titles.append(f"Scene {num}: {title}")
        narrations.append(scene.get("narration", ""))

        briefs.append(SceneBrief(
            scene_number   = num,
            scene_title    = title,
            visual_prompt  = scene.get("visual_description", ""),
            b_roll         = "",                               # A3 will fill this
            on_screen_text = scene.get("on_screen_text", ""),
        ))

    full_script = "\n\n".join(
        f"[Scene {i+1}]\n{n}" for i, n in enumerate(narrations)
    )
    return scene_titles, briefs, full_script


# ── Agent node function ────────────────────────────────────────────────────────

def agent2_script(state: VideoState) -> VideoState:
    """
    LangGraph node — Agent 2: Script + Scene Builder.

    Reads  : All A1 fields from state + optional revision_notes
    Writes : title, hook (final), scene_list, visual_briefs,
             full_script, narration_style, cta, estimated_duration_min
    Returns: updated VideoState
    """
    print(f"\n[A2] Building script for: '{state.get('refined_topic', state.get('topic'))}'")

    iteration = state.get("qa_iteration", 0)
    if iteration > 0:
        print(f"[A2] QA retry #{iteration} — applying revision notes")

    # ── Build prompt inputs ────────────────────────────────────────────────────
    agent1_context  = _build_agent1_context(state)
    revision_block  = _build_revision_block(state)

    # ── Invoke LLM chain ──────────────────────────────────────────────────────
    llm = ChatAnthropic(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        api_key=ANTHROPIC_API_KEY,
    )
    chain  = SCRIPT_PROMPT | llm | JsonOutputParser()
    raw: dict = chain.invoke({
        "agent1_context" : agent1_context,
        "revision_block" : revision_block,
    })

    # ── Validate required fields ───────────────────────────────────────────────
    required = ["video_title", "duration_minutes", "narration_style", "cta", "scenes"]
    missing  = [k for k in required if k not in raw]
    if missing:
        raise ValueError(f"[A2] LLM response missing fields: {missing}\nRaw: {raw}")

    scenes: list[dict] = raw["scenes"]
    if not scenes:
        raise ValueError("[A2] LLM returned empty scenes list")

    # ── Convert scenes into state fields ──────────────────────────────────────
    scene_titles, visual_briefs, full_script = _scenes_to_state(scenes)

    # ── Write into VideoState ──────────────────────────────────────────────────
    updates: VideoState = {
        "title"                 : raw["video_title"],
        "hook"                  : scenes[0].get("narration", state.get("hook", "")),
        "scene_list"            : scene_titles,
        "visual_briefs"         : visual_briefs,
        "full_script"           : full_script,
        "narration_style"       : raw.get("narration_style", "friendly and curious"),
        "cta"                   : raw.get("cta", ""),
        "estimated_duration_min": int(raw.get("duration_minutes", 7)),
        "status"                : "in_progress",
    }

    print(f"[A2] Done — '{raw['video_title']}'  |  {len(scenes)} scenes  |  {raw.get('duration_minutes')} min")
    print(f"[A2] Narration style: {raw.get('narration_style')}")
    print(f"[A2] CTA: {raw.get('cta')}")

    return {**state, **updates}


# ── Standalone test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from state import initial_state
    from agents.agent1_research import agent1_research

    TEST_TOPIC  = "What is Retrieval Augmented Generation (RAG)?"
    TEST_DOMAIN = "AI"

    print(f"\n{'='*60}")
    print(f"PIPELINE TEST: A1 → A2")
    print(f"Topic : {TEST_TOPIC}")
    print(f"Domain: {TEST_DOMAIN}")
    print('='*60)

    # Run A1 first
    s = initial_state(TEST_TOPIC, TEST_DOMAIN)
    s = agent1_research(s)

    # Run A2
    s = agent2_script(s)

    # Display results
    print(f"\n{'─'*60}")
    print(f"FINAL TITLE    : {s.get('title')}")
    print(f"DURATION       : {s.get('estimated_duration_min')} min")
    print(f"NARRATION STYLE: {s.get('narration_style')}")
    print(f"CTA            : {s.get('cta')}")
    print(f"\nSCENES ({len(s.get('scene_list', []))}):")
    for scene_title in s.get("scene_list", []):
        print(f"  {scene_title}")
    print(f"\nFULL SCRIPT PREVIEW (first 400 chars):")
    print(s.get("full_script", "")[:400] + "...")
    print(f"\nVISUAL BRIEFS:")
    for brief in s.get("visual_briefs", []):
        print(f"  Scene {brief['scene_number']}: {brief['visual_prompt'][:80]}")
