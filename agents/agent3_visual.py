"""
agents/agent3_visual.py

Agent 3: Visual Generation Planner
=====================================
Role   : AI video production planner
Goal   : Convert each script scene into production-ready asset prompts
Input  : Full A2 output (scenes/visual_briefs) from VideoState
Writes : visual_briefs[] (enriched), thumbnail_prompt,
         color_palette, font_suggestion

Design notes:
  - Receives partial SceneBrief list from A2 (visual_prompt + on_screen_text filled)
  - Enriches each brief with: image_prompt, video_prompt, voiceover_text,
    voice_style, background_music, b_roll
  - Adds thumbnail_prompt, color_palette, font_suggestion at video level
  - Style consistency enforced by passing a style_guide built from scene 1
    into the prompt for all subsequent scenes
  - Avoids complex visuals per rules: prompts stay flat/2D/icon-based
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
You are an AI video production planner specializing in educational YouTube content.
You convert scripts into detailed, production-ready asset prompts for image, video,
and voice generation tools.
Your prompts are precise, consistent, and optimized for AI generation tools
like DALL-E, Midjourney, Stable Diffusion, ElevenLabs, and Sora.
Always respond with valid JSON only — no markdown fences, no preamble.
"""

# ── Human prompt — your specification, faithfully implemented ─────────────────

HUMAN_PROMPT = """\
INPUT (Agent 2 script output):
{agent2_context}

TASK: Convert each scene into prompts for image/video/voice generation tools.

RULES:
- Image prompts must be detailed and visual — include style, colors, composition
- Keep visual style CONSISTENT across all scenes (use the style_guide field)
- Voiceover text must match the scene narration exactly — do not paraphrase
- voice_style must stay the same for all scenes (pick once, apply everywhere)
- background_music should complement the scene energy (calm/upbeat/dramatic/neutral)
- Avoid complex visuals: flat 2D, icon-based, or simple animation — no photorealism
- image_prompt and video_prompt must be distinct:
    image_prompt = static frame (for thumbnails or still assets)
    video_prompt = motion description (what animates, how it moves)
- Total asset count must equal total scene count

OUTPUT FORMAT (valid JSON, no markdown):
{{
  "style_guide"    : "one-line visual style applied to ALL scenes",
  "voice_style"    : "gender, tone, pace — applied to ALL scenes",
  "thumbnail_prompt": "detailed image gen prompt for the video thumbnail",
  "color_palette"  : ["#hex1", "#hex2", "#hex3", "#hex4"],
  "font_suggestion": "font name and style note",
  "assets": [
    {{
      "scene_number"     : 1,
      "scene_title"      : "scene name",
      "image_prompt"     : "detailed static image generation prompt",
      "video_prompt"     : "motion/animation description for video gen",
      "voiceover_text"   : "exact narration text for this scene",
      "voice_style"      : "same as top-level voice_style",
      "background_music" : "calm / upbeat / dramatic / neutral / inspiring",
      "b_roll"           : "suggested stock footage or motion graphics description"
    }}
  ]
}}
"""

# ── Prompt template ────────────────────────────────────────────────────────────

VISUAL_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human",  HUMAN_PROMPT),
])


# ── Helpers ────────────────────────────────────────────────────────────────────

def _build_agent2_context(state: VideoState) -> str:
    """Serialize A2's output as clean JSON for the prompt."""
    # Reconstruct raw scene list from visual_briefs + full_script
    briefs: list[SceneBrief] = state.get("visual_briefs", [])
    script_lines = state.get("full_script", "").split("\n\n")

    # Build scene narration map from full_script blocks
    narration_map: dict[int, str] = {}
    for block in script_lines:
        lines = block.strip().split("\n", 1)
        if len(lines) == 2 and lines[0].startswith("[Scene"):
            try:
                num = int(lines[0].replace("[Scene ", "").replace("]", "").strip())
                narration_map[num] = lines[1].strip()
            except ValueError:
                pass

    scenes_out = []
    for brief in briefs:
        num = brief["scene_number"]
        scenes_out.append({
            "scene_number"      : num,
            "scene_title"       : brief["scene_title"],
            "narration"         : narration_map.get(num, ""),
            "visual_description": brief["visual_prompt"],
            "on_screen_text"    : brief["on_screen_text"],
        })

    context = {
        "video_title"           : state.get("title", ""),
        "domain"                : state.get("domain", ""),
        "narration_style"       : state.get("narration_style", "friendly and curious"),
        "duration_minutes"      : state.get("estimated_duration_min", 7),
        "cta"                   : state.get("cta", ""),
        "target_audience"       : state.get("target_audience", "students aged 15-25"),
        "analogy"               : state.get("analogy", ""),
        "scenes"                : scenes_out,
    }
    return json.dumps(context, indent=2)


def _merge_assets_into_briefs(
    briefs: list[SceneBrief],
    assets: list[dict],
    style_guide: str,
    voice_style: str,
) -> list[SceneBrief]:
    """
    Merge A3's enriched asset data back into the SceneBrief list.
    A3 adds: b_roll, and stores image/video/voice prompts.
    We extend SceneBrief with extra keys (TypedDict total=False allows this).
    """
    asset_map = {int(a.get("scene_number", 0)): a for a in assets}

    enriched: list[SceneBrief] = []
    for brief in briefs:
        num   = brief["scene_number"]
        asset = asset_map.get(num, {})

        enriched_brief = SceneBrief(
            scene_number   = num,
            scene_title    = brief["scene_title"],
            visual_prompt  = asset.get("image_prompt", brief["visual_prompt"]),
            b_roll         = asset.get("b_roll", ""),
            on_screen_text = brief["on_screen_text"],
        )
        # Store extra production fields alongside the brief
        enriched_brief["video_prompt"]    = asset.get("video_prompt", "")   # type: ignore
        enriched_brief["voiceover_text"]  = asset.get("voiceover_text", "") # type: ignore
        enriched_brief["voice_style"]     = voice_style                     # type: ignore
        enriched_brief["background_music"]= asset.get("background_music", "neutral") # type: ignore
        enriched_brief["style_guide"]     = style_guide                     # type: ignore
        enriched.append(enriched_brief)

    return enriched


# ── Agent node function ────────────────────────────────────────────────────────

def agent3_visual(state: VideoState) -> VideoState:
    """
    LangGraph node — Agent 3: Visual Generation Planner.

    Reads  : visual_briefs[], full_script, title, narration_style,
             cta, domain, target_audience from state
    Writes : visual_briefs[] (enriched), thumbnail_prompt,
             color_palette, font_suggestion
    Returns: updated VideoState
    """
    title = state.get("title", state.get("topic", ""))
    print(f"\n[A3] Planning visuals for: '{title}'")

    # ── Build prompt context ───────────────────────────────────────────────────
    agent2_context = _build_agent2_context(state)

    # ── Invoke LLM chain ──────────────────────────────────────────────────────
    llm = ChatAnthropic(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        api_key=ANTHROPIC_API_KEY,
    )
    chain  = VISUAL_PROMPT | llm | JsonOutputParser()
    raw: dict = chain.invoke({"agent2_context": agent2_context})

    # ── Validate required fields ───────────────────────────────────────────────
    required = ["style_guide", "voice_style", "thumbnail_prompt",
                "color_palette", "assets"]
    missing  = [k for k in required if k not in raw]
    if missing:
        raise ValueError(f"[A3] LLM response missing fields: {missing}")

    assets: list[dict] = raw["assets"]
    if not assets:
        raise ValueError("[A3] LLM returned empty assets list")

    # ── Merge asset data into existing SceneBriefs ─────────────────────────────
    existing_briefs = state.get("visual_briefs", [])
    enriched_briefs = _merge_assets_into_briefs(
        briefs      = existing_briefs,
        assets      = assets,
        style_guide = raw["style_guide"],
        voice_style = raw["voice_style"],
    )

    # ── Write into VideoState ──────────────────────────────────────────────────
    updates: VideoState = {
        "visual_briefs"   : enriched_briefs,
        "thumbnail_prompt": raw["thumbnail_prompt"],
        "color_palette"   : raw.get("color_palette", []),
        "font_suggestion" : raw.get("font_suggestion", ""),
    }

    print(f"[A3] Done — {len(enriched_briefs)} scene assets planned")
    print(f"[A3] Style guide : {raw['style_guide']}")
    print(f"[A3] Voice style : {raw['voice_style']}")
    print(f"[A3] Color palette: {raw.get('color_palette', [])}")
    print(f"[A3] Thumbnail   : {raw['thumbnail_prompt'][:80]}...")

    return {**state, **updates}


# ── Standalone test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from state import initial_state
    from agents.agent1_research import agent1_research
    from agents.agent2_script    import agent2_script

    TEST_TOPIC  = "What is Retrieval Augmented Generation (RAG)?"
    TEST_DOMAIN = "AI"

    print(f"\n{'='*60}")
    print(f"PIPELINE TEST: A1 → A2 → A3")
    print(f"Topic : {TEST_TOPIC}")
    print('='*60)

    s = initial_state(TEST_TOPIC, TEST_DOMAIN)
    s = agent1_research(s)
    s = agent2_script(s)
    s = agent3_visual(s)

    print(f"\n{'─'*60}")
    print(f"THUMBNAIL PROMPT : {s.get('thumbnail_prompt', '')[:120]}...")
    print(f"COLOR PALETTE    : {s.get('color_palette')}")
    print(f"FONT SUGGESTION  : {s.get('font_suggestion')}")
    print(f"\nPER-SCENE ASSETS ({len(s.get('visual_briefs', []))}):")
    for brief in s.get("visual_briefs", []):
        print(f"\n  Scene {brief['scene_number']}: {brief['scene_title']}")
        print(f"    Image  : {brief['visual_prompt'][:80]}...")
        print(f"    Video  : {brief.get('video_prompt', '')[:80]}...")  # type: ignore
        print(f"    Voice  : {brief.get('voiceover_text', '')[:60]}...") # type: ignore
        print(f"    Music  : {brief.get('background_music', '')}")       # type: ignore
        print(f"    B-roll : {brief.get('b_roll', '')[:60]}...")
