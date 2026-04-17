"""
agents/agent7_invideo.py

Agent 7: InVideo AI Prompt Engineer
======================================
Role   : Expert AI video prompt engineer
Goal   : Convert Agent 2's structured script into a single, ready-to-use
         InVideo AI prompt string — no JSON, no explanation, just the prompt
Input  : A2 output fields from VideoState (scenes, narration, visuals)
         + A3 enriched briefs (voice_style, style_guide, music)
Writes : invideo_prompt (str) — stored in state for sending / display
         invideo_status ("ready" | "sent" | "error")

Architecture note:
  Unlike A1–A6 which output JSON, this agent outputs a PLAIN STRING.
  The LLM is instructed to return raw text — not JSON, not markdown.
  We use a StrOutputParser instead of JsonOutputParser.

InVideo AI integration:
  Option A (manual)  — print the prompt; user pastes into invideo.io
  Option B (API)     — POST to InVideo API endpoint if INVIDEO_API_KEY is set
  The send_to_invideo() function handles both paths.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from state import VideoState, SceneBrief
from config import ANTHROPIC_API_KEY, LLM_MODEL, LLM_TEMPERATURE


# ── InVideo API config (optional) ─────────────────────────────────────────────
INVIDEO_API_KEY      = os.getenv("INVIDEO_API_KEY", "")
INVIDEO_API_ENDPOINT = "https://api.invideo.io/v1/generate"   # placeholder URL


# ── System prompt ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert AI video prompt engineer.
Your job is to convert a structured video script into a HIGH-QUALITY prompt
specifically optimized for InVideo AI.

CRITICAL RULES — read carefully:
- Output ONLY the final InVideo-ready prompt string
- Do NOT explain anything
- Do NOT return JSON
- Do NOT use markdown fences or headers
- Do NOT add preamble like "Here is your prompt:"
- Start immediately with: "Create a YouTube video with the following scenes:"
- End with the Global Instructions and Topic/Audience lines
"""

# ── Human prompt — your specification, faithfully implemented ──────────────────

HUMAN_PROMPT = """\
INPUT (structured video script):
Title          : {title}
Topic          : {topic}
Domain         : {domain}
Target audience: {target_audience}
Duration target: {duration} minutes
Narration style: {narration_style}
Style guide    : {style_guide}
Voice style    : {voice_style}
CTA            : {cta}

SCENES:
{scenes_block}

TASK: Transform the input into a detailed, scene-by-scene prompt that can be
directly used in InVideo AI to generate a professional YouTube video.

PROMPT REQUIREMENTS:
- Clearly define video purpose and audience
- Include full scene breakdown
- Each scene must include:
  * Narration (exact words to speak)
  * Visual description (what appears on screen)
  * On-screen text (caption or lower-third)
- Add global instructions for:
  * Style
  * Voiceover
  * Subtitles
  * Background music
- Keep scenes short (5–8 seconds each)
- Ensure visuals match narration precisely
- Maintain consistent style across all scenes

STYLE GUIDELINES:
- Clean, modern YouTube explainer style
- Use simple visuals (icons, animations, diagrams)
- Avoid complex cinematic instructions
- Focus on clarity and engagement

SPECIAL INSTRUCTIONS:
- First scene MUST be a strong hook
- Last scene MUST include summary + call-to-action
- Keep total video under 10 minutes

OUTPUT EXAMPLE STRUCTURE (follow this exact pattern):
Create a YouTube video with the following scenes:

Scene 1:
Narration: [exact words]
Visual: [what appears on screen]
On-screen text: [caption text]

Scene 2:
...

Global Instructions:
- Add subtitles
- Use AI voiceover (clear, neutral)
- Add light background music
- Use smooth transitions
- Keep pacing engaging

Topic: [topic]
Audience: Beginners aged 13-25

IMPORTANT: Output ONLY the prompt. Nothing else.
"""

# ── Prompt template ────────────────────────────────────────────────────────────

INVIDEO_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human",  HUMAN_PROMPT),
])


# ── Helpers ────────────────────────────────────────────────────────────────────

def _build_scenes_block(state: VideoState) -> str:
    """
    Format scenes for the prompt.
    Prefer enriched visual_briefs (A3 output); fall back to scene_list (A2).
    """
    briefs: list[SceneBrief] = state.get("visual_briefs", [])
    lines: list[str] = []

    # Extract narration map from full_script
    narration_map: dict[int, str] = {}
    for block in state.get("full_script", "").split("\n\n"):
        parts = block.strip().split("\n", 1)
        if len(parts) == 2 and parts[0].startswith("[Scene"):
            try:
                num = int(parts[0].replace("[Scene ", "").replace("]", "").strip())
                narration_map[num] = parts[1].strip()
            except ValueError:
                pass

    for brief in briefs:
        num     = brief["scene_number"]
        title   = brief["scene_title"]
        narr    = narration_map.get(num, "")
        visual  = brief.get("visual_prompt", "")
        text    = brief.get("on_screen_text", "")
        music   = brief.get("background_music", "neutral")    # type: ignore
        b_roll  = brief.get("b_roll", "")                     # type: ignore

        block = (
            f"Scene {num} — {title}:\n"
            f"  Narration     : {narr}\n"
            f"  Visual        : {visual}"
            + (f" | B-roll: {b_roll}" if b_roll else "") + "\n"
            f"  On-screen text: {text}\n"
            f"  Music energy  : {music}"
        )
        lines.append(block)

    if not lines:
        # Fallback: use scene_list strings
        for i, scene_title in enumerate(state.get("scene_list", []), 1):
            lines.append(
                f"Scene {i} — {scene_title}:\n"
                f"  Narration     : [from script]\n"
                f"  Visual        : [explainer animation]\n"
                f"  On-screen text: [key term]"
            )

    return "\n\n".join(lines)


def _extract_style_info(state: VideoState) -> tuple[str, str]:
    """Extract style_guide and voice_style from A3's enriched briefs."""
    briefs = state.get("visual_briefs", [])
    if briefs:
        style_guide = briefs[0].get("style_guide", "flat 2D minimalist")  # type: ignore
        voice_style = briefs[0].get("voice_style", "neutral, clear, conversational")  # type: ignore
    else:
        style_guide = "flat 2D minimalist, icon-based animations"
        voice_style = "neutral, clear, conversational pace"
    return style_guide, voice_style


# ── InVideo API sender ─────────────────────────────────────────────────────────

def send_to_invideo(prompt: str, title: str) -> dict:
    """
    Send the prompt to InVideo AI.

    Option A (no API key): prints the prompt for manual paste.
    Option B (API key set): POSTs to InVideo API endpoint.

    Returns a status dict.
    """
    if not INVIDEO_API_KEY:
        # Option A — manual path
        separator = "─" * 60
        print(f"\n{separator}")
        print("INVIDEO PROMPT — copy and paste into invideo.io/ai-video-generator")
        print(separator)
        print(prompt)
        print(separator)
        print(f"Title: {title}")
        print(f"Characters: {len(prompt)} | Scenes: {prompt.count('Scene ')}")
        print(separator + "\n")
        return {"status": "ready", "method": "manual", "char_count": len(prompt)}

    # Option B — API path
    try:
        import urllib.request
        import json as _json

        payload = _json.dumps({
            "prompt"    : prompt,
            "title"     : title,
            "api_key"   : INVIDEO_API_KEY,
            "resolution": "1080p",
            "aspect_ratio": "16:9",
        }).encode()

        req = urllib.request.Request(
            INVIDEO_API_ENDPOINT,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            result = _json.loads(resp.read())
            video_id = result.get("video_id", "")
            print(f"[A7] InVideo job created — video_id: {video_id}")
            return {"status": "sent", "method": "api", "video_id": video_id}

    except Exception as exc:
        print(f"[A7] InVideo API error: {exc}")
        return {"status": "error", "method": "api", "error": str(exc)}


# ── Agent node function ────────────────────────────────────────────────────────

def agent7_invideo(state: VideoState) -> VideoState:
    """
    LangGraph node — Agent 7: InVideo Prompt Engineer.

    Reads  : visual_briefs[], full_script, title, topic, domain,
             narration_style, cta, target_audience from state
    Writes : invideo_prompt, invideo_status (as extra state keys)
    Returns: updated VideoState
    """
    title = state.get("title", state.get("topic", ""))
    print(f"\n[A7] Generating InVideo prompt for: '{title}'")

    # ── Prepare inputs ─────────────────────────────────────────────────────────
    scenes_block        = _build_scenes_block(state)
    style_guide, voice_style = _extract_style_info(state)

    # ── Invoke chain — StrOutputParser returns plain text ─────────────────────
    llm = ChatAnthropic(
        model=LLM_MODEL,
        temperature=0.4,    # Lower = more deterministic prompt structure
        api_key=ANTHROPIC_API_KEY,
    )
    chain = INVIDEO_PROMPT_TEMPLATE | llm | StrOutputParser()

    invideo_prompt: str = chain.invoke({
        "title"           : title,
        "topic"           : state.get("refined_topic", state.get("topic", "")),
        "domain"          : state.get("domain", "AI"),
        "target_audience" : state.get("target_audience", "beginners aged 13-25"),
        "duration"        : state.get("estimated_duration_min", 7),
        "narration_style" : state.get("narration_style", "friendly and curious"),
        "style_guide"     : style_guide,
        "voice_style"     : voice_style,
        "cta"             : state.get("cta", "Like and subscribe for more!"),
        "scenes_block"    : scenes_block,
    })

    # ── Validate: must start with the required opening line ───────────────────
    if not invideo_prompt.strip().startswith("Create a YouTube video"):
        # Trim any accidental preamble
        lines = invideo_prompt.strip().split("\n")
        for i, line in enumerate(lines):
            if line.strip().startswith("Create a YouTube video"):
                invideo_prompt = "\n".join(lines[i:])
                break

    scene_count = invideo_prompt.count("Scene ")
    char_count  = len(invideo_prompt)
    print(f"[A7] Prompt generated — {scene_count} scenes, {char_count} characters")

    # ── Send to InVideo (or print for manual use) ─────────────────────────────
    send_result = send_to_invideo(invideo_prompt, title)

    # ── Write into VideoState ──────────────────────────────────────────────────
    updates: VideoState = {}
    updates["invideo_prompt"]  = invideo_prompt      # type: ignore
    updates["invideo_status"]  = send_result["status"]  # type: ignore
    if send_result.get("video_id"):
        updates["invideo_video_id"] = send_result["video_id"]  # type: ignore

    print(f"[A7] InVideo status: {send_result['status']} "
          f"(method: {send_result.get('method')})")

    return {**state, **updates}


# ── Convenience: generate prompt only (no state required) ─────────────────────

def generate_invideo_prompt_only(
    title: str,
    topic: str,
    domain: str,
    scenes: list[dict],
    style_guide: str = "flat 2D minimalist, icon-based animations",
    voice_style: str = "neutral, clear, conversational",
    cta: str = "Like and subscribe for more videos!",
    duration: int = 7,
) -> str:
    """
    Standalone helper — generate an InVideo prompt without running
    the full pipeline. Useful for quick testing or one-off generation.

    scenes format: [{"scene_number": 1, "scene_title": "...",
                     "narration": "...", "visual": "...", "on_screen_text": "..."}]
    """
    scenes_block = "\n\n".join(
        f"Scene {s['scene_number']} — {s.get('scene_title', '')}:\n"
        f"  Narration     : {s.get('narration', '')}\n"
        f"  Visual        : {s.get('visual', '')}\n"
        f"  On-screen text: {s.get('on_screen_text', '')}"
        for s in scenes
    )

    llm   = ChatAnthropic(model=LLM_MODEL, temperature=0.4, api_key=ANTHROPIC_API_KEY)
    chain = INVIDEO_PROMPT_TEMPLATE | llm | StrOutputParser()

    return chain.invoke({
        "title": title, "topic": topic, "domain": domain,
        "target_audience": "beginners aged 13-25",
        "duration": duration, "narration_style": "friendly and curious",
        "style_guide": style_guide, "voice_style": voice_style,
        "cta": cta, "scenes_block": scenes_block,
    })


# ── Standalone test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from state import initial_state
    from agents.agent1_research import agent1_research
    from agents.agent2_script    import agent2_script
    from agents.agent3_visual    import agent3_visual
    from agents.agent4_qa        import agent4_qa
    from agents.agent5_publish   import agent5_publish

    TEST_TOPIC  = "What is Retrieval Augmented Generation (RAG)?"
    TEST_DOMAIN = "AI"

    print(f"\n{'='*60}")
    print(f"PIPELINE TEST: A1 → A2 → A3 → A4 → A5 → A7")
    print(f"Topic : {TEST_TOPIC}")
    print('='*60)

    s = initial_state(TEST_TOPIC, TEST_DOMAIN)
    s = agent1_research(s)
    s = agent2_script(s)
    s = agent3_visual(s)
    s = agent4_qa(s)
    s["qa_status"] = "pass"
    s = agent5_publish(s)
    s = agent7_invideo(s)

    print(f"\n{'─'*60}")
    print(f"INVIDEO STATUS: {s.get('invideo_status')}")          # type: ignore
    print(f"PROMPT LENGTH : {len(s.get('invideo_prompt', ''))} chars")  # type: ignore
    print(f"\nPROMPT PREVIEW (first 500 chars):")
    print(s.get("invideo_prompt", "")[:500] + "...")              # type: ignore
