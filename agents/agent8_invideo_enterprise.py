"""
agents/agent8_invideo_enterprise.py

Agent 8: Enterprise InVideo AI Prompt Engineer
================================================
Role   : Expert AI video prompt engineer for enterprise and educational explainers
Goal   : Convert Agent 2's script into a Microsoft-style professional InVideo prompt
         optimized per domain — AI / Data Governance / SAP MDG
Input  : A2 + A3 output fields from VideoState
Writes : invideo_enterprise_prompt (str), invideo_enterprise_status

Key differences from Agent 7 (standard):
  - Domain-aware visual language (AI = neural nets / Data Gov = data lineage
    diagrams / SAP MDG = process flowcharts with SAP color conventions)
  - Enterprise color theme: white/blue (#0078D4 Microsoft blue, #FFFFFF, #F3F2F1)
  - Authoritative but accessible tone (15-yr-old level, professional register)
  - Scene pacing tightened to 5–7 seconds (vs 5–8 in A7)
  - Diagrams and flowcharts mandated over generic stock footage
  - Output is a plain string — StrOutputParser, no JSON

Architecture note:
  Agent 7 and Agent 8 can both run in the pipeline, producing
  two ready-to-use InVideo prompts: one consumer-style (A7),
  one enterprise-style (A8). The graph routes to both in parallel
  after A5 passes QA.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from state import VideoState, SceneBrief
from config import ANTHROPIC_API_KEY, LLM_MODEL

# ── InVideo API config (shared with agent7) ────────────────────────────────────
INVIDEO_API_KEY      = os.getenv("INVIDEO_API_KEY", "")
INVIDEO_API_ENDPOINT = "https://api.invideo.io/v1/generate"


# ── Domain-specific visual palettes ───────────────────────────────────────────

DOMAIN_VISUAL_GUIDES: dict[str, dict] = {
    "AI": {
        "color_theme"   : "white background, Microsoft blue (#0078D4) accents, dark text",
        "visual_style"  : "neural network diagrams, data flow arrows, node-edge graphs, "
                          "abstract AI brain icons, pipeline architecture diagrams",
        "diagram_types" : "transformer architecture diagrams, decision trees, "
                          "flowcharts showing model training steps",
        "icon_set"      : "circuit nodes, neural layers, data packets, model cards",
        "avoid"         : "generic robot stock footage, sci-fi aesthetics, neon colors",
    },
    "Data Governance": {
        "color_theme"   : "white background, deep blue (#003087) for primary, "
                          "teal (#008080) for data flows, light grey for containers",
        "visual_style"  : "data lineage diagrams, entity relationship models, "
                          "metadata hierarchy trees, policy shield icons",
        "diagram_types" : "data catalog architecture, lineage flow (source→transform→target), "
                          "governance framework pyramids, RACI charts",
        "icon_set"      : "database cylinders, lock shields, checkmark badges, "
                          "org hierarchy nodes",
        "avoid"         : "generic server racks, abstract swirls, stock office footage",
    },
    "SAP MDG": {
        "color_theme"   : "SAP blue (#0070F2), white, light grey (#F7F7F7), "
                          "with SAP Fiori-style flat UI components",
        "visual_style"  : "SAP Fiori tile layouts, master data hierarchy diagrams, "
                          "ERP process flowcharts, workflow approval chains",
        "diagram_types" : "material master data structure, business partner hierarchy, "
                          "MDG workflow: request→review→approve→activate, "
                          "consolidation hub diagrams",
        "icon_set"      : "SAP Fiori icons (flat, outlined), data record cards, "
                          "workflow arrows, approval stamps",
        "avoid"         : "generic ERP screenshots, complex UI mockups, SAP GUI screenshots",
    },
}

FALLBACK_VISUAL = DOMAIN_VISUAL_GUIDES["AI"]


# ── System prompt ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert AI video prompt engineer specializing in educational \
and enterprise explainer videos.
You produce Microsoft-style professional prompts for InVideo AI.
Your prompts are domain-aware: you use the correct visual language for
AI, Data Governance, or SAP MDG topics.

CRITICAL OUTPUT RULES:
- Output ONLY the final InVideo prompt string
- Do NOT explain anything
- Do NOT return JSON or markdown
- Do NOT add preamble like "Here is your prompt:"
- Start immediately with: "Create a professional YouTube explainer video with the following scenes:"
- End with Global Instructions and Topic/Audience lines
"""

# ── Human prompt — your specification, faithfully implemented ──────────────────

HUMAN_PROMPT = """\
INPUT (enterprise video script):
Title          : {title}
Topic          : {topic}
Domain         : {domain}
Target audience: {target_audience}
Duration target: {duration} minutes
Narration style: {narration_style}
CTA            : {cta}

DOMAIN VISUAL GUIDE for {domain}:
- Color theme   : {color_theme}
- Visual style  : {visual_style}
- Diagram types : {diagram_types}
- Icon set      : {icon_set}
- Avoid         : {avoid}

SCENES:
{scenes_block}

TASK: Convert the script into a professional InVideo AI prompt optimized for
{domain} content in an enterprise explainer format.

REQUIREMENTS:

1. SCENE STRUCTURE:
   - Break into scenes of 5–7 seconds each
   - Each scene must include:
       Narration:      [exact spoken words]
       Visual:         [specific diagram, chart, or animation — no generic stock]
       On-screen text: [short bold caption, max 8 words]

2. VISUAL STYLE:
   - Use diagrams, flowcharts, icons, and simple animations
   - Show data flow, system interactions, and process steps
   - Domain-specific visuals from the DOMAIN VISUAL GUIDE above
   - Avoid generic stock footage unless explicitly necessary

3. VOICE & TONE:
   - Professional, clear, slightly authoritative
   - Easy to understand (15-year-old level)
   - Consistent narrator voice throughout

4. GLOBAL SETTINGS:
   - Add subtitles
   - Add smooth transitions between scenes
   - Add light background music (low volume, corporate/neutral)
   - Use consistent color theme: {color_theme}

5. STRUCTURE:
   - Scene 1: Strong hook — curiosity-driven opening question
   - Middle scenes: Explanation + concrete example with domain-specific diagram
   - Final scene: Summary + call-to-action

OUTPUT FORMAT (follow exactly):
Create a professional YouTube explainer video with the following scenes:

Scene 1:
Narration: [exact words]
Visual: [enterprise diagram description]
On-screen text: [bold caption]

Scene 2:
...

Global Instructions:
- Use consistent visual style: {color_theme}
- Add subtitles
- Use AI voiceover (clear, neutral accent, professional tone)
- Add background music (low volume, corporate style)
- Keep video under 10 minutes
- Use smooth transitions

Topic: {topic}
Audience: Beginners in {domain}

IMPORTANT: Return ONLY the final prompt. No explanations. No JSON.
"""

# ── Prompt template ────────────────────────────────────────────────────────────

ENTERPRISE_PROMPT_TEMPLATE = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human",  HUMAN_PROMPT),
])


# ── Helpers ────────────────────────────────────────────────────────────────────

def _get_domain_guide(domain: str) -> dict:
    """Return the domain-specific visual guide dict."""
    return DOMAIN_VISUAL_GUIDES.get(domain, FALLBACK_VISUAL)


def _build_enterprise_scenes_block(state: VideoState) -> str:
    """
    Build a scene block emphasising diagram/flowchart visuals.
    Annotates each scene with domain-appropriate visual suggestions.
    """
    briefs: list[SceneBrief] = state.get("visual_briefs", [])
    domain = state.get("domain", "AI")
    guide  = _get_domain_guide(domain)

    # Narration map from full_script
    narration_map: dict[int, str] = {}
    for block in state.get("full_script", "").split("\n\n"):
        parts = block.strip().split("\n", 1)
        if len(parts) == 2 and parts[0].startswith("[Scene"):
            try:
                num = int(parts[0].replace("[Scene ", "").replace("]", "").strip())
                narration_map[num] = parts[1].strip()
            except ValueError:
                pass

    lines: list[str] = []
    for brief in briefs:
        num    = brief["scene_number"]
        title  = brief["scene_title"]
        narr   = narration_map.get(num, "")
        visual = brief.get("visual_prompt", "")
        text   = brief.get("on_screen_text", "")
        music  = brief.get("background_music", "corporate neutral")   # type: ignore

        # Enrich visual description with domain-aware diagram hint
        diagram_hint = (
            f"[{domain} style: use {guide['diagram_types'].split(',')[0].strip()}]"
            if not any(d in visual.lower() for d in ["diagram", "flowchart", "chart", "graph"])
            else ""
        )

        block = (
            f"Scene {num} — {title}:\n"
            f"  Narration     : {narr}\n"
            f"  Visual        : {visual} {diagram_hint}\n"
            f"  On-screen text: {text}\n"
            f"  Music energy  : {music}"
        )
        lines.append(block)

    if not lines:
        for i, scene_title in enumerate(state.get("scene_list", []), 1):
            lines.append(
                f"Scene {i} — {scene_title}:\n"
                f"  Narration     : [from script]\n"
                f"  Visual        : [enterprise diagram for {domain}]\n"
                f"  On-screen text: [key concept]"
            )

    return "\n\n".join(lines)


def send_enterprise_prompt(prompt: str, title: str, domain: str) -> dict:
    """
    Deliver the enterprise InVideo prompt.
    Manual mode (no API key): prints with enterprise formatting.
    API mode: POSTs with enterprise-specific parameters.
    """
    if not INVIDEO_API_KEY:
        sep = "═" * 65
        print(f"\n{sep}")
        print(f"ENTERPRISE INVIDEO PROMPT  [{domain}]")
        print(f"Copy and paste into: invideo.io/ai-video-generator")
        print(sep)
        print(prompt)
        print(sep)
        print(f"Title      : {title}")
        print(f"Domain     : {domain}")
        print(f"Characters : {len(prompt)}")
        print(f"Scenes     : {prompt.count('Scene ')}")
        print(sep + "\n")
        return {"status": "ready", "method": "manual",
                "char_count": len(prompt), "domain": domain}

    try:
        import urllib.request
        import json as _json

        payload = _json.dumps({
            "prompt"       : prompt,
            "title"        : title,
            "api_key"      : INVIDEO_API_KEY,
            "resolution"   : "1080p",
            "aspect_ratio" : "16:9",
            "style"        : "corporate",
            "domain"       : domain,
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
            print(f"[A8] InVideo enterprise job created — video_id: {video_id}")
            return {"status": "sent", "method": "api",
                    "video_id": video_id, "domain": domain}

    except Exception as exc:
        print(f"[A8] InVideo API error: {exc}")
        return {"status": "error", "method": "api",
                "error": str(exc), "domain": domain}


# ── Agent node function ────────────────────────────────────────────────────────

def agent8_invideo_enterprise(state: VideoState) -> VideoState:
    """
    LangGraph node — Agent 8: Enterprise InVideo Prompt Engineer.

    Reads  : visual_briefs[], full_script, title, topic, domain,
             narration_style, cta, target_audience from state
    Writes : invideo_enterprise_prompt, invideo_enterprise_status
    Returns: updated VideoState
    """
    title  = state.get("title", state.get("topic", ""))
    domain = state.get("domain", "AI")
    print(f"\n[A8] Generating enterprise InVideo prompt for: '{title}' [{domain}]")

    # ── Domain visual guide ────────────────────────────────────────────────────
    guide = _get_domain_guide(domain)

    # ── Build scene context ────────────────────────────────────────────────────
    scenes_block = _build_enterprise_scenes_block(state)

    # ── Invoke chain — plain string output ────────────────────────────────────
    llm = ChatAnthropic(
        model=LLM_MODEL,
        temperature=0.3,    # Enterprise = more deterministic, structured
        api_key=ANTHROPIC_API_KEY,
    )
    chain = ENTERPRISE_PROMPT_TEMPLATE | llm | StrOutputParser()

    enterprise_prompt: str = chain.invoke({
        "title"           : title,
        "topic"           : state.get("refined_topic", state.get("topic", "")),
        "domain"          : domain,
        "target_audience" : state.get("target_audience", "professionals and students"),
        "duration"        : state.get("estimated_duration_min", 7),
        "narration_style" : state.get("narration_style", "professional and clear"),
        "cta"             : state.get("cta", "Like, subscribe, and share with your team!"),
        "color_theme"     : guide["color_theme"],
        "visual_style"    : guide["visual_style"],
        "diagram_types"   : guide["diagram_types"],
        "icon_set"        : guide["icon_set"],
        "avoid"           : guide["avoid"],
        "scenes_block"    : scenes_block,
    })

    # ── Trim any accidental preamble ──────────────────────────────────────────
    if not enterprise_prompt.strip().startswith("Create a professional"):
        for i, line in enumerate(enterprise_prompt.strip().split("\n")):
            if line.strip().startswith("Create a professional"):
                enterprise_prompt = "\n".join(
                    enterprise_prompt.strip().split("\n")[i:]
                )
                break

    scene_count = enterprise_prompt.count("Scene ")
    char_count  = len(enterprise_prompt)
    print(f"[A8] Prompt generated — {scene_count} scenes, {char_count} characters [{domain}]")

    # ── Deliver prompt ────────────────────────────────────────────────────────
    result = send_enterprise_prompt(enterprise_prompt, title, domain)

    # ── Write into VideoState ─────────────────────────────────────────────────
    updates: VideoState = {}
    updates["invideo_enterprise_prompt"] = enterprise_prompt      # type: ignore
    updates["invideo_enterprise_status"] = result["status"]       # type: ignore
    if result.get("video_id"):
        updates["invideo_enterprise_video_id"] = result["video_id"]  # type: ignore

    print(f"[A8] Enterprise status: {result['status']} (method: {result.get('method')})")
    return {**state, **updates}


# ── Standalone test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from state import initial_state
    from agents.agent1_research import agent1_research
    from agents.agent2_script    import agent2_script
    from agents.agent3_visual    import agent3_visual
    from agents.agent4_qa        import agent4_qa
    from agents.agent5_publish   import agent5_publish

    # Test all three domains
    TEST_CASES = [
        ("What is Retrieval Augmented Generation?", "AI"),
        ("What is Data Lineage in Data Governance?", "Data Governance"),
        ("What is SAP MDG and why does it matter?",  "SAP MDG"),
    ]

    for topic, domain in TEST_CASES[:1]:   # Run first test only to save API calls
        print(f"\n{'='*65}")
        print(f"ENTERPRISE PIPELINE TEST: A1→A2→A3→A4→A5→A8")
        print(f"Topic : {topic}  [{domain}]")
        print('='*65)

        s = initial_state(topic, domain)
        s = agent1_research(s)
        s = agent2_script(s)
        s = agent3_visual(s)
        s = agent4_qa(s)
        s["qa_status"] = "pass"
        s = agent5_publish(s)
        s = agent8_invideo_enterprise(s)

        print(f"\n{'─'*65}")
        print(f"STATUS    : {s.get('invideo_enterprise_status')}")          # type: ignore
        print(f"DOMAIN    : {domain}")
        print(f"LENGTH    : {len(s.get('invideo_enterprise_prompt', ''))} chars") # type: ignore
        print(f"\nPROMPT PREVIEW (first 600 chars):")
        print(s.get("invideo_enterprise_prompt", "")[:600] + "...")         # type: ignore
