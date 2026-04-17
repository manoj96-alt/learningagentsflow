"""
agents/agent5_publish.py

Agent 5: Publishing Agent
===========================
Role   : YouTube SEO and publishing expert
Goal   : Generate YouTube metadata, push to Notion, save local JSON,
         assign publish date from the content calendar
Input  : QA-approved VideoState (qa_status == "pass")
Writes : notion_page_id, publish_date, publish_day,
         local_json_path, status="scheduled"
         Also enriches: title (SEO-optimised), plus adds metadata fields

SEO output fields (stored as extra keys on state):
  yt_title, yt_description, yt_tags, yt_hashtags,
  thumbnail_text, upload_schedule

Design notes:
  - SEO prompt is faithfully implemented from Manoj's specification
  - Notion push is gated: skipped gracefully if NOTION_TOKEN not set
  - Local JSON is always written regardless of Notion status
  - Calendar assigns next available Tuesday or Friday slot
"""

import json
import os
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from state import VideoState
from config import (
    ANTHROPIC_API_KEY, LLM_MODEL, LLM_TEMPERATURE,
    NOTION_TOKEN, NOTION_DATABASE_ID,
    OUTPUT_DIR, PUBLISH_DAYS,
)


# ── Calendar helpers ──────────────────────────────────────────────────────────

DAY_MAP = {
    "Monday": 0, "Tuesday": 1, "Wednesday": 2,
    "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6,
}
CALENDAR_FILE = Path(OUTPUT_DIR) / "content_calendar.json"


def _load_calendar() -> list[dict]:
    if CALENDAR_FILE.exists():
        return json.loads(CALENDAR_FILE.read_text())
    return []


def _save_calendar(calendar: list[dict]) -> None:
    CALENDAR_FILE.parent.mkdir(parents=True, exist_ok=True)
    CALENDAR_FILE.write_text(json.dumps(calendar, indent=2))


def _next_publish_slot() -> tuple[str, str]:
    """Return (ISO date string, day name) for the next free Tue or Fri slot."""
    calendar    = _load_calendar()
    booked      = {e["publish_date"] for e in calendar}
    target_days = [DAY_MAP[d] for d in PUBLISH_DAYS]
    candidate   = date.today() + timedelta(days=1)
    while True:
        if candidate.weekday() in target_days and candidate.isoformat() not in booked:
            return candidate.isoformat(), candidate.strftime("%A")
        candidate += timedelta(days=1)


# ── System prompt ──────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a YouTube SEO and publishing expert.
You specialise in educational content for audiences aged 13–25.
You write titles and descriptions that rank well in YouTube search
while staying honest and accurate.
Always respond with valid JSON only — no markdown fences, no preamble.
"""

# ── Human prompt — your specification, faithfully implemented ──────────────────

HUMAN_PROMPT = """\
INPUT:
- Video Title (from script): {title}
- Domain                   : {domain}
- Topic                    : {topic}
- Hook                     : {hook}
- Key Facts                : {key_facts}
- Script Summary           : {script_summary}
- Target Audience          : {target_audience}
- Upload Schedule          : {upload_schedule}
- Thumbnail brief          : {thumbnail_brief}

TASK: Generate complete metadata for YouTube publishing.

RULES:
- Title must be catchy, SEO-friendly, and under 60 characters
- Description must include primary keywords naturally (not stuffed)
- Description must be 150–300 words
- Tags: 10–15 relevant tags, mix of broad and specific
- Hashtags: 3–5 relevant hashtags (include the # symbol)
- Thumbnail text: max 5 words, bold impact — what the viewer sees first
- Keep thumbnail text short and bold — one punchy phrase
- upload_schedule: use the provided schedule date/time

OUTPUT FORMAT (valid JSON, no markdown):
{{
  "yt_title"        : "final SEO-optimised title under 60 chars",
  "yt_description"  : "150–300 word description with keywords naturally woven in",
  "yt_tags"         : ["tag1", "tag2", "tag3"],
  "yt_hashtags"     : ["#Tag1", "#Tag2", "#Tag3"],
  "thumbnail_text"  : "MAX 5 WORDS",
  "upload_schedule" : "{upload_schedule}",
  "seo_keywords"    : ["primary keyword", "secondary keyword"]
}}
"""

# ── Prompt template ────────────────────────────────────────────────────────────

PUBLISH_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human",  HUMAN_PROMPT),
])


# ── Notion helpers ─────────────────────────────────────────────────────────────

def _rt(text: str, bold: bool = False) -> dict:
    obj: dict = {"type": "text", "text": {"content": str(text)}}
    if bold:
        obj["annotations"] = {"bold": True}
    return obj


def _heading2(text: str) -> dict:
    return {"object": "block", "type": "heading_2",
            "heading_2": {"rich_text": [_rt(text)]}}


def _paragraph(text: str) -> dict:
    return {"object": "block", "type": "paragraph",
            "paragraph": {"rich_text": [_rt(text)]}}


def _bullet(text: str) -> dict:
    return {"object": "block", "type": "bulleted_list_item",
            "bulleted_list_item": {"rich_text": [_rt(text)]}}


def _divider() -> dict:
    return {"object": "block", "type": "divider", "divider": {}}


def _build_notion_blocks(state: VideoState, metadata: dict) -> list[dict]:
    """Build rich Notion page blocks from full VideoState + SEO metadata."""
    blocks = []

    # SEO metadata
    blocks += [
        _heading2("YouTube Metadata"),
        _paragraph(f"Title: {metadata.get('yt_title', '')}"),
        _paragraph(f"Schedule: {metadata.get('upload_schedule', '')}"),
        _paragraph(f"Thumbnail text: {metadata.get('thumbnail_text', '')}"),
        _heading2("Description"),
        _paragraph(metadata.get("yt_description", "")),
        _heading2("Tags & Hashtags"),
        _bullet("Tags: " + ", ".join(metadata.get("yt_tags", []))),
        _bullet("Hashtags: " + " ".join(metadata.get("yt_hashtags", []))),
        _divider(),
    ]

    # Hook + key facts
    blocks += [
        _heading2("Hook"),
        _paragraph(state.get("hook", "")),
        _heading2("Key Facts"),
        *[_bullet(f) for f in state.get("key_facts", [])],
        _divider(),
    ]

    # Full script
    blocks += [
        _heading2("Full Script"),
        _paragraph(state.get("full_script", "")),
        _divider(),
    ]

    # Visual briefs summary
    briefs = state.get("visual_briefs", [])
    if briefs:
        blocks.append(_heading2("Visual Briefs"))
        for b in briefs:
            blocks.append(_bullet(
                f"Scene {b['scene_number']} — {b['scene_title']}: "
                f"{b.get('visual_prompt', '')[:120]}"
            ))
        blocks.append(_divider())

    # Thumbnail prompt
    blocks += [
        _heading2("Thumbnail Prompt"),
        _paragraph(state.get("thumbnail_prompt", "")),
        _divider(),
    ]

    # QA scores
    qa = state.get("qa_scores", {})
    if qa:
        blocks += [
            _heading2("QA Scores"),
            _bullet(f"Accuracy : {qa.get('accuracy', 0):.2f}"),
            _bullet(f"Tone     : {qa.get('tone',     0):.2f}"),
            _bullet(f"SEO      : {qa.get('seo',      0):.2f}"),
            _bullet(f"Length   : {qa.get('length',   0):.2f}"),
        ]

    return blocks


def _push_to_notion(state: VideoState, metadata: dict, publish_date: str) -> str:
    """Create a Notion database entry. Returns page_id or '' on failure."""
    try:
        from notion_client import Client
        notion = Client(auth=NOTION_TOKEN)

        props = {
            "Name"          : {"title": [{"text": {"content": metadata.get("yt_title", state.get("title", ""))}}]},
            "Domain"        : {"select": {"name": state.get("domain", "AI")}},
            "Status"        : {"select": {"name": "Scheduled"}},
            "Publish Date"  : {"date": {"start": publish_date}},
            "Duration (min)": {"number": state.get("estimated_duration_min", 7)},
        }

        page = notion.pages.create(
            parent={"database_id": NOTION_DATABASE_ID},
            properties=props,
            children=_build_notion_blocks(state, metadata),
        )
        page_id: str = page["id"]
        print(f"[A5] Notion page created: {page['url']}")
        return page_id

    except ImportError:
        print("[A5] notion-client not installed — skipping Notion push")
        return ""
    except Exception as exc:
        print(f"[A5] Notion push failed: {exc}")
        return ""


def _save_local_json(state: VideoState, metadata: dict, publish_date: str) -> str:
    """Save the complete video package as a local JSON file."""
    out_dir = Path(OUTPUT_DIR) / "scripts"
    out_dir.mkdir(parents=True, exist_ok=True)

    slug = state.get("title", "video").lower()
    for ch in ' /\\:*?"<>|':
        slug = slug.replace(ch, "_")
    slug = slug[:60]

    filename  = f"{publish_date}_{slug}.json"
    filepath  = out_dir / filename

    payload = {
        "topic"              : state.get("topic"),
        "domain"             : state.get("domain"),
        "title"              : state.get("title"),
        "hook"               : state.get("hook"),
        "simple_explanation" : state.get("simple_explanation"),
        "analogy"            : state.get("analogy"),
        "real_world_example" : state.get("real_world_example"),
        "key_facts"          : state.get("key_facts"),
        "full_script"        : state.get("full_script"),
        "scene_list"         : state.get("scene_list"),
        "visual_briefs"      : state.get("visual_briefs"),
        "thumbnail_prompt"   : state.get("thumbnail_prompt"),
        "color_palette"      : state.get("color_palette"),
        "qa_scores"          : state.get("qa_scores"),
        "publish_date"       : publish_date,
        "seo_metadata"       : metadata,
    }
    filepath.write_text(json.dumps(payload, indent=2))
    print(f"[A5] Saved → {filepath}")
    return str(filepath)


def _update_calendar(state: VideoState, metadata: dict,
                     publish_date: str, publish_day: str,
                     notion_page_id: str, local_json_path: str) -> None:
    """Append this video to the local content calendar."""
    calendar = _load_calendar()
    calendar.append({
        "title"          : metadata.get("yt_title", state.get("title", "")),
        "domain"         : state.get("domain", ""),
        "publish_date"   : publish_date,
        "publish_day"    : publish_day,
        "status"         : "scheduled",
        "notion_page_id" : notion_page_id,
        "local_json_path": local_json_path,
        "created_at"     : date.today().isoformat(),
    })
    _save_calendar(calendar)
    print(f"[A5] Calendar updated — {len(calendar)} total entries")


# ── Agent node function ────────────────────────────────────────────────────────

def agent5_publish(state: VideoState) -> VideoState:
    """
    LangGraph node — Agent 5: Publishing.

    Reads  : QA-approved VideoState
    Writes : notion_page_id, publish_date, publish_day,
             local_json_path, status + SEO metadata fields
    Returns: updated VideoState
    """
    title = state.get("title", state.get("topic", ""))
    print(f"\n[A5] Publishing: '{title}'")

    # ── Build script summary for the SEO prompt ────────────────────────────────
    key_facts = state.get("key_facts", [])
    script_summary = ". ".join(key_facts[:3]) if key_facts else state.get("simple_explanation", "")

    # ── Determine publish slot ─────────────────────────────────────────────────
    publish_date, publish_day = _next_publish_slot()
    upload_schedule = f"next {publish_day} 6PM PST ({publish_date})"
    print(f"[A5] Scheduled for: {upload_schedule}")

    # ── Invoke SEO chain ───────────────────────────────────────────────────────
    llm = ChatAnthropic(
        model=LLM_MODEL,
        temperature=0.5,
        api_key=ANTHROPIC_API_KEY,
    )
    chain  = PUBLISH_PROMPT | llm | JsonOutputParser()
    metadata: dict = chain.invoke({
        "title"           : title,
        "domain"          : state.get("domain", "AI"),
        "topic"           : state.get("refined_topic", state.get("topic", "")),
        "hook"            : state.get("hook", ""),
        "key_facts"       : json.dumps(key_facts),
        "script_summary"  : script_summary,
        "target_audience" : state.get("target_audience", "students aged 15-25"),
        "upload_schedule" : upload_schedule,
        "thumbnail_brief" : state.get("thumbnail_prompt", ""),
    })

    # ── Validate metadata ──────────────────────────────────────────────────────
    required = ["yt_title", "yt_description", "yt_tags", "yt_hashtags", "thumbnail_text"]
    missing  = [k for k in required if k not in metadata]
    if missing:
        raise ValueError(f"[A5] SEO metadata missing fields: {missing}")

    # ── Push to Notion (if configured) ────────────────────────────────────────
    notion_page_id = ""
    if NOTION_TOKEN and NOTION_DATABASE_ID:
        notion_page_id = _push_to_notion(state, metadata, publish_date)
    else:
        print("[A5] Notion not configured — skipping push")

    # ── Save local JSON ────────────────────────────────────────────────────────
    local_json_path = _save_local_json(state, metadata, publish_date)

    # ── Update calendar ────────────────────────────────────────────────────────
    _update_calendar(state, metadata, publish_date, publish_day,
                     notion_page_id, local_json_path)

    # ── Log SEO output ─────────────────────────────────────────────────────────
    print(f"[A5] YT Title       : {metadata.get('yt_title')}")
    print(f"[A5] Thumbnail text : {metadata.get('thumbnail_text')}")
    print(f"[A5] Tags           : {', '.join(metadata.get('yt_tags', [])[:5])}...")
    print(f"[A5] Hashtags       : {' '.join(metadata.get('yt_hashtags', []))}")

    # ── Write into VideoState ──────────────────────────────────────────────────
    updates: VideoState = {
        "title"           : metadata.get("yt_title", title),
        "notion_page_id"  : notion_page_id,
        "publish_date"    : publish_date,
        "publish_day"     : publish_day,
        "local_json_path" : local_json_path,
        "status"          : "scheduled",          # type: ignore[typeddict-item]
    }
    # Store SEO metadata as extra keys (TypedDict total=False permits this)
    updates["yt_title"]       = metadata.get("yt_title", "")         # type: ignore
    updates["yt_description"] = metadata.get("yt_description", "")   # type: ignore
    updates["yt_tags"]        = metadata.get("yt_tags", [])          # type: ignore
    updates["yt_hashtags"]    = metadata.get("yt_hashtags", [])      # type: ignore
    updates["thumbnail_text"] = metadata.get("thumbnail_text", "")   # type: ignore
    updates["upload_schedule"]= metadata.get("upload_schedule", "")  # type: ignore

    return {**state, **updates}


# ── Standalone test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from state import initial_state
    from agents.agent1_research import agent1_research
    from agents.agent2_script    import agent2_script
    from agents.agent3_visual    import agent3_visual
    from agents.agent4_qa        import agent4_qa

    TEST_TOPIC  = "What is Retrieval Augmented Generation (RAG)?"
    TEST_DOMAIN = "AI"

    print(f"\n{'='*60}")
    print(f"PIPELINE TEST: A1 → A2 → A3 → A4 → A5")
    print(f"Topic : {TEST_TOPIC}")
    print('='*60)

    s = initial_state(TEST_TOPIC, TEST_DOMAIN)
    s = agent1_research(s)
    s = agent2_script(s)
    s = agent3_visual(s)
    s = agent4_qa(s)

    # Force pass for test (skip QA retry logic)
    s["qa_status"] = "pass"

    s = agent5_publish(s)

    print(f"\n{'─'*60}")
    print(f"STATUS        : {s.get('status')}")
    print(f"PUBLISH DATE  : {s.get('publish_date')} ({s.get('publish_day')})")
    print(f"YT TITLE      : {s.get('yt_title')}")                   # type: ignore
    print(f"THUMBNAIL TEXT: {s.get('thumbnail_text')}")              # type: ignore
    print(f"TAGS          : {s.get('yt_tags', [])[:5]}")             # type: ignore
    print(f"LOCAL JSON    : {s.get('local_json_path')}")
    print(f"NOTION PAGE   : {s.get('notion_page_id') or 'not configured'}")
    print(f"\nDESCRIPTION PREVIEW:")
    desc = s.get("yt_description", "")                               # type: ignore
    print(desc[:300] + "..." if len(desc) > 300 else desc)
