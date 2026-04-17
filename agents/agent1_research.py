"""
agents/agent1_research.py

Agent 1: Research + Simplification
===================================
Role    : Expert AI educator and storyteller
Goal    : Research a topic and simplify it so a 15-year-old can understand it
Input   : state["topic"], state["domain"]
Writes  : refined_topic, simple_explanation, analogy, real_world_example,
          key_facts, target_audience, research_confidence,
          title (draft), hook (draft), scene_list (story_flow outline)

Design notes:
  - Prompt is faithfully implemented from Manoj's specification
  - Uses structured JSON output → validated → written into VideoState
  - research_confidence is a self-rating the agent produces;
    route_after_research() uses it to decide whether to re-run
  - story_flow becomes the seed for Agent 2's scene_list
"""

import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from state import VideoState
from config import ANTHROPIC_API_KEY, LLM_MODEL, LLM_TEMPERATURE


# ── System prompt — your specification, faithfully implemented ─────────────────

SYSTEM_PROMPT = """\
You are an expert AI educator and storyteller.
Your task is to research and simplify a given topic so that a 15-year-old \
can understand it easily.

RULES:
- Avoid jargon unless explained immediately after
- Keep sentences short (max 20 words each)
- Make it engaging and storytelling-based
- Ensure accuracy — no hallucinations, no invented facts
- Always respond with valid JSON only — no markdown fences, no preamble
"""

# ── Human prompt — your INPUT / TASKS / OUTPUT FORMAT spec ────────────────────

HUMAN_PROMPT = """\
INPUT:
- Topic  : {topic}
- Domain : {domain}

TASKS:
1. Research the topic (use your knowledge, no browsing needed)
2. Simplify the concept using:
   * Simple language
   * Real-world analogy
3. Create a clear storyline for a 5–10 minute video

OUTPUT FORMAT (valid JSON, no markdown):
{{
  "refined_topic"         : "sharpened, specific topic title",
  "title"                 : "catchy SEO-friendly YouTube title (max 60 chars)",
  "hook"                  : "attention-grabbing first sentence — max 15 words",
  "simple_explanation"    : "plain-language explanation, 3–5 short sentences",
  "analogy"               : "one vivid real-world analogy that makes it click",
  "real_world_example"    : "concrete relatable example from everyday life",
  "key_facts"             : [
      "fact 1 — one sentence",
      "fact 2 — one sentence",
      "fact 3 — one sentence"
  ],
  "story_flow"            : [
      "Intro   : ...",
      "Problem : ...",
      "Solution: ...",
      "Example : ...",
      "Summary : ..."
  ],
  "target_audience"       : "who will benefit most from this video",
  "research_confidence"   : 0.95,
  "estimated_duration_min": 7
}}

RULES (repeat — these override everything else):
- Avoid jargon unless explained
- Keep sentences short
- Make it engaging and storytelling-based
- Ensure accuracy (no hallucinations)
"""


# ── Prompt template ────────────────────────────────────────────────────────────

RESEARCH_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human",  HUMAN_PROMPT),
])


# ── Agent node function ────────────────────────────────────────────────────────

def agent1_research(state: VideoState) -> VideoState:
    """
    LangGraph node — Agent 1: Research + Simplification.

    Reads  : state["topic"], state["domain"]
    Writes : research fields + draft title/hook/story_flow into state
    Returns: updated VideoState (LangGraph merges it)
    """
    topic  = state.get("topic", "")
    domain = state.get("domain", "AI")

    print(f"\n[A1] Researching: '{topic}' [{domain}]")

    # ── Build and invoke the chain ─────────────────────────────────────────────
    llm = ChatAnthropic(
        model=LLM_MODEL,
        temperature=LLM_TEMPERATURE,
        api_key=ANTHROPIC_API_KEY,
    )

    chain = RESEARCH_PROMPT | llm | JsonOutputParser()

    raw: dict = chain.invoke({
        "topic" : topic,
        "domain": domain,
    })

    # ── Validate required fields ───────────────────────────────────────────────
    required = [
        "refined_topic", "title", "hook", "simple_explanation",
        "analogy", "real_world_example", "key_facts",
        "story_flow", "target_audience", "research_confidence",
    ]
    missing = [k for k in required if k not in raw]
    if missing:
        raise ValueError(f"[A1] LLM response missing fields: {missing}\nRaw: {raw}")

    # ── Normalise types ────────────────────────────────────────────────────────
    confidence = float(raw.get("research_confidence", 0.9))
    duration   = int(raw.get("estimated_duration_min", 7))

    # ── Write into VideoState ──────────────────────────────────────────────────
    updates: VideoState = {
        # Research fields
        "refined_topic"        : raw["refined_topic"],
        "simple_explanation"   : raw["simple_explanation"],
        "analogy"              : raw["analogy"],
        "real_world_example"   : raw["real_world_example"],
        "key_facts"            : raw["key_facts"],
        "target_audience"      : raw["target_audience"],
        "research_confidence"  : confidence,

        # Draft fields that A2 will expand
        "title"                : raw["title"],
        "hook"                 : raw["hook"],
        "scene_list"           : raw["story_flow"],          # A2 expands this

        # Carry forward
        "estimated_duration_min": duration,
        "status"               : "in_progress",
    }

    print(f"[A1] Done — refined topic: '{raw['refined_topic']}'")
    print(f"[A1] Confidence: {confidence:.2f} | Est. duration: {duration} min")
    print(f"[A1] Hook draft: {raw['hook']}")

    return {**state, **updates}


# ── Standalone test ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    from state import initial_state
    import json

    test_cases = [
        ("What is RAG in AI?",              "AI"),
        ("What is Master Data Governance?", "Data Governance"),
        ("What is SAP MDG?",                "SAP MDG"),
    ]

    for topic, domain in test_cases:
        print(f"\n{'='*60}")
        print(f"TEST: {topic} [{domain}]")
        print('='*60)
        s = initial_state(topic, domain)
        result = agent1_research(s)

        print(f"\n  Refined topic  : {result['refined_topic']}")
        print(f"  Title          : {result['title']}")
        print(f"  Hook           : {result['hook']}")
        print(f"  Explanation    : {result['simple_explanation'][:120]}...")
        print(f"  Analogy        : {result['analogy'][:100]}...")
        print(f"  Key facts      :")
        for f in result['key_facts']:
            print(f"    • {f}")
        print(f"  Story flow     :")
        for s_item in result['scene_list']:
            print(f"    → {s_item}")
        print(f"  Confidence     : {result['research_confidence']}")
