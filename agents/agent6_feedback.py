"""
agents/agent6_feedback.py

Agent 6: Feedback + Growth Engine
====================================
Role   : YouTube growth analyst
Goal   : Analyse video performance (analytics + comments), extract lessons,
         generate next-topic suggestions that feed back into the pipeline
Input  : VideoFeedback dict (views, ctr, watch_time, retention, top_comments)
         + full VideoState context (topic, domain, title, key_facts)
Writes : feedback_summary, lessons_learned, next_topic_suggestions,
         feedback (enriched VideoFeedback with sentiment + growth_score)

Growth score formula (0.0–1.0 composite):
  0.35 × normalised_views   (benchmark: 1000 views = 1.0)
  0.30 × ctr_score          (benchmark: 5% CTR = 1.0)
  0.25 × retention_score    (watch_time_percent / 100)
  0.10 × sentiment_score    (positive=1.0, mixed=0.5, negative=0.0)

Feedback loop:
  next_topic_suggestions → written into state
  → graph.py reads them and seeds new pipeline runs (one per suggestion)
"""

import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from state import VideoState, VideoFeedback
from config import ANTHROPIC_API_KEY, LLM_MODEL, LLM_TEMPERATURE


# ── Growth score benchmarks ───────────────────────────────────────────────────

VIEW_BENCHMARK      = 1000.0    # views at which normalised_views = 1.0
CTR_BENCHMARK       = 0.05      # 5% CTR = 1.0
SENTIMENT_MAP       = {"positive": 1.0, "mixed": 0.5, "negative": 0.0}


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are a YouTube growth analyst specialising in educational content for teenagers.
You analyse video performance data and viewer comments to extract actionable insights.
You are data-driven but also sensitive to qualitative signals in comments.
You always tie performance insights back to specific content decisions.
Always respond with valid JSON only — no markdown fences, no preamble.
"""

# ── Human prompt — your specification, faithfully implemented ─────────────────

HUMAN_PROMPT = """\
INPUT:
Video metrics:
{video_metrics}

Video context:
- Title        : {title}
- Topic        : {topic}
- Domain       : {domain}
- Key facts    : {key_facts}
- Hook used    : {hook}
- Narr. style  : {narration_style}
- Duration     : {duration} minutes
- Publish date : {publish_date}

Top viewer comments ({comment_count} total):
{top_comments}

Computed growth score: {growth_score:.2f} / 1.0
(0.35×views + 0.30×CTR + 0.25×retention + 0.10×sentiment)

TASK: Analyse performance and suggest improvements.

RULES:
- Focus on improving engagement in future videos
- Suggest actionable improvements — name the specific element to change
- Tie every insight to a specific metric or comment (quote the comment if relevant)
- next_video_suggestions must be on the SAME domain: {domain}
- Suggest 3–5 follow-up topics that build on what worked in this video
- optimization_tips must be concrete — not generic ("improve thumbnail")
  but specific ("add a question mark to thumbnail text to boost curiosity CTR")

OUTPUT FORMAT (valid JSON, no markdown):
{{
  "performance_summary"    : "2–3 sentence plain-language summary of how the video performed",
  "sentiment"              : "positive / mixed / negative",
  "what_worked"            : [
      "specific element that drove engagement, with metric evidence"
  ],
  "what_failed"            : [
      "specific element that hurt retention or CTR, with metric evidence"
  ],
  "next_video_suggestions" : [
      "Topic title — one-line rationale tied to this video's performance"
  ],
  "optimization_tips"      : [
      "specific, actionable tip for the next video"
  ],
  "lessons_learned"        : [
      "one transferable lesson from this video's performance"
  ]
}}
"""

# ── Prompt template ────────────────────────────────────────────────────────────

FEEDBACK_PROMPT = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human",  HUMAN_PROMPT),
])


# ── Growth score calculator ────────────────────────────────────────────────────

def compute_growth_score(feedback: VideoFeedback) -> float:
    """
    Compute a 0.0–1.0 composite growth score from raw metrics.
    Uses fixed benchmarks so the score is comparable across videos.
    """
    views     = feedback.get("views", 0)
    ctr       = feedback.get("ctr", 0.0)
    retention = feedback.get("watch_time_percent", 0.0)
    sentiment = feedback.get("sentiment", "mixed")

    view_score      = min(1.0, views / VIEW_BENCHMARK)
    ctr_score       = min(1.0, ctr / CTR_BENCHMARK)
    retention_score = min(1.0, retention / 100.0)
    sentiment_score = SENTIMENT_MAP.get(sentiment, 0.5)

    score = (
        0.35 * view_score
        + 0.30 * ctr_score
        + 0.25 * retention_score
        + 0.10 * sentiment_score
    )
    return round(score, 3)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _build_metrics_context(feedback: VideoFeedback) -> str:
    """Format raw metrics as clean JSON for the prompt."""
    return json.dumps({
        "views"              : feedback.get("views", 0),
        "ctr_percent"        : f"{feedback.get('ctr', 0.0) * 100:.1f}%",
        "watch_time_percent" : f"{feedback.get('watch_time_percent', 0.0):.1f}%",
        "sentiment"          : feedback.get("sentiment", "mixed"),
    }, indent=2)


def _format_comments(comments: list[str]) -> str:
    """Format top comments as a numbered list for the prompt."""
    if not comments:
        return "(no comments yet)"
    return "\n".join(f"{i+1}. \"{c}\"" for i, c in enumerate(comments[:10]))


# ── Agent node function ────────────────────────────────────────────────────────

def agent6_feedback(state: VideoState) -> VideoState:
    """
    LangGraph node — Agent 6: Feedback + Growth Engine.

    Reads  : state["feedback"] (VideoFeedback dict with metrics + comments)
             + VideoState context fields
    Writes : feedback (enriched with growth_score + sentiment),
             feedback_summary, lessons_learned, next_topic_suggestions
    Returns: updated VideoState (next_topic_suggestions seeds new runs)
    """
    title  = state.get("title", state.get("topic", ""))
    print(f"\n[A6] Analysing feedback for: '{title}'")

    # ── Extract feedback metrics ───────────────────────────────────────────────
    feedback: VideoFeedback = state.get("feedback", VideoFeedback(
        views=0, watch_time_percent=0.0, ctr=0.0,
        top_comments=[], sentiment="mixed", growth_score=0.0,
    ))

    # ── Compute growth score (Python, not LLM) ────────────────────────────────
    growth_score = compute_growth_score(feedback)
    feedback["growth_score"] = growth_score
    print(f"[A6] Growth score: {growth_score:.2f}  "
          f"(views={feedback.get('views')}, "
          f"CTR={feedback.get('ctr', 0)*100:.1f}%, "
          f"retention={feedback.get('watch_time_percent')}%)")

    # ── Build prompt inputs ────────────────────────────────────────────────────
    top_comments  = feedback.get("top_comments", [])
    metrics_ctx   = _build_metrics_context(feedback)
    comments_fmt  = _format_comments(top_comments)

    # ── Invoke LLM chain ──────────────────────────────────────────────────────
    llm = ChatAnthropic(
        model=LLM_MODEL,
        temperature=0.6,
        api_key=ANTHROPIC_API_KEY,
    )
    chain  = FEEDBACK_PROMPT | llm | JsonOutputParser()
    raw: dict = chain.invoke({
        "video_metrics"   : metrics_ctx,
        "title"           : title,
        "topic"           : state.get("refined_topic", state.get("topic", "")),
        "domain"          : state.get("domain", "AI"),
        "key_facts"       : json.dumps(state.get("key_facts", [])),
        "hook"            : state.get("hook", ""),
        "narration_style" : state.get("narration_style", ""),
        "duration"        : state.get("estimated_duration_min", 7),
        "publish_date"    : state.get("publish_date", ""),
        "top_comments"    : comments_fmt,
        "comment_count"   : len(top_comments),
        "growth_score"    : growth_score,
    })

    # ── Validate required fields ───────────────────────────────────────────────
    required = ["performance_summary", "what_worked", "what_failed",
                "next_video_suggestions", "optimization_tips", "lessons_learned"]
    missing  = [k for k in required if k not in raw]
    if missing:
        raise ValueError(f"[A6] LLM response missing fields: {missing}")

    # ── Enrich feedback with LLM sentiment ────────────────────────────────────
    llm_sentiment = raw.get("sentiment", feedback.get("sentiment", "mixed"))
    feedback["sentiment"] = llm_sentiment   # type: ignore[literal-required]

    # ── Log results ───────────────────────────────────────────────────────────
    print(f"[A6] Sentiment    : {llm_sentiment}")
    print(f"[A6] What worked  : {len(raw.get('what_worked', []))} items")
    print(f"[A6] What failed  : {len(raw.get('what_failed', []))} items")
    print(f"[A6] Next topics  : {len(raw.get('next_video_suggestions', []))} suggestions")
    for suggestion in raw.get("next_video_suggestions", []):
        print(f"       → {suggestion}")

    # ── Write into VideoState ──────────────────────────────────────────────────
    updates: VideoState = {
        "feedback"               : feedback,
        "feedback_summary"       : raw["performance_summary"],
        "lessons_learned"        : raw["lessons_learned"],
        "next_topic_suggestions" : raw["next_video_suggestions"],
        "status"                 : "posted",            # type: ignore[typeddict-item]
    }
    # Store full analysis as extra keys
    updates["what_worked"]       = raw.get("what_worked", [])        # type: ignore
    updates["what_failed"]       = raw.get("what_failed", [])        # type: ignore
    updates["optimization_tips"] = raw.get("optimization_tips", [])  # type: ignore

    return {**state, **updates}


# ── Convenience: inject feedback into state from raw metrics dict ──────────────

def inject_feedback(
    state: VideoState,
    views: int,
    ctr: float,
    watch_time_percent: float,
    top_comments: list[str],
    sentiment: str = "mixed",
) -> VideoState:
    """
    Helper to populate state["feedback"] from raw numbers before running A6.
    Call this before invoking agent6_feedback() in the graph or CLI.

    Example:
        state = inject_feedback(state, views=850, ctr=0.042,
                                watch_time_percent=58.0,
                                top_comments=["Great video!", "More please"])
        state = agent6_feedback(state)
    """
    state["feedback"] = VideoFeedback(
        views              = views,
        watch_time_percent = watch_time_percent,
        ctr                = ctr,
        top_comments       = top_comments,
        sentiment          = sentiment,         # type: ignore[typeddict-item]
        growth_score       = 0.0,               # computed in agent6_feedback
    )
    return state


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
    print(f"FULL PIPELINE TEST: A1 → A2 → A3 → A4 → A5 → A6")
    print(f"Topic : {TEST_TOPIC}")
    print('='*60)

    s = initial_state(TEST_TOPIC, TEST_DOMAIN)
    s = agent1_research(s)
    s = agent2_script(s)
    s = agent3_visual(s)
    s = agent4_qa(s)
    s["qa_status"] = "pass"           # force pass for test
    s = agent5_publish(s)

    # Inject realistic performance data
    s = inject_feedback(
        s,
        views              = 843,
        ctr                = 0.038,   # 3.8%
        watch_time_percent = 61.0,
        top_comments       = [
            "Best RAG explanation I've seen — the library analogy was perfect!",
            "Can you do a video on vector databases next?",
            "I finally understand how ChatGPT searches the web. Thank you!",
            "Scene 4 was a bit fast, could you slow down?",
            "More AI videos please!",
        ],
        sentiment = "positive",
    )

    s = agent6_feedback(s)

    print(f"\n{'─'*60}")
    print(f"STATUS           : {s.get('status')}")
    print(f"GROWTH SCORE     : {s['feedback']['growth_score']:.2f}")
    print(f"PERFORMANCE      :\n  {s.get('feedback_summary')}")
    print(f"\nWHAT WORKED:")
    for w in s.get("what_worked", []):           # type: ignore
        print(f"  + {w}")
    print(f"\nWHAT FAILED:")
    for f in s.get("what_failed", []):           # type: ignore
        print(f"  - {f}")
    print(f"\nNEXT TOPIC SUGGESTIONS:")
    for t in s.get("next_topic_suggestions", []):
        print(f"  → {t}")
    print(f"\nOPTIMIZATION TIPS:")
    for tip in s.get("optimization_tips", []):   # type: ignore
        print(f"  * {tip}")
    print(f"\nLESSONS LEARNED:")
    for lesson in s.get("lessons_learned", []):
        print(f"  ! {lesson}")
