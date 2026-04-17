# LearningAgentsFlow 🎓⚡

**An 8-agent AI pipeline for creating, publishing, and growing educational YouTube content — built with LangChain, LangGraph, and Claude.**

Covers AI, Data Governance, and SAP MDG topics. Produces research, scripts, visuals, QA-reviewed content, YouTube metadata, InVideo AI prompts (consumer + enterprise), and growth analytics — all from a single topic input.

---

## Pipeline Overview

```
Topic Input
    │
    ▼
A1 Research Agent          — simplifies topic for a 15-year-old
    │
    ▼
A2 Script Agent            — scene-by-scene script with hook + CTA
    │                           ↑ retries (max 2) with revision notes
    ▼
A3 Visual Planner          — image/video prompts, voiceover, B-roll
    │
    ▼
A4 QA Agent (Gatekeeper)   — accuracy · tone · SEO · length scoring
    │  pass ─────────────────────────────────────────────┐
    │  fail → back to A2                                 │
                                                         ▼
                                             A5 Publishing Agent
                                             (Notion + JSON + SEO metadata)
                                                         │
                              ┌──────────────────────────┤
                              ▼                          ▼
                   A7 InVideo Agent         A8 Enterprise InVideo Agent
                   (Consumer style)         (Microsoft/enterprise style)
                                                         │
                                                         ▼
                                             A6 Feedback Agent
                                             (Growth Engine → seeds next run)
```

---

## Agents

| # | Agent | Role | Temp | Key Output |
|---|-------|------|------|------------|
| A1 | Research Agent | Educator + Researcher | 0.7 | `simple_explanation`, `analogy`, `key_facts` |
| A2 | Script Agent | Scriptwriter + Scene Builder | 0.7 | `full_script`, `scene_list`, `hook` |
| A3 | Visual Planner | Visual Generation Planner | 0.7 | `visual_briefs[]`, `thumbnail_prompt` |
| A4 | QA Agent | Quality Gatekeeper | 0.3 | `qa_scores{accuracy,tone,seo,length}`, `qa_status` |
| A5 | Publishing Agent | SEO + Publisher | 0.5 | `yt_title`, `yt_tags`, `notion_page_id`, `publish_date` |
| A6 | Feedback Agent | Growth Engine | 0.6 | `growth_score`, `next_topic_suggestions` |
| A7 | InVideo Agent | Consumer Video Prompter | 0.4 | `invideo_prompt` (plain string) |
| A8 | Enterprise Video Agent | Microsoft-Style Prompter | 0.3 | `invideo_enterprise_prompt` (plain string) |

---

## Project Structure

```
learningagentsflow/
├── agents/
│   ├── agent1_research.py
│   ├── agent2_script.py
│   ├── agent3_visual.py
│   ├── agent4_qa.py
│   ├── agent5_publish.py
│   ├── agent6_feedback.py
│   ├── agent7_invideo.py
│   └── agent8_invideo_enterprise.py
├── ui/
│   └── index.html                  # LearningAgentsFlow visual dashboard
├── output/
│   ├── scripts/                    # Generated script JSON files
│   └── content_calendar.json       # Auto-managed publishing calendar
├── state.py                        # VideoState TypedDict (shared state)
├── config.py                       # Environment + settings
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/learningagentsflow.git
cd learningagentsflow
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your keys
```

### 3. Notion database setup (optional)

Create a Notion database with these columns:

| Column | Type |
|--------|------|
| Name | Title |
| Domain | Select |
| Status | Select |
| Publish Date | Date |
| Duration (min) | Number |

---

## Usage

### Run individual agents (each runs the full pipeline to that point)

```bash
python agents/agent1_research.py     # A1 only
python agents/agent2_script.py       # A1 → A2
python agents/agent3_visual.py       # A1 → A2 → A3
python agents/agent4_qa.py           # A1 → A2 → A3 → A4
python agents/agent5_publish.py      # A1 → ... → A5
python agents/agent6_feedback.py     # Full pipeline
python agents/agent7_invideo.py      # Full + consumer InVideo prompt
python agents/agent8_invideo_enterprise.py  # Full + enterprise prompt
```

### UI Dashboard

Open `ui/index.html` in any browser. No server required.

Click **▶ Run Pipeline** to simulate the full 8-agent flow with live logs, progress bars, and QA score meters.

---

## Domains

- **AI** — neural networks, transformers, RAG, agents, LLMs
- **Data Governance** — data lineage, metadata management, data quality
- **SAP MDG** — master data governance, MDG workflows, SAP Fiori

---

## QA Scoring (Agent 4)

| Dimension | Method | Checks |
|-----------|--------|--------|
| Accuracy | LLM | Factual correctness, no hallucinations |
| Tone | LLM | 15-yr-old suitability, jargon-free |
| SEO | LLM | Title/hook keyword strength |
| Length | Python | 700–1400 words = 5–10 min at 140 wpm |

Pass threshold: **0.75** on all four. On fail: routes back to A2 with `revision_notes`. Max **2 retries**.

---

## Growth Score Formula (Agent 6)

```python
growth_score = (
    0.35 * min(views / 1000, 1.0)
  + 0.30 * min(ctr / 0.05, 1.0)
  + 0.25 * min(watch_time_percent / 100, 1.0)
  + 0.10 * sentiment_score   # positive=1.0, mixed=0.5, negative=0.0
)
```

---

## Enterprise Visual Themes (Agent 8)

| Domain | Color | Visual Style |
|--------|-------|--------------|
| AI | `#0078D4` | Transformer diagrams, neural net flows |
| Data Governance | `#003087` + `#008080` | Data lineage, metadata hierarchies |
| SAP MDG | `#0070F2` | Fiori tiles, MDG approval workflows |

---

## Stack

- [LangChain](https://python.langchain.com/) + LangGraph
- [Anthropic Claude](https://anthropic.com) (`claude-sonnet-4-20250514`)
- [Notion API](https://developers.notion.com/)
- [InVideo AI](https://invideo.io)
- Vanilla HTML/CSS/JS (UI dashboard)

---

## License

MIT
