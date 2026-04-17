"""
main.py — CLI entrypoint for the YouTube Content Agent System.

Usage:
  python main.py run   --topic "What is RAG?" --domain AI
  python main.py ui                             # open AgentFlow UI
  python main.py feedback --title "..." --views 843 --ctr 0.038 --retention 61
"""

import argparse, sys, webbrowser
from pathlib import Path


def cmd_run(args):
    from graph import run_pipeline
    result = run_pipeline(args.topic, args.domain)
    print(f"\n✅  Pipeline complete")
    print(f"   Title       : {result.get('title')}")
    print(f"   Publish     : {result.get('publish_date')}")
    print(f"   QA scores   : {result.get('qa_scores')}")
    print(f"   Local JSON  : {result.get('local_json_path')}")


def cmd_ui(args):
    ui_path = Path(__file__).parent / "ui" / "index.html"
    webbrowser.open(f"file://{ui_path.resolve()}")
    print(f"🌐  AgentFlow UI opened: {ui_path}")


def cmd_feedback(args):
    from state import initial_state
    from agents.agent6_feedback import agent6_feedback, inject_feedback
    import json
    from pathlib import Path
    from config import OUTPUT_DIR

    # Find latest local JSON for this title
    scripts_dir = Path(OUTPUT_DIR) / "scripts"
    matches = list(scripts_dir.glob(f"*{args.title[:20].lower().replace(' ','_')}*.json"))
    if not matches:
        print("❌  No matching script found. Run the pipeline first.")
        sys.exit(1)

    data = json.loads(matches[-1].read_text())
    s    = initial_state(data["topic"], data["domain"])
    s.update(data)
    s    = inject_feedback(s, args.views, args.ctr, args.retention,
                           args.comments or [], args.sentiment)
    s    = agent6_feedback(s)
    print(f"\n📈  Growth score  : {s['feedback']['growth_score']:.2f}")
    print(f"📋  Next topics  :")
    for t in s.get("next_topic_suggestions", []):
        print(f"     → {t}")


def build_parser():
    p = argparse.ArgumentParser(prog="yt-agents")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run",      help="Run full pipeline")
    r.add_argument("--topic",  required=True)
    r.add_argument("--domain", default="AI",
                   choices=["AI","Data Governance","SAP MDG"])
    r.set_defaults(func=cmd_run)

    u = sub.add_parser("ui",       help="Open AgentFlow UI")
    u.set_defaults(func=cmd_ui)

    f = sub.add_parser("feedback", help="Inject performance data")
    f.add_argument("--title",     required=True)
    f.add_argument("--views",     type=int,   default=0)
    f.add_argument("--ctr",       type=float, default=0.0)
    f.add_argument("--retention", type=float, default=0.0)
    f.add_argument("--sentiment", default="mixed",
                   choices=["positive","mixed","negative"])
    f.add_argument("--comments",  nargs="*", default=[])
    f.set_defaults(func=cmd_feedback)

    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    args.func(args)
