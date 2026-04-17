"""
graph.py — LangGraph orchestrator for the 8-agent YouTube content pipeline.

Flow:
  A1 Research → A2 Script → A3 Visual → A4 QA
                 ↑ retry ──────────────┘
  QA pass → A5 Publish → (A6 Feedback in parallel)
                        → (A7 InVideo in parallel)
                        → (A8 Enterprise InVideo in parallel)
"""

from langgraph.graph import StateGraph, END
from state import (
    VideoState, initial_state,
    route_after_qa, route_after_research,
)
from agents.agent1_research          import agent1_research
from agents.agent2_script            import agent2_script
from agents.agent3_visual            import agent3_visual
from agents.agent4_qa                import agent4_qa
from agents.agent5_publish           import agent5_publish
from agents.agent6_feedback          import agent6_feedback
from agents.agent7_invideo           import agent7_invideo
from agents.agent8_invideo_enterprise import agent8_invideo_enterprise


def build_graph() -> StateGraph:
    g = StateGraph(VideoState)

    # Register nodes
    g.add_node("agent1_research",           agent1_research)
    g.add_node("agent2_script",             agent2_script)
    g.add_node("agent3_visual",             agent3_visual)
    g.add_node("agent4_qa",                 agent4_qa)
    g.add_node("agent5_publish",            agent5_publish)
    g.add_node("agent6_feedback",           agent6_feedback)
    g.add_node("agent7_invideo",            agent7_invideo)
    g.add_node("agent8_invideo_enterprise", agent8_invideo_enterprise)

    # Entry point
    g.set_entry_point("agent1_research")

    # Sequential core: A1 → A2 → A3 → A4
    g.add_conditional_edges("agent1_research", route_after_research,
        {"agent2_script": "agent2_script", "agent1_research": "agent1_research"})
    g.add_edge("agent2_script", "agent3_visual")
    g.add_edge("agent3_visual", "agent4_qa")

    # QA gate with retry loop
    g.add_conditional_edges("agent4_qa", route_after_qa, {
        "agent5_publish" : "agent5_publish",
        "agent2_script"  : "agent2_script",
        END              : END,
    })

    # Post-publish: all three run independently
    g.add_edge("agent5_publish", "agent6_feedback")
    g.add_edge("agent5_publish", "agent7_invideo")
    g.add_edge("agent5_publish", "agent8_invideo_enterprise")

    g.add_edge("agent6_feedback",           END)
    g.add_edge("agent7_invideo",            END)
    g.add_edge("agent8_invideo_enterprise", END)

    return g


def run_pipeline(topic: str, domain: str) -> VideoState:
    """Run the full pipeline and return the final state."""
    graph = build_graph().compile()
    state = initial_state(topic, domain)
    return graph.invoke(state)


if __name__ == "__main__":
    import sys
    topic  = sys.argv[1] if len(sys.argv) > 1 else "What is RAG in AI?"
    domain = sys.argv[2] if len(sys.argv) > 2 else "AI"
    result = run_pipeline(topic, domain)
    print(f"\nFinal status : {result.get('status')}")
    print(f"Title        : {result.get('title')}")
    print(f"Publish date : {result.get('publish_date')}")
    print(f"QA scores    : {result.get('qa_scores')}")
