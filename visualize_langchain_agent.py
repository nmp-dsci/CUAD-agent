#!/usr/bin/env python3
"""Generate a visual diagram of the LangChain CUAD agent architecture.

Produces dashboards/langchain_agent_diagram.html with:
  - The actual LangChain chain graph (via get_graph().draw_mermaid())
  - A full-pipeline architecture diagram
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda

from cuad_agent.evaluators.langchain_runner import build_chain_for_agent

CHAIN_MERMAID_LABEL = "LangChain Chain (per category agent)"
PIPELINE_MERMAID = """\
---
config:
  flowchart:
    curve: linear
---
flowchart TD
    subgraph DATA["Data Layer"]
        DS[(CUAD Dataset)]
        SEL["select_evaluation_set()\\n(sample_size, seed, eval_split)"]
        BD["build_devset()\\n→ list[dict] examples"]
        DS --> SEL --> BD
    end

    subgraph PROMPTS["Prompt Layer"]
        SP["compose_system_prompt()\\n(question, category, description, format)"]
        PO["load_prompt_overrides()\\n(CATEGORY_SYSTEM_PROMPTS)"]
        PO -- override --> SP
    end

    subgraph AGENTS["Agent Layer (41 × ContractQuestionAgent)"]
        BA["build_agents()\\n→ dict[question_index, agent]"]
        direction LR
        A1["Agent: Governing Law"]
        A2["Agent: Termination for Conv."]
        AN["Agent: ... (×41 total)"]
        BA --> A1 & A2 & AN
    end

    subgraph CHAIN["LangChain Chain (per agent)"]
        direction LR
        RL["RunnableLambda\\nmake_messages()"]
        LLM["BaseChatModel\\n(ChatDeepSeek / ChatOpenAI)"]
        PAR["PydanticOutputParser\\n→ CuadAnswer"]
        RL --> LLM --> PAR
    end

    subgraph EVAL["Evaluation"]
        EB["chain.batch()\\n(max_concurrency=4)"]
        F1["token_overlap_f1()"]
        RR["result_record()\\n→ dict"]
        EB --> F1 --> RR
    end

    subgraph OUTPUT["Output Layer"]
        CSV["results.csv"]
        JSONL["results.jsonl\\n(incremental cache)"]
        SUM["summary.json\\n(mean F1, correct@0.5)"]
        HTML["evaluation_MODEL_ID.html\\n(dashboard)"]
        RR --> CSV & JSONL & SUM & HTML
    end

    BD --> BA
    SP --> BA
    BD --> EVAL
    A1 & A2 & AN --> CHAIN
    CHAIN --> EVAL
"""


def chain_mermaid() -> str:
    llm = FakeListChatModel(responses=["{}"])
    chain = build_chain_for_agent(llm, "You are a legal assistant.")

    def make_messages(inputs: dict) -> list:
        return [SystemMessage(content="sys"), HumanMessage(content="user")]

    chain = RunnableLambda(make_messages) | llm
    return chain.get_graph().draw_mermaid()


def render_html(chain_diagram: str, pipeline_diagram: str, output_path: Path) -> None:
    html = f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>CUAD LangChain Agent — Architecture</title>
<script src="https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.min.js"></script>
<style>
  body {{
    font-family: system-ui, sans-serif;
    max-width: 1100px;
    margin: 2rem auto;
    padding: 0 1.5rem;
    background: #fafafa;
    color: #222;
  }}
  h1 {{ font-size: 1.6rem; margin-bottom: 0.25rem; }}
  h2 {{ font-size: 1.1rem; color: #555; margin-top: 2rem; }}
  .mermaid {{
    background: #fff;
    border: 1px solid #e0e0e0;
    border-radius: 6px;
    padding: 1rem;
    margin-top: 0.5rem;
  }}
  p.caption {{
    font-size: 0.85rem;
    color: #888;
    margin-top: 0.3rem;
  }}
</style>
</head>
<body>
<h1>CUAD LangChain Agent — Architecture Diagrams</h1>
<p>Generated from <code>langchain_runner.py</code> using LangChain's built-in graph API.</p>

<h2>1. LangChain Chain (per category agent)</h2>
<div class="mermaid">
{chain_diagram}
</div>
<p class="caption">
  The inner <code>Runnable</code> for each of the 41 CUAD category agents:<br>
  <code>RunnableLambda(make_messages) | BaseChatModel</code>.
  Input fields are assembled into a <code>[SystemMessage, HumanMessage]</code> pair
  and sent to the LLM. The raw response is parsed client-side by
  <code>PydanticOutputParser → CuadAnswer</code>.
</p>

<h2>2. Full Evaluation Pipeline</h2>
<div class="mermaid">
{pipeline_diagram}
</div>
<p class="caption">
  End-to-end flow from the CUAD dataset through 41 parallel agents to HTML dashboard output.
  RAG context modes (raw / rag-dense / rag-hybrid / rag-hierarchical-*) are applied
  in the Data Layer before examples reach the agents.
</p>

<script>
  mermaid.initialize({{ startOnLoad: true, theme: 'neutral' }});
</script>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    print(f"Saved: {output_path}")


def main() -> None:
    output_path = Path("dashboards/langchain_agent_diagram.html")
    print("Building chain graph…")
    chain_diagram = chain_mermaid()
    print("Rendering HTML…")
    render_html(chain_diagram, PIPELINE_MERMAID, output_path)


if __name__ == "__main__":
    main()
