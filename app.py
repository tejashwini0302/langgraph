import os
import sys
import io
import uuid
import traceback
from typing import TypedDict, List, Optional

from flask import Flask, request, jsonify, render_template

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI

# ==========================================
# 1. LLM INITIALIZATION
# ==========================================
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    print("WARNING: GEMINI_API_KEY environment variable not set.")

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite-preview",
    google_api_key=GEMINI_API_KEY,
)

# ==========================================
# 2. STATE DEFINITION
# ==========================================
class CrewState(TypedDict):
    messages: List[BaseMessage]
    next_step: Optional[str]
    code: Optional[str]
    report: Optional[str]


# ==========================================
# 3. TOOLS
# ==========================================
@tool
def run_python_code(code: str) -> str:
    """Execute python code and return the standard output or error trace."""
    if not isinstance(code, str):
        code = str(code)
    clean_code = code.replace("```python", "").replace("```", "").strip()

    old_stdout = sys.stdout
    new_stdout = io.StringIO()
    sys.stdout = new_stdout

    try:
        local_scope = {}
        exec(clean_code, {}, local_scope)
        result = new_stdout.getvalue()
    except Exception:
        result = f"Execution Error:\n{traceback.format_exc()}"
    finally:
        sys.stdout = old_stdout

    return result.strip() if result.strip() else "Success (no terminal output)"


@tool
def generate_test_cases(task_description: str) -> str:
    """Generate specific test scenarios for a given coding task."""
    prompt = (
        f"You are a Senior QA Engineer. Generate 3 to 5 highly specific test scenarios "
        f"for the following coding task: '{task_description}'.\n"
        f"Include standard cases and edge cases. Return them as a numbered list."
    )
    response = llm.invoke(prompt)
    return response.content if hasattr(response, "content") else str(response)


def _extract_text(content):
    """Safely parse Gemini's content format (plain string or list-of-dict chunks)."""
    if isinstance(content, list):
        if content and isinstance(content[0], dict):
            return content[0].get("text", "")
        return str(content[0]) if content else ""
    return str(content)


# ==========================================
# 4. GRAPH NODES
# ==========================================
def real_time_developer(state: CrewState):
    task = state["messages"][-1].content
    dev_prompt = (
        f"Write a clean Python script to solve this: {task}. "
        f"Only return the code, no explanation or markdown formatting."
    )
    response = llm.invoke(dev_prompt)
    code_str = _extract_text(response.content)
    return {"code": code_str}


def real_time_tester(state: CrewState):
    task = state["messages"][-1].content

    test_cases = generate_test_cases.invoke(task)
    cases_str = _extract_text(test_cases)

    execution_result = run_python_code.invoke({"code": state["code"]})

    report = (
        f"### EXECUTION OUTPUT:\n{execution_result}\n\n"
        f"### TEST SCENARIOS EVALUATED:\n{cases_str}"
    )
    return {"report": report}


# ==========================================
# 5. GRAPH CONSTRUCTION
# ==========================================
# NOTE: the original notebook's task_input / manager_decision nodes used
# input() in a terminal loop. That doesn't work in a deployed web server,
# so those two "human in the loop" steps are now handled as separate
# HTTP endpoints below (/api/run_task and /api/archive) instead of graph nodes.
rt_workflow = StateGraph(CrewState)
rt_workflow.add_node("developer", real_time_developer)
rt_workflow.add_node("tester", real_time_tester)
rt_workflow.add_edge(START, "developer")
rt_workflow.add_edge("developer", "tester")
rt_workflow.add_edge("tester", END)

rt_app = rt_workflow.compile()

# ==========================================
# 6. FLASK APP
# ==========================================
app = Flask(__name__)

# In-memory archive (per-process; resets when the Render service restarts/redeploys).
# Swap this for a real DB later if you need persistence across deploys.
ARCHIVE = []


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/api/run_task", methods=["POST"])
def run_task():
    """Equivalent of task_input_node -> developer -> tester -> manager report."""
    data = request.get_json(force=True) or {}
    task = (data.get("task") or "").strip()
    if not task:
        return jsonify({"error": "task is required"}), 400

    try:
        result = rt_app.invoke(
            {"messages": [HumanMessage(content=task)]},
            config={"recursion_limit": 50},
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify(
        {
            "task": task,
            "code": result.get("code"),
            "report": result.get("report"),
        }
    )


@app.route("/api/archive", methods=["POST"])
def archive_task():
    """Equivalent of manager_decision_node choosing 'store' -> archiver_node."""
    data = request.get_json(force=True) or {}
    entry = {
        "id": str(uuid.uuid4()),
        "task": data.get("task"),
        "code": data.get("code"),
        "report": data.get("report"),
    }
    ARCHIVE.append(entry)
    return jsonify({"status": "stored", "id": entry["id"]})


@app.route("/api/archive", methods=["GET"])
def list_archive():
    return jsonify({"items": ARCHIVE})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
