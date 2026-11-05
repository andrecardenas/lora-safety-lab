import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agent import Tools, parse_action, run_agent

SUMMARY = {"base": {"warning_pass_rate": 0.1}, "fine_tuned": {"warning_pass_rate": 0.8}}
REPLIES = "## Base model\n\n- **x, FLAGGED.** one\n- **y, pass.** two\n\n## Fine-tuned model\n\n- **z, FLAGGED.** three\n"


def scripted(*replies):
    queue = list(replies)
    return lambda messages: queue.pop(0)


def test_agent_uses_tools_then_finishes():
    outcome = run_agent(
        scripted(
            '{"tool": "get_summary"}',
            '{"tool": "list_flagged", "model": "fine_tuned"}',
            '{"tool": "finish", "report": "Fine-tuned did better."}',
        ),
        Tools(SUMMARY, REPLIES),
    )
    assert outcome["finished"]
    assert outcome["report"] == "Fine-tuned did better."
    assert [step["tool"] for step in outcome["steps"]] == ["get_summary", "list_flagged", "finish"]


def test_agent_recovers_from_bad_output():
    outcome = run_agent(
        scripted("not json", '{"tool": "delete_everything"}', '{"tool": "finish", "report": "Done."}'),
        Tools(SUMMARY, REPLIES),
    )
    assert outcome["finished"]
    assert "error" in outcome["steps"][0]
    assert "unknown tool" in outcome["steps"][1]["error"]


def test_agent_stops_at_the_step_limit():
    outcome = run_agent(lambda messages: '{"tool": "get_summary"}', Tools(SUMMARY, REPLIES), max_steps=3)
    assert not outcome["finished"]
    assert len(outcome["steps"]) == 3


def test_flagged_replies_are_separated_by_model():
    tools = Tools(SUMMARY, REPLIES)
    assert tools.list_flagged("base")["count"] == 1
    assert tools.list_flagged("fine_tuned")["count"] == 1


def test_parse_action_reads_json_inside_text():
    assert parse_action('Sure: {"tool": "get_summary"}')["tool"] == "get_summary"
