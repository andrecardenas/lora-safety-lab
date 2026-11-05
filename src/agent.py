"""A small tool-using agent that reviews evaluation results.

A language model picks one allowed tool per step until it finishes or hits the
step limit. Tools are read-only, and invalid model output is reported back to it.
"""
import argparse
import json
import re
from pathlib import Path

INSTRUCTIONS = """You review the results of a model evaluation. Reply with one JSON object and nothing else.
Available tools:
{"tool": "get_summary"} returns the pass rates.
{"tool": "list_flagged", "model": "base" or "fine_tuned"} returns the flagged replies.
{"tool": "finish", "report": "two sentences on which model did better and what is still wrong"} ends the review.
Call get_summary first, then list_flagged for fine_tuned, then finish."""


class Tools:
    def __init__(self, summary, replies_markdown):
        self.summary = summary
        self.sections = self._split(replies_markdown)

    @staticmethod
    def _split(markdown):
        parts = {"base": [], "fine_tuned": []}
        current = None
        for line in markdown.splitlines():
            if line.startswith("## Base"):
                current = "base"
            elif line.startswith("## Fine-tuned"):
                current = "fine_tuned"
            elif line.startswith("## "):
                current = None
            elif current and "FLAGGED" in line:
                parts[current].append(line)
        return parts

    def get_summary(self):
        return {"base": self.summary["base"], "fine_tuned": self.summary["fine_tuned"]}

    def list_flagged(self, model):
        if model not in self.sections:
            raise ValueError("model must be base or fine_tuned")
        items = self.sections[model]
        return {"count": len(items), "first_three": [item[:160] for item in items[:3]]}


def parse_action(text):
    match = re.search(r"\{.*\}", str(text), flags=re.S)
    if not match:
        raise ValueError("no JSON object found")
    action = json.loads(match.group(0))
    if not isinstance(action, dict) or "tool" not in action:
        raise ValueError("JSON object needs a tool")
    return action


def run_agent(complete, tools, max_steps=6):
    """complete(messages) returns the model's text. Returns the report and every step."""
    messages = [{"role": "system", "content": INSTRUCTIONS}, {"role": "user", "content": "Begin the review."}]
    steps = []
    for number in range(1, max_steps + 1):
        text = complete(messages)
        messages.append({"role": "assistant", "content": text})
        try:
            action = parse_action(text)
            tool = action["tool"]
            if tool == "finish":
                report = str(action.get("report", "")).strip()
                if not report:
                    raise ValueError("finish needs a report")
                steps.append({"step": number, "tool": tool})
                return {"finished": True, "report": report, "steps": steps}
            if tool == "get_summary":
                result = tools.get_summary()
            elif tool == "list_flagged":
                result = tools.list_flagged(action.get("model"))
            else:
                raise ValueError(f"unknown tool {tool}")
            steps.append({"step": number, "tool": tool})
            next_tool = "list_flagged with model fine_tuned" if tool == "get_summary" else "finish with your report"
            messages.append({"role": "user", "content": f"Tool result: {json.dumps(result)}\nNow reply with only the JSON call for {next_tool}."})
        except (ValueError, json.JSONDecodeError) as error:
            steps.append({"step": number, "error": str(error)})
            messages.append({"role": "user", "content": f"Error: {error}. Reply with one valid JSON tool call."})
    return {"finished": False, "report": "", "steps": steps}


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--results", default="results")
    parser.add_argument("--max-new-tokens", type=int, default=160)
    args = parser.parse_args()

    results = Path(args.results)
    tools = Tools(
        json.loads((results / "eval_summary.json").read_text(encoding="utf8")),
        (results / "replies.md").read_text(encoding="utf8"),
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device).eval()
    set_seed(1)

    def complete(messages):
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=3000).to(device)
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        return tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

    outcome = run_agent(complete, tools)
    (results / "agent_review.json").write_text(json.dumps(outcome, indent=2) + "\n", encoding="utf8")
    print(json.dumps(outcome, indent=2))


if __name__ == "__main__":
    main()
