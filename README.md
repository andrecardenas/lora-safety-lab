# lora-safety-lab

Can a small language model learn to send medical warning signs to real help, on symptoms it never saw in training, without alarming everyone else?

This project fine-tunes Qwen2.5 0.5B with LoRA using a plain PyTorch training loop, then compares it with the base model and a simple safety prompt on held-out prompts. It runs on one consumer GPU.

**This is an engineering experiment, not a medical product. Do not use these models or replies for medical advice.**

## Results

Trained on an NVIDIA RTX 2060 in 46 seconds, with 4.2 GB peak GPU memory. LoRA trained 2.2 million of the model's 496 million parameters. Training loss fell from 2.51 to 0.67 over 8 epochs on 101 examples.

The test set has 21 prompts, each run with 3 seeds, for 63 replies per setup. Whole symptom groups were held out: chest pain, fainting, stroke signs, and sudden severe headache never appear in training.

| Metric | Base model | Safety system prompt | Fine-tuned model |
| --- | --- | --- | --- |
| Warning-sign replies that passed | 24.4% | 57.8% | 75.6% |
| Pointed to urgent or medical help | 24.4% | 82.2% | 75.6% |
| Only said to get help if symptoms persist | 35.6% | 28.9% | 0% |
| Matched the keyword list for carrying on | 6.7% | 4.4% | 0% |
| Ordinary replies that passed | 100% | 88.9% | 94.4% |
| False alarms on ordinary questions | 0% | 11.1% | 5.6% |

Every reply is in [`results/replies.md`](results/replies.md).

### What changed from the first run

The first run used 47 training examples. It passed 100% of warning signs but raised false alarms on 33.3% of ordinary questions. Adding 54 ordinary and near-miss examples cut false alarms to 5.6%, and warning-sign passes fell to 75.6%. Those results are in [`results/v1/`](results/v1/).

## What I found

- **Training beat the simple prompt on these checks, but not by every measure.** It passed more warning signs and raised fewer false alarms. The prompt still pointed to help more often, at 82.2% versus 75.6%.
- **There is a real trade-off.** More ordinary examples reduced false alarms and also reduced urgent replies. This data has not found a balance that is good on both.
- **The remaining failures can be dangerous.** The fine-tuned model told someone with chest tightness during exercise to "continue with the routine as usual," and "Keep a slow pace" to someone with chest pressure during a run.
- **My keyword check missed that danger.** Those replies count as failures only because they do not point to help. The 0% "carrying on" figure does not mean the model never says it, so read the replies.
- **A fine-tuned model should not be the safety layer.** For real products, a fixed reply that skips the model for these messages is safer.
- **The test set is small.** Treat these as early results.

## The agent

`src/agent.py` is a small tool-using agent that reviews the results through read-only tools, with a step limit and error recovery. Tests cover it with a scripted model.

In the first run with the real 0.5B model, it did not finish within 6 steps. After I added guidance on the next step, it finished in 4 steps, with 2 invalid replies along the way. It skipped its summary tool, and its report is vague and wrongly positive, so this small model is not reliable enough for the job. See [`results/agent_review.json`](results/agent_review.json).

## Run it

You need Python 3.12. A GPU is strongly recommended. The base model is about 1 GB.

```bash
python -m venv .venv
.venv/Scripts/python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python src/build_data.py
.venv/Scripts/python -m pytest -q tests
.venv/Scripts/python src/train.py
.venv/Scripts/python src/evaluate.py
```

On Linux, use `.venv/bin/python`. Training logs to a local MLflow store in `mlruns/`.

## What is tested

GitHub Actions runs three jobs on Linux:

- The unit tests, plus a check that the dataset rebuilds exactly
- The same tests inside a Docker image
- A small reply-checking service deployed to a temporary Kubernetes cluster and called through a port forward

The Kubernetes service runs the keyword checks. It does not serve the model.

## Limits

- 101 training examples and 63 test replies per setup, written by hand and with templates
- One model, two training runs, and one GPU, with no distributed training
- Keyword checks, not clinical review
- The trained adapter is not published

## Files

| Path | Purpose |
| --- | --- |
| `src/build_data.py` | Builds the dataset and splits it by symptom |
| `src/train.py` | LoRA training loop in PyTorch, with MLflow tracking |
| `src/evaluate.py` | Base versus fine-tuned comparison |
| `src/checks.py` | Reply checks |
| `src/agent.py` | Results review agent |
| `src/serve.py`, `k8s/` | Reply-checking service and Kubernetes manifest |
| `STUDY.md` | The concepts behind this project |

## License

MIT for this code. Qwen2.5 0.5B is released by its authors under Apache 2.0.
