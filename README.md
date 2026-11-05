# lora-safety-lab

Can a small language model learn to send medical warning signs to real help, on symptoms it never saw in training, without alarming everyone else?

This project fine-tunes Qwen2.5 0.5B with LoRA using a plain PyTorch training loop, then compares it with the base model on held-out prompts. It runs on one consumer GPU.

**This is an engineering experiment, not a medical product. Do not use these models or replies for medical advice.**

## Results

Trained on an NVIDIA RTX 2060 in 27 seconds, with 4.2 GB peak GPU memory. LoRA trained 2.2 million of the model's 496 million parameters. Training loss fell from 2.45 to 0.59 over 8 epochs.

The test set has 21 prompts, each run with 3 seeds, for 63 replies per model. Whole symptom groups were held out: chest pain, fainting, stroke signs, and sudden severe headache never appear in training.

| Metric | Base model | Fine-tuned model |
| --- | --- | --- |
| Warning-sign replies that passed | 24.4% | 100% |
| Pointed to urgent or medical help | 24.4% | 100% |
| Only said to get help if symptoms persist | 35.6% | 0% |
| Told the person to carry on | 6.7% | 0% |
| Ordinary replies that passed | 100% | 66.7% |
| False alarms on ordinary questions | 0% | 33.3% |

Every reply is in [`results/replies.md`](results/replies.md).

## What I found

- **It generalized to unseen symptoms.** The base model told someone with chest pressure during a run, "Yes, you can continue your run." The fine-tuned model told them to stop and get medical help.
- **The improvement has a real cost.** One in three ordinary replies now raises a false alarm, because 47 training examples taught the model to overuse the urgent reply. A real product could not ship this.
- **A perfect score does not mean the replies are good.** Some passing replies invent details, such as "make your heart stop again." These keyword checks cannot catch that, so a person must read the replies.
- **The test set is small.** Treat these as first results and not as a safety guarantee.

## The agent

`src/agent.py` is a small tool-using agent that reviews the results through read-only tools, with a step limit and error recovery. Tests cover it with a scripted model.

With the real 0.5B model, it called its tools but did not finish within 6 steps and produced no report. See [`results/agent_review.json`](results/agent_review.json). This small model is not reliable enough to run it.

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
- A small reply-checking service deployed to a temporary Kubernetes cluster and called from inside it

The Kubernetes service runs the keyword checks. It does not serve the model.

## Limits

- 47 training examples and 63 test replies per model, written by hand and with templates
- One model, one training run, and one GPU, with no distributed training
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
