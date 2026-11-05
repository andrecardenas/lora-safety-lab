"""Fine-tune a small language model with LoRA, using a plain PyTorch training loop."""
import argparse
import json
import math
import random
import time
from pathlib import Path

import mlflow
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup

SYSTEM = "You are a brief wellness assistant. Reply in two to four sentences."


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf8").splitlines() if line.strip()]


def encode(tokenizer, row, max_length):
    """Mask the prompt with -100 so only the assistant's reply counts toward the loss."""
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": row["prompt"]}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    full = prompt + row["reply"] + (tokenizer.eos_token or "")
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full, add_special_tokens=False, truncation=True, max_length=max_length)["input_ids"]
    labels = [-100] * min(len(prompt_ids), len(full_ids)) + full_ids[len(prompt_ids):]
    return {"input_ids": full_ids, "labels": labels}


def collate(batch, pad_id):
    width = max(len(item["input_ids"]) for item in batch)
    input_ids, labels, mask = [], [], []
    for item in batch:
        gap = width - len(item["input_ids"])
        input_ids.append(item["input_ids"] + [pad_id] * gap)
        labels.append(item["labels"] + [-100] * gap)
        mask.append([1] * len(item["input_ids"]) + [0] * gap)
    return {
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(labels),
        "attention_mask": torch.tensor(mask),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--train", default="data/train.jsonl")
    parser.add_argument("--out", default="outputs/adapter")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--target-modules", default="q_proj,k_proj,v_proj,o_proj")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).to(device)
    model = get_peft_model(
        model,
        LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_r,
            lora_dropout=0.05,
            target_modules=args.target_modules.split(","),
            task_type="CAUSAL_LM",
        ),
    )
    trainable, total = model.get_nb_trainable_parameters()
    print(f"device={device} trainable={trainable:,} of {total:,}")

    rows = [encode(tokenizer, row, args.max_length) for row in read_jsonl(args.train)]
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=0.01)
    steps = args.epochs * math.ceil(len(rows) / args.batch_size)
    scheduler = get_cosine_schedule_with_warmup(optimizer, max(1, steps // 20), steps)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    mlflow.set_tracking_uri(f"file:{Path('mlruns').resolve().as_posix()}")
    mlflow.set_experiment("lora-safety-lab")
    started = time.time()
    with mlflow.start_run():
        mlflow.log_params({**vars(args), "device": device, "trainable_params": trainable, "train_rows": len(rows)})
        model.train()
        step = 0
        history = []
        for epoch in range(args.epochs):
            random.shuffle(rows)
            total_loss = 0.0
            batches = 0
            for start in range(0, len(rows), args.batch_size):
                batch = collate(rows[start:start + args.batch_size], tokenizer.pad_token_id)
                batch = {key: value.to(device) for key, value in batch.items()}
                with torch.autocast(device_type=device, enabled=use_amp, dtype=torch.float16):
                    loss = model(**batch).loss
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                total_loss += loss.item()
                batches += 1
                step += 1
            average = total_loss / max(1, batches)
            history.append(round(average, 4))
            mlflow.log_metric("train_loss", average, step=epoch)
            print(f"epoch {epoch + 1}/{args.epochs} loss {average:.4f}")
        elapsed = time.time() - started
        peak = torch.cuda.max_memory_allocated() / 1e9 if use_amp else None
        mlflow.log_metric("train_seconds", elapsed)
        if peak is not None:
            mlflow.log_metric("peak_gpu_gb", peak)

    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    summary = {
        "model": args.model,
        "device": torch.cuda.get_device_name(0) if use_amp else "cpu",
        "train_rows": len(rows),
        "epochs": args.epochs,
        "steps": step,
        "trainable_params": trainable,
        "total_params": total,
        "train_seconds": round(elapsed, 1),
        "peak_gpu_gb": round(peak, 2) if peak is not None else None,
        "loss_by_epoch": history,
    }
    Path(args.out, "train_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf8")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
