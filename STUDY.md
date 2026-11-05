# Study guide

Questions to be able to answer without notes.

## Training

1. What does LoRA do, and why are only 2.2 million of 496 million parameters trained?
2. What do `r` and `lora_alpha` control?
3. Why are prompt tokens labeled -100 in `src/train.py`?
4. What does `loss.backward()` compute, and what does `optimizer.step()` change?
5. Why clip gradients, and what is the gradient scaler for?
6. What does a cosine schedule with warm-up do to the learning rate?
7. Training loss fell to 0.59. Why does that not prove the model is good?
8. What are the signs of overfitting here, and how would you reduce them?

## Evaluation

1. Why split by symptom group instead of by random prompt?
2. Why run 3 seeds, and what does `temperature` change?
3. The false alarm rate rose to 33%. How would you bring it down?
4. Why can 100% on keyword checks still hide bad replies?
5. What would a better evaluation add?

## Transformers

1. What do attention queries, keys, and values do?
2. Why target `q_proj`, `k_proj`, `v_proj`, and `o_proj`?
3. What are tokens, and what is a chat template?
4. What is the difference between pre-training, supervised fine-tuning, and reinforcement learning?

## Systems

1. What does MLflow record, and how would you compare two runs?
2. What does the Dockerfile do at each step?
3. What are a Kubernetes Deployment, Service, and readiness probe?
4. Why did the agent fail with this small model, and how would you fix it?

## Practice

1. Run `src/train.py` and `src/evaluate.py` yourself.
2. Add 30 ordinary examples, retrain, and record what happens to the false alarm rate.
3. Change `lora_r` from 16 to 8 and compare the results.
