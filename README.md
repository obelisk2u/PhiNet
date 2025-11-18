# Neural Temporal Point Process Reproduction (NeurIPS 2019)

This project reproduces the core modeling pipeline from the NeurIPS 2019 paper  
**“Fully Neural Network Based Model for General Temporal Point Processes”** 

The goal is to demonstrate how to model temporal point processes using:

- A GRU history encoder
- A neural cumulative hazard function Φ(τ | h)
- Automatic differentiation to obtain intensity λ(τ | h)
- The exact point-process log-likelihood
- Training and evaluation on real-world sequences  
  (San Francisco 911 call dataset)

---

## Contents

- `src/model.py` — GRU encoder, hazard MLP, and λ via autodiff  
- `src/preprocess_data.py` — preprocess SF 911 calls into inter-arrival sequences  
- `src/real_data.py` — Dataset + collate_fn with padding + masks  
- `src/train_sf.py` — training loop, evaluation, TensorBoard logging  
- `data/` — preprocessed `.pt` file (after running preprocessing)

---

## 1. Project Motivation

Temporal point processes (TPPs) model sequences of event times:  
t₁ < t₂ < … < tₙ.  

Classical models (Poisson, Hawkes, Renewal) use hand-designed intensities λ(t | history).  
The NeurIPS 2019 paper replaces these with a fully neural architecture:

1. Encode event history using a GRU  
2. Predict the cumulative hazard Φ(τ | h) using a small MLP  
3. Obtain intensity via  
   λ(τ | h) = dΦ/dτ  
   using PyTorch’s autodiff  
4. Train using the exact log-likelihood:  
   ℓᵢ = log λ(τᵢ | hᵢ) − Φ(τᵢ | hᵢ)

This repo reproduces this pipeline from scratch.

---

## 2. Real Data: SF 911 Calls

We follow the original paper’s real-data experiment using 911 call logs:

- Group events by address
- Keep top K addresses with >= minimum call volume
- Convert timestamps → seconds since first event  
- Compute inter-arrival times Δtᵢ
- Truncate sequences to a maximum length to reduce compute load
- Split into train/test segments per address
- Save as a PyTorch file: `sf_tpp_sequences.pt`

### Preprocessing Command

```bash
python preprocess_data.py \
  --csv ../data/police-department-calls-for-service.csv \
  --out ../data/sf_tpp_sequences.pt \
  --top-k 100 \
  --min-events 50 \
  --max-events-per-seq 500
```

---

## 3. Model Architecture

### GRU Encoder
Takes inter-arrival sequence Δt₁, Δt₂, …, Δtₙ  
Produces hidden states h₁, h₂, …, hₙ.

### Hazard Network (Φ-network)
Small MLP predicting cumulative hazard:

Φ(τᵢ | hᵢ) > 0 via softplus.

### Intensity via Autodiff
PyTorch computes:

λ(τᵢ | hᵢ) = ∂Φ(τᵢ | hᵢ) / ∂τᵢ

### Log-Likelihood
The exact TPP log-likelihood:

ℓᵢ = log λ(τᵢ | hᵢ) − Φ(τᵢ | hᵢ)

Average across non-padded positions using a binary mask.

---

## 4. Training

Run training on real SF 911 sequences:

```bash
python train_sf.py \
  --data-path ../data/sf_tpp_sequences.pt \
  --logdir ../runs_sf \
  --epochs 20 \
  --batch-size 16
```

This produces:

- Train/test log-likelihood curves  
- Gradient norm curves  
- `best_sf_model.pt` saved in `checkpoints_sf/`

TensorBoard:

```bash
tensorboard --logdir runs_sf
```

---

## 5. Results Summary

The model reproduces the expected behaviors:

- Train and test log-likelihood improve together  
- No overfitting on real sequences  
- Gradient norms decrease smoothly  
- Stable GRU + MLP hazard learning  
- Intensity functions learned automatically from data  
- Clear improvement over a constant-rate baseline
---

## 6. Repo Structure

```
PhiNet/
│
├── data/
│   └── sf_tpp_sequences.pt        # generated via preprocess
│
├── src/
│   ├── model.py                   # GRU + hazard network
│   ├── preprocess_data.py         # CSV → TPP sequences
│   ├── real_data.py               # Dataset + padding/mask
│   └── train_sf.py                # training loop + TB logs
```

---

## 7. Requirements

```
python >= 3.10
torch >= 2.0
pandas
numpy
tensorboard
```

Install:

```bash
pip install -r requirements.txt
```

---

## 8. Citation

Original paper:

**Takahiro Omi, Naonori Ueda, Kazuyuki Aihara.**  
“Fully Neural Network Based Model for General Temporal Point Processes.”  
NeurIPS 2019.

---