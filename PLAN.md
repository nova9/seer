# LLM from Scratch — Learning Plan (Rust + Candle)

**Goal:** Build a GPT-style language model end-to-end, learning Rust and ML concepts at each step.

---

## Phase 1 — Tensors & Candle Basics
**Learn:** Candle's tensor API, Rust ownership with numeric data, device abstraction (CPU/GPU)

- [ ] 1.1 Tensor creation and arithmetic — shape, dtype, add/mul/matmul
- [ ] 1.2 Indexing and slicing tensors
- [ ] 1.3 Saving and loading tensors (safetensors format)
- [ ] 1.4 Understand `Device`, `DType`, and how Candle handles memory

---

## Phase 2 — Tokenizer & Dataset Pipeline
**Learn:** Text preprocessing, Rust iterators, file I/O, trait objects

- [x] 2.1 Load a BPE tokenizer (`tokenizer.json`)
- [x] 2.2 Build a `Dataset` struct that reads a text file and yields token batches
- [x] 2.3 Implement a sliding-window `DataLoader` iterator

---

## Phase 3 — Embedding Layer
**Learn:** `candle_nn`, `nn::Embedding`, gradient flow basics

- [x] 3.1 Token embedding lookup table
- [x] 3.2 Positional encoding (learned vs sinusoidal)
- [x] 3.3 Combine token + position embeddings

---

## Phase 4 — Attention Mechanism
**Learn:** The core of transformers, matrix ops in Candle, masking

- [ ] 4.1 Scaled dot-product attention
- [ ] 4.2 Causal (autoregressive) mask
- [ ] 4.3 Multi-head attention (split heads, concat, project)

---

## Phase 5 — Transformer Block
**Learn:** Layer norm, MLP/FFN, residual connections, Rust structs as modules

- [ ] 5.1 `LayerNorm` wrapper
- [ ] 5.2 Feed-forward network (MLP)
- [ ] 5.3 Full `TransformerBlock` struct implementing `candle_nn::Module`

---

## Phase 6 — GPT Model
**Learn:** Stacking blocks, language model head, parameter counting

- [ ] 6.1 Stack N transformer blocks
- [ ] 6.2 Final linear projection to vocabulary (`lm_head`)
- [ ] 6.3 Cross-entropy loss for next-token prediction

---

## Phase 7 — Training Loop
**Learn:** Optimizers, backprop in Candle, Rust error handling at scale

- [ ] 7.1 AdamW optimizer via `candle_nn::optim`
- [ ] 7.2 Forward pass → loss → backward → step
- [ ] 7.3 Loss logging, checkpointing to safetensors

---

## Phase 8 — Text Generation
**Learn:** Autoregressive inference, temperature, top-k sampling

- [ ] 8.1 Greedy decoding
- [ ] 8.2 Temperature + top-k sampling
- [ ] 8.3 KV-cache for efficient generation (stretch goal)

---

## Notes

- **Suggested dataset:** TinyShakespeare (~1MB) — small enough for CPU training, rich enough to produce coherent output
- **Current status:** Phase 2.1 complete (tokenizer working)
- **Suggested next step:** Phase 1 (tensor fundamentals) to solidify the Candle foundation
