use candle_core::{Device, IndexOp, Result, Tensor};
use rand::Rng;

use crate::embedding::{LearnedPosEmbedding, TokenEmbedding};
use crate::gpt::Gpt;

pub fn generate(
    prompt_tokens: Vec<u32>,
    max_new_tokens: usize,
    token_embed: &TokenEmbedding,
    pos_embed: &LearnedPosEmbedding,
    model: &Gpt,
    device: &Device,
    temperature: f64,
    top_k: usize,
) -> Result<Vec<u32>> {
    let mut tokens = prompt_tokens; // we'll grow this Vec each step

    for _ in 0..max_new_tokens {
        let seq_len = tokens.len();

        // shape: [1, seq_len]  (batch size of 1 for inference)
        let input = Tensor::from_vec(tokens.clone(), (1, seq_len), device)?;

        // shape: [1, seq_len, embed_dim]
        let tok = token_embed.forward(&input)?;

        // shape: [seq_len, embed_dim]  →  broadcast_add needs same rank, so unsqueeze
        let pos = pos_embed.forward(seq_len, device)?.unsqueeze(0)?;

        // shape: [1, seq_len, embed_dim]
        let x = tok.broadcast_add(&pos)?;

        // shape: [1, seq_len, vocab_size]
        let logits = model.forward(&x)?;

        // grab only the last position's logits → shape: [vocab_size]
        let last_logits = logits.i((0, seq_len - 1))?;

        // argmax over vocab dimension → the index with the highest score
        let next_token = sample_token(&last_logits, temperature, top_k)?;

        tokens.push(next_token);
    }

    Ok(tokens)
}

fn sample_token(logists: &Tensor, temperature: f64, top_k: usize) -> Result<u32> {
    let mut logit_vector: Vec<f32> = logists.to_dtype(candle_core::DType::F32)?.to_vec1()?;

    let mut sorted = logit_vector.clone();
    sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());

    let threshold = sorted[top_k - 1];

    for v in logit_vector.iter_mut() {
        if *v < threshold {
            *v = f32::NEG_INFINITY;
        }
    }
    for v in logit_vector.iter_mut() {
        *v /= temperature as f32;
    }

    let max = logit_vector
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);
    for v in logit_vector.iter_mut() {
        *v = (*v - max).exp();
    }
    let sum: f32 = logit_vector.iter().sum();
    for v in logit_vector.iter_mut() {
        *v /= sum;
    }

    let r = rand::thread_rng().gen_range(0.0f32..1.0f32);
    let mut cumulative_sum = 0.0f32;
    for (i, &p) in logit_vector.iter().enumerate() {
        cumulative_sum += p;
        if cumulative_sum >= r {
            return Ok(i as u32);
        }
    }

    Ok((logit_vector.len() - 1) as u32)
}
