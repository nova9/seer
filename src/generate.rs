use candle_core::{Device, IndexOp, Result, Tensor};

use crate::embedding::{LearnedPosEmbedding, TokenEmbedding};
use crate::gpt::Gpt;

pub fn generate(
    prompt_tokens: Vec<u32>,
    max_new_tokens: usize,
    token_embed: &TokenEmbedding,
    pos_embed: &LearnedPosEmbedding,
    model: &Gpt,
    device: &Device,
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
        let next_token = last_logits.argmax(0)?.to_scalar::<u32>()?;

        tokens.push(next_token);
    }

    Ok(tokens)
}
