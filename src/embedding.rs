use candle_core::{Device, Result, Tensor};
use candle_nn::{Embedding, Module, VarBuilder, embedding};

pub struct TokenEmbedding {
    embed: Embedding,
}

impl TokenEmbedding {
    pub fn new(vocab_size: usize, embed_dim: usize, vb: VarBuilder) -> Result<Self> {
        let embed = embedding(vocab_size, embed_dim, vb)?;
        Ok(Self { embed })
    }

    pub fn forward(&self, token_ids: &Tensor) -> Result<Tensor> {
        self.embed.forward(token_ids)
    }
}

pub struct LearnedPosEmbedding {
    embed: Embedding,
}

impl LearnedPosEmbedding {
    pub fn new(max_seq_len: usize, embed_dim: usize, vb: VarBuilder) -> Result<Self> {
        let embed = embedding(max_seq_len, embed_dim, vb)?;
        Ok(Self { embed })
    }

    pub fn forward(&self, seq_len: usize, device: &Device) -> Result<Tensor> {
        // Build [0, 1, 2, ..., seq_len-1] and look them up
        let positions = Tensor::arange(0u32, seq_len as u32, device)?;
        self.embed.forward(&positions)
    }
}
