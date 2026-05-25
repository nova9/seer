use candle_core::{Result, Tensor};
use candle_nn::{Linear, VarBuilder, linear_no_bias, Module};

use crate::transformer_block::TransformerBlock;

pub struct Gpt {
    blocks: Vec<TransformerBlock>,
    lm_head: Linear,
}

impl Gpt {
    pub fn new(
        n_layers: usize,
        embed_dim: usize,
        num_heads: usize,
        vocab_size: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let blocks = (0..n_layers)
            .map(|i| TransformerBlock::new(embed_dim, num_heads, vb.pp(format!("block_{i}"))))
            .collect::<Result<Vec<_>>>()?;
        let lm_head = linear_no_bias(embed_dim, vocab_size, vb.pp("lm_head"))?;
        Ok(Self { blocks, lm_head })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut out = x.clone();
        for block in &self.blocks {
            out = block.forward(&out)?;
        }
        // out is [batch, seq_len, embed_dim]
        // lm_head projects to [batch, seq_len, vocab_size]
        self.lm_head.forward(&out)
    }
}
