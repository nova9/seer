use candle_core::{Result, Tensor};
use candle_nn::VarBuilder;

use crate::attention::MultiHeadAttention;
use crate::layer_norm::MyLayerNorm;
use crate::mlp::Mlp;

pub struct TransformerBlock {
    norm1: MyLayerNorm,
    attn:  MultiHeadAttention,
    norm2: MyLayerNorm,
    mlp:   Mlp,
}

impl TransformerBlock {
    pub fn new(embed_dim: usize, num_heads: usize, vb: VarBuilder) -> Result<Self> {
        let norm1 = MyLayerNorm::new(embed_dim, vb.pp("norm1"))?;
        let attn  = MultiHeadAttention::new(embed_dim, num_heads, vb.pp("attn"))?;
        let norm2 = MyLayerNorm::new(embed_dim, vb.pp("norm2"))?;
        let mlp   = Mlp::new(embed_dim, vb.pp("mlp"))?;
        Ok(Self { norm1, attn, norm2, mlp })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // sub-layer 1: attention with residual
        let x1 = (x + self.attn.forward(&self.norm1.forward(x)?)?)?;

        // sub-layer 2: mlp with residual
        let x2 = (&x1 + self.mlp.forward(&self.norm2.forward(&x1)?)?)?;

        Ok(x2)
    }
}
