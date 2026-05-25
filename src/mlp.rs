use candle_core::{Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder, linear};

pub struct Mlp {
    fc1: Linear, // expand: embed_dim → 4 * embed_dim
    fc2: Linear, // compress: 4 * embed_dim → embed_dim
}

impl Mlp {
    pub fn new(embed_dim: usize, vb: VarBuilder) -> Result<Self> {
        let hidden = embed_dim * 4;
        let fc1 = linear(embed_dim, hidden, vb.pp("fc1"))?;
        let fc2 = linear(hidden, embed_dim, vb.pp("fc2"))?;
        Ok(Self { fc1, fc2 })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.fc1.forward(x)?; // [batch, seq_len, 4*embed_dim]
        let x = x.gelu()?; // still same shape, values changed
        self.fc2.forward(&x) // [batch, seq_len, embed_dim]
    }
}
