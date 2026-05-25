use candle_core::{Result, Tensor};
use candle_nn::{LayerNorm, Module, VarBuilder, layer_norm};

pub struct MyLayerNorm {
    inner: LayerNorm,
}

impl MyLayerNorm {
    pub fn new(embed_dim: usize, vb: VarBuilder) -> Result<Self> {
        let inner = layer_norm(embed_dim, 1e-5, vb)?;
        Ok(Self { inner })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.inner.forward(x)
    }
}
