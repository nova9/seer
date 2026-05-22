use candle_core::{Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder, linear_no_bias};

pub struct ScaledDotProductAttention {
    w_q: Linear,
    w_k: Linear,
    w_v: Linear,
    scale: f64,
}

impl ScaledDotProductAttention {
    pub fn new(embed_dim: usize, vb: VarBuilder) -> Result<Self> {
        let w_q = linear_no_bias(embed_dim, embed_dim, vb.pp("w_q"))?;
        let w_k = linear_no_bias(embed_dim, embed_dim, vb.pp("w_k"))?;
        let w_v = linear_no_bias(embed_dim, embed_dim, vb.pp("w_v"))?;

        let scale = (embed_dim as f64).sqrt();
        Ok(Self {
            w_q,
            w_k,
            w_v,
            scale,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let q = self.w_q.forward(x)?;
        let k = self.w_k.forward(x)?;
        let v = self.w_v.forward(x)?;

        let scores = q.matmul(&k.t()?)?;
        let scores = (scores / self.scale)?;

        let seq_len = x.dim(1)?;
        let mask = casual_mask(seq_len, x.device())?;

        let scores = scores.broadcast_add(&mask.unsqueeze(0)?)?;

        let weights = candle_nn::ops::softmax(&scores, 2)?;

        weights.matmul(&v)
    }
}

fn casual_mask(seq_len: usize, device: &candle_core::Device) -> Result<Tensor> {
    let mask: Vec<f32> = (0..seq_len)
        .flat_map(|row| {
            (0..seq_len).map(move |col| if col <= row { 0.0 } else { f32::NEG_INFINITY })
        })
        .collect();

    Tensor::from_vec(mask, (seq_len, seq_len), device)
}
