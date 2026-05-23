use candle_core::{Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder, linear_no_bias};

#[allow(dead_code)]
pub struct ScaledDotProductAttention {
    w_q: Linear,
    w_k: Linear,
    w_v: Linear,
    scale: f64,
}

impl ScaledDotProductAttention {
    #[allow(dead_code)]
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

    #[allow(dead_code)]
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let q = self.w_q.forward(x)?;
        let k = self.w_k.forward(x)?;
        let v = self.w_v.forward(x)?;

        let scores = q.matmul(&k.t()?)?;
        let scores = (scores / self.scale)?;

        let seq_len = x.dim(1)?;
        let mask = causal_mask(seq_len, x.device())?;

        let scores = scores.broadcast_add(&mask.unsqueeze(0)?)?;

        let weights = candle_nn::ops::softmax(&scores, 2)?;

        weights.matmul(&v)
    }
}

pub struct MultiHeadAttention {
    w_q: Linear,
    w_k: Linear,
    w_v: Linear,
    w_o: Linear,
    num_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl MultiHeadAttention {
    pub fn new(embed_dim: usize, num_heads: usize, vb: VarBuilder) -> Result<Self> {
        let head_dim = embed_dim / num_heads;

        let w_q = linear_no_bias(embed_dim, embed_dim, vb.pp("w_q"))?;
        let w_k = linear_no_bias(embed_dim, embed_dim, vb.pp("w_k"))?;
        let w_v = linear_no_bias(embed_dim, embed_dim, vb.pp("w_v"))?;
        let w_o = linear_no_bias(embed_dim, embed_dim, vb.pp("w_o"))?;

        Ok(Self {
            w_q,
            w_k,
            w_v,
            w_o,
            num_heads,
            head_dim,
            scale: (head_dim as f64).sqrt(), // scale by head_dim, not embed_dim
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (batch, seq_len, _) = x.dims3()?;
        let h = self.num_heads;
        let d = self.head_dim;

        let q = self.w_q.forward(x)?;
        let k = self.w_k.forward(x)?;
        let v = self.w_v.forward(x)?;

        let q = q
            .reshape((batch, seq_len, h, d))?
            .transpose(1, 2)?
            .contiguous()?;
        let k = k
            .reshape((batch, seq_len, h, d))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((batch, seq_len, h, d))?
            .transpose(1, 2)?
            .contiguous()?;

        let scores = q.matmul(&k.transpose(2, 3)?)?;
        let scores = (scores / self.scale)?;

        let mask = causal_mask(seq_len, x.device())?;
        let scores = scores.broadcast_add(&mask.unsqueeze(0)?.unsqueeze(0)?)?;

        let weights = candle_nn::ops::softmax(&scores, 3)?;

        let out = weights.matmul(&v)?;
        let out = out.transpose(1, 2)?.contiguous()?;
        let out = out.reshape((batch, seq_len, h * d))?;

        self.w_o.forward(&out)
    }
}

fn causal_mask(seq_len: usize, device: &candle_core::Device) -> Result<Tensor> {
    let mask: Vec<f32> = (0..seq_len)
        .flat_map(|row| {
            (0..seq_len).map(move |col| if col <= row { 0.0 } else { f32::NEG_INFINITY })
        })
        .collect();

    Tensor::from_vec(mask, (seq_len, seq_len), device)
}
