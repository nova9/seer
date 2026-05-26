pub struct HyperParams {
    pub embed_dim: usize,
    pub num_heads: usize,
    pub n_layers: usize,
    pub seq_len: usize,
    pub batch_size: usize,
    pub lr: f64,
    pub n_epochs: usize,
}

impl HyperParams {
    pub fn preset(name: &str) -> Self {
        match name {
            "cpu" => Self {
                embed_dim: 64,
                num_heads: 4,
                n_layers: 2,
                seq_len: 128,
                batch_size: 2,
                lr: 3e-4,
                n_epochs: 10,
            },
            "M3" => Self {
                embed_dim: 128,
                num_heads: 8,
                n_layers: 4,
                seq_len: 256,
                batch_size: 8,
                lr: 3e-4,
                n_epochs: 10,
            },
            "T4" => Self {
                embed_dim: 256,
                num_heads: 8,
                n_layers: 6,
                seq_len: 512,
                batch_size: 16,
                lr: 3e-4,
                n_epochs: 10,
            },
            "V100" => Self {
                embed_dim: 320,
                num_heads: 10,
                n_layers: 7,
                seq_len: 640,
                batch_size: 32,
                lr: 3e-4,
                n_epochs: 10,
            },
            "A40" => Self {
                embed_dim: 384,
                num_heads: 12,
                n_layers: 8,
                seq_len: 768,
                batch_size: 24, // 4 score copies × 24 × 12 × 768² × 4B × 8L ≈ 22 GB
                lr: 3e-4,
                n_epochs: 10,
            },
            "A100_SXM" => Self {
                embed_dim: 512,
                num_heads: 16,
                n_layers: 10,
                seq_len: 1024,
                batch_size: 16, // 4 score copies × 16 × 16 × 1024² × 4B × 10L ≈ 43 GB
                lr: 3e-4,
                n_epochs: 10,
            },
            "H100" => Self {
                embed_dim: 512,
                num_heads: 16,
                n_layers: 12,
                seq_len: 1024,
                batch_size: 16, // 4 score copies × 16 × 16 × 1024² × 4B × 12L ≈ 51 GB
                lr: 3e-4,
                n_epochs: 10,
            },
            _ => {
                eprintln!("Unknown preset '{name}', falling back to 'cpu'");
                Self::preset("cpu")
            }
        }
    }
}
