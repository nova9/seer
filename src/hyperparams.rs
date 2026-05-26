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
            "H100" => Self {
                embed_dim: 512,
                num_heads: 16,
                n_layers: 12,
                seq_len: 1024,
                batch_size: 64,
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
