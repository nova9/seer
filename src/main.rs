mod attention;
mod dataloader;
mod dataset;
mod embedding;
mod gpt;
mod layer_norm;
mod mlp;
mod transformer_block;

use candle_core::{DType, Device, Tensor};
use candle_nn::optim::{AdamW, Optimizer};
use candle_nn::{VarBuilder, VarMap};
use tokenizers::Tokenizer;

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    println!("Loading tokenizer...");
    let tokenizer = Tokenizer::from_file("tokenizer.json")?;

    // set up the parameter store
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    // create the token embedding layer
    let embed_dim = 128;
    let vocab_size = tokenizer.get_vocab_size(true);
    println!("Vocab size: {vocab_size}");
    let token_embed = embedding::TokenEmbedding::new(vocab_size, embed_dim, vb.pp("token_emb"))?;

    // create the positional embedding layer
    let max_seq_len = 1024;
    let pos_embed = embedding::LearnedPosEmbedding::new(max_seq_len, embed_dim, vb.pp("pos_emb"))?;

    // create GPT model (stack of N transformer blocks)
    let num_heads = 4;
    let n_layers = 4;
    println!("Building model ({n_layers} layers, {embed_dim} dim, {num_heads} heads)...");
    let model = gpt::Gpt::new(n_layers, embed_dim, num_heads, vocab_size, vb.pp("gpt"))?;

    let mut optimizer = AdamW::new_lr(varmap.all_vars(), 3e-4)?;

    // data
    let seq_len = 256;
    println!("Tokenizing dataset (this may take a moment)...");
    let dataset = dataset::Dataset::new("shakespeare.txt", &tokenizer, seq_len)?;
    let batch_size = 4;
    println!("Dataset ready: {} samples", dataset.len());

    let n_epochs = 3;

    for epoch in 0..n_epochs {
        println!("--- Epoch {epoch} ---");
        let dataloader = dataloader::DataLoader::new(dataset.clone(), 2, batch_size);
        for (step, (inputs, targets)) in dataloader.enumerate() {
            let input_tensor = Tensor::from_vec(inputs, (batch_size, seq_len), &device)?;

            let tok_out = token_embed.forward(&input_tensor)?;
            let pos_out = pos_embed.forward(seq_len, &device)?;
            let pos_out_unsqueezed = pos_out.unsqueeze(0)?;
            let combined = tok_out.broadcast_add(&pos_out_unsqueezed)?;

            let gpt_out = model.forward(&combined)?;

            let (_, _, vocab) = gpt_out.dims3()?;

            let logits_flat = gpt_out.reshape((batch_size * seq_len, vocab))?;

            let target_tensor = Tensor::from_vec(targets, (batch_size * seq_len,), &device)?;

            let loss = candle_nn::loss::cross_entropy(&logits_flat, &target_tensor)?;

            optimizer.backward_step(&loss)?;

            if step % 1 == 0 {
                println!(
                    "epoch {epoch} step {step} loss: {:.4}",
                    loss.to_scalar::<f32>()?
                );
            }
        }
    }

    Ok(())
}
