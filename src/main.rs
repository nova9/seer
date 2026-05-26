mod attention;
mod dataloader;
mod dataset;
mod embedding;
mod generate;
mod gpt;
mod hyperparams;
mod layer_norm;
mod mlp;
mod transformer_block;

use candle_core::{DType, Device, Tensor};
use candle_nn::optim::{AdamW, Optimizer};
use candle_nn::{VarBuilder, VarMap};
use hyperparams::HyperParams;
use std::fs;
use std::path::Path;
use tokenizers::Tokenizer;

fn get_device() -> candle_core::Result<Device> {
    #[cfg(feature = "cuda")]
    if let Ok(dev) = Device::new_cuda(0) {
        println!("Using CUDA device");
        return Ok(dev);
    }
    #[cfg(feature = "metal")]
    if let Ok(dev) = Device::new_metal(0) {
        println!("Using Metal device");
        return Ok(dev);
    }
    println!("Using CPU device");
    Ok(Device::Cpu)
}

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let args: Vec<String> = std::env::args().collect();
    let only_generate = args.iter().any(|a| a == "--generate");
    let preset_name = args.iter()
        .position(|a| a == "--preset")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
        .unwrap_or("M3");

    let hp = HyperParams::preset(preset_name);
    println!("Preset: {preset_name} | embed={} heads={} layers={} seq={} batch={} lr={}", hp.embed_dim, hp.num_heads, hp.n_layers, hp.seq_len, hp.batch_size, hp.lr);

    println!("Loading tokenizer...");
    let tokenizer = Tokenizer::from_file("tokenizer.json")?;

    // set up the parameter store
    let device = get_device()?;
    let mut varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    // create the token embedding layer
    let vocab_size = tokenizer.get_vocab_size(true);
    println!("Vocab size: {vocab_size}");
    let token_embed = embedding::TokenEmbedding::new(vocab_size, hp.embed_dim, vb.pp("token_emb"))?;

    // create the positional embedding layer
    let pos_embed = embedding::LearnedPosEmbedding::new(hp.seq_len, hp.embed_dim, vb.pp("pos_emb"))?;

    // create GPT model (stack of N transformer blocks)
    println!("Building model ({} layers, {} dim, {} heads)...", hp.n_layers, hp.embed_dim, hp.num_heads);
    let model = gpt::Gpt::new(hp.n_layers, hp.embed_dim, hp.num_heads, vocab_size, vb.pp("gpt"))?;

    let checkpoint = "checkpoint.safetensors";
    let progress_file = "checkpoint.progress";

    if only_generate {
        if !Path::new(checkpoint).exists() {
            eprintln!("No checkpoint found — train first.");
            std::process::exit(1);
        }
        println!("Loading checkpoint for generation...");
        varmap.load(checkpoint)?;
    } else {
        let (start_epoch, start_step) = if Path::new(checkpoint).exists() {
            let progress = fs::read_to_string(progress_file).unwrap_or("0,0".to_string());
            let mut parts = progress.trim().splitn(2, ',');
            let epoch = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0);
            let step = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0);
            println!("Resuming from {checkpoint} (epoch {epoch}, step {step})...");
            varmap.load(checkpoint)?;
            (epoch, step)
        } else {
            println!("No checkpoint found, starting fresh.");
            (0, 0)
        };

        let mut optimizer = AdamW::new_lr(varmap.all_vars(), hp.lr)?;

        println!("Tokenizing dataset (this may take a moment)...");
        let dataset = dataset::Dataset::new("story.txt", &tokenizer, hp.seq_len)?;
        println!("Dataset ready: {} samples", dataset.len());

        for epoch in start_epoch..hp.n_epochs {
            let resume_step = if epoch == start_epoch { start_step } else { 0 };
            println!("--- Epoch {epoch} (starting from step {resume_step}) ---");
            let dataloader =
                dataloader::DataLoader::from_step(dataset.clone(), 1, hp.batch_size, resume_step);
            for (step, (inputs, targets)) in
                dataloader.enumerate().map(|(s, b)| (s + resume_step, b))
            {
                let input_tensor = Tensor::from_vec(inputs, (hp.batch_size, hp.seq_len), &device)?;

                let tok_out = token_embed.forward(&input_tensor)?;
                let pos_out = pos_embed.forward(hp.seq_len, &device)?;
                let pos_out_unsqueezed = pos_out.unsqueeze(0)?;
                let combined = tok_out.broadcast_add(&pos_out_unsqueezed)?;

                let gpt_out = model.forward(&combined)?;

                let (_, _, vocab) = gpt_out.dims3()?;

                let logits_flat = gpt_out.reshape((hp.batch_size * hp.seq_len, vocab))?;

                let target_tensor = Tensor::from_vec(targets, (hp.batch_size * hp.seq_len,), &device)?;

                let loss = candle_nn::loss::cross_entropy(&logits_flat, &target_tensor)?;

                optimizer.backward_step(&loss)?;

                println!(
                    "epoch {epoch} step {step} loss: {:.4}",
                    loss.to_scalar::<f32>()?
                );
                if step % 50 == 0 {
                    varmap.save(checkpoint)?;
                    fs::write(progress_file, format!("{epoch},{step}"))?;
                }
            }
        }
    }

    // --- Text generation ---
    println!("\n--- Generating text ---");
    let prompt = "To be or not to be";
    let prompt_tokens: Vec<u32> = tokenizer
        .encode(prompt, false)
        .map_err(|e| format!("tokenizer error: {e}"))?
        .get_ids()
        .to_vec();

    let top_k = 40;
    let temperature = 0.8;
    let generated = generate::generate(
        prompt_tokens,
        100, // generate 100 new tokens
        &token_embed,
        &pos_embed,
        &model,
        &device,
        temperature,
        top_k,
    )?;

    let output = tokenizer
        .decode(&generated, true)
        .map_err(|e| format!("decode error: {e}"))?;
    println!("{output}");

    Ok(())
}
