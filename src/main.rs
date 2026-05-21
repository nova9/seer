mod dataloader;
mod dataset;
mod embedding;
use candle_core::{DType, Device, Tensor};
use candle_nn::{VarBuilder, VarMap};
use tokenizers::Tokenizer;

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let tokenizer = Tokenizer::from_file("tokenizer.json")?;

    // set up the parameter store
    let device = Device::Cpu;
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);

    // create the token embedding layer
    let embed_dim = 128;
    let vocab_size = tokenizer.get_vocab_size(true);
    let token_embed = embedding::TokenEmbedding::new(vocab_size, embed_dim, vb.pp("token_emb"))?;

    // create the positional embedding layer
    let max_seq_len = 1024;
    let pos_embed = embedding::LearnedPosEmbedding::new(max_seq_len, embed_dim, vb.pp("pos_emb"))?;

    // data
    let seq_len = 256;
    let dataset = dataset::Dataset::new("shakespeare.txt", &tokenizer, seq_len)?;
    let dataloader = dataloader::DataLoader::new(dataset, 2);

    for (inputs, targets) in dataloader.take(1) {
        let input_tensor = Tensor::new(inputs.as_slice(), &device)?;
        println!("{:?}", input_tensor.shape());
        let _target_tensor = Tensor::new(targets.as_slice(), &device)?;

        let tok_out = token_embed.forward(&input_tensor)?;
        let pos_out = pos_embed.forward(seq_len, &device)?;

        let combined = (tok_out + pos_out)?;

        println!("output shape: {:?}", combined.shape()); // [seq_len, embed_dim]

        println!("")
    }

    Ok(())
}
