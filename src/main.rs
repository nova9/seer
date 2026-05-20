mod dataloader;
mod dataset;
use tokenizers::Tokenizer;

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let tokenizer = Tokenizer::from_file("tokenizer.json")?;

    let dataset = dataset::Dataset::new("shakespeare.txt", &tokenizer, 256)?;

    let dataloader = dataloader::DataLoader::new(dataset, 2);

    for batch in dataloader {
        println!("{:?}", batch);
        println!("")
    }

    Ok(())
}
