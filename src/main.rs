mod dataset;
use tokenizers::Tokenizer;

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let tokenizer = Tokenizer::from_file("tokenizer.json")?;

    let dataset = dataset::Dataset::new("shakespeare.txt", &tokenizer, 256)?;

    let (input, target) = dataset.get(0);
    println!("input:  {:?}", input);
    println!("target: {:?}", target);

    Ok(())
}
