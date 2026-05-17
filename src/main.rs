use tokenizers::Tokenizer;

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let tokenizer = Tokenizer::from_file("tokenizer.json")?;

    let text = "Hello World";
    let encoding = tokenizer.encode(text, false)?;

    let tokens: Vec<String> = encoding.get_tokens().to_vec();
    let ids: Vec<u32> = encoding.get_ids().to_vec();

    println!("{:?}", tokens);
    println!("{:?}", ids);

    Ok(())
}
