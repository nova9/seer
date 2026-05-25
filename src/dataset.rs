use std::fs;
use tokenizers::Tokenizer;

#[derive(Clone)]
pub struct Dataset {
    tokens: Vec<u32>,
    seq_len: usize,
}

impl Dataset {
    pub fn new(
        path: &str,
        tokenizer: &Tokenizer,
        seq_len: usize,
    ) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        let text = fs::read_to_string(path)?;
        let encoding = tokenizer.encode(text, false)?;
        let tokens = encoding.get_ids().to_vec();

        Ok(Self { tokens, seq_len })
    }

    pub fn len(&self) -> usize {
        self.tokens.len() - self.seq_len
    }

    pub fn get(&self, index: usize) -> (Vec<u32>, Vec<u32>) {
        let input = self.tokens[index..(index + self.seq_len)].to_vec();
        let target = self.tokens[(index + 1)..(index + self.seq_len + 1)].to_vec();
        (input, target)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_dataset(tokens: Vec<u32>, seq_len: usize) -> Dataset {
        Dataset { tokens, seq_len }
    }

    #[test]
    fn test_len() {
        let ds = make_dataset(vec![1, 2, 3, 4, 5], 3);
        assert_eq!(ds.len(), 2);
    }

    #[test]
    fn test_get_returns_correct_slices() {
        let ds = make_dataset(vec![10, 20, 30, 40, 50], 3);
        let (input, target) = ds.get(0);
        assert_eq!(input, vec![10, 20, 30]);
        assert_eq!(target, vec![20, 30, 40]);
    }

    #[test]
    fn test_get_at_last_valid_index() {
        let ds = make_dataset(vec![1, 2, 3, 4, 5], 3);
        let (input, target) = ds.get(ds.len() - 1);
        assert_eq!(input, vec![2, 3, 4]);
        assert_eq!(target, vec![3, 4, 5]);
    }

    #[test]
    fn test_input_target_offset_by_one() {
        let ds = make_dataset(vec![0, 1, 2, 3], 2);
        let (input, target) = ds.get(0);
        assert_eq!(target[0], input[1]);
    }
}
