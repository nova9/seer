use crate::dataset::Dataset;

pub struct DataLoader {
    dataset: Dataset,
    pos: usize,
    stride: usize,
}

impl DataLoader {
    pub fn new(dataset: Dataset, stride: usize) -> Self {
        Self {
            dataset,
            pos: 0,
            stride,
        }
    }
}

impl Iterator for DataLoader {
    type Item = (Vec<u32>, Vec<u32>);

    fn next(&mut self) -> Option<Self::Item> {
        if self.pos >= self.dataset.len() {
            return None;
        }

        let batch = self.dataset.get(self.pos);
        self.pos += self.stride;
        Some(batch)
    }
}
