use crate::dataset::Dataset;

pub struct DataLoader {
    dataset: Dataset,
    pos: usize,
    stride: usize,
    batch_size: usize,
}

impl DataLoader {
    pub fn new(dataset: Dataset, stride: usize, batch_size: usize) -> Self {
        Self {
            dataset,
            pos: 0,
            stride,
            batch_size,
        }
    }
}

impl Iterator for DataLoader {
    type Item = (Vec<u32>, Vec<u32>);

    fn next(&mut self) -> Option<Self::Item> {
        let last = self.pos + (self.batch_size - 1) * self.stride;
        if last >= self.dataset.len() {
            return None;
        }

        let mut inputs = Vec::new();
        let mut targets = Vec::new();

        for i in 0..self.batch_size {
            let (inp, tgt) = self.dataset.get(self.pos + i * self.stride);
            inputs.extend(inp);
            targets.extend(tgt);
        }

        self.pos += self.batch_size * self.stride;

        Some((inputs, targets))
    }
}
