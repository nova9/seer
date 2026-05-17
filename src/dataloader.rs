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
