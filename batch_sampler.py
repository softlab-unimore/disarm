import torch
from torch.utils.data import Sampler

class BalancedSampler(Sampler):
    def __init__(self, dataset1, dataset2, batch_size):
        self.dataset1 = dataset1
        self.dataset2 = dataset2
        self.batch_size = batch_size
        self.total_size = len(dataset1) + len(dataset2)
        self.indices1 = list(range(len(dataset1)))
        self.indices2 = list(range(len(dataset2)))
        self.epoch = 0

    def __iter__(self):
        self.epoch += 1
        batch = []
        indices1 = self.indices1.copy()
        indices2 = self.indices2.copy()

        indices1 = torch.randperm(len(self.dataset1)).cpu().tolist()
        indices2 = torch.randperm(len(self.dataset2)) +len(indices1)
        indices2 = indices2.cpu().tolist()

        for i in range(self.total_size // self.batch_size):
            batch_size1 = min(self.batch_size // 2, len(indices1))
            if batch_size1 < (self.batch_size // 2):
              break
            batch_size2 = self.batch_size - batch_size1
            batch.extend([indices1.pop() for _ in range(batch_size1)])
            batch.extend([indices2.pop() for _ in range(batch_size2)])

            yield batch
            batch = []
            if len(indices1) == 0:
              break

    def __len__(self):
        return (self.total_size + self.batch_size - 1) // self.batch_size