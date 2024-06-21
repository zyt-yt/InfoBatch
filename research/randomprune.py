import random
import numpy as np
from torch.utils.data import Dataset, Sampler

class RandomPrune(Dataset):
    def __init__(self, dataset, prune_ratio = 0.5, num_epoch = None):
        self.dataset = dataset
        self.prune_ratio = prune_ratio
        self.num_epoch = num_epoch
        self.current_epoch = 0
        self.indices = list(range(len(dataset)))
        self.pruned_indices = self.indices
        self.save_num = 0

    def __len__(self):
        return len(self.pruned_indices)

    def __getitem__(self, idx):
        data, target = self.dataset[idx]
        return data, target

    def prune(self):
        pruned_num = int(len(self.indices)* self.prune_ratio)
        self.save_num += pruned_num
        print('Cut {} samples for next iteration'.format(pruned_num))
        self.pruned_indices = random.sample(self.indices, len(self.indices) - pruned_num)

    def reset(self):
        self.pruned_indices = self.indices
    
    def total_save(self):
        return self.save_num
    
    def on_epoch_end(self):
        if self.current_epoch < self.num_epoch:
            self.prune()
            self.current_epoch += 1
        else:
            self.reset()


class RandomPruneSampler(Sampler):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        self.dataset.on_epoch_end()
        #print(self.dataset.pruned_indices)
        return iter(self.dataset.pruned_indices)