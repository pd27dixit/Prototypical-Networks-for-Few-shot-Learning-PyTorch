# coding=utf-8
import numpy as np
import torch
import os


class PrototypicalBatchSampler(object):
    '''
    PrototypicalBatchSampler: yield a batch of indexes at each iteration.
    Indexes are calculated by keeping in account 'classes_per_it' and 'num_samples',
    In fact at every iteration the batch indexes will refer to  'num_support' + 'num_query' samples
    for 'classes_per_it' random classes.

    __len__ returns the number of episodes per epoch (same as 'self.iterations').
    '''

    def __init__(self, labels, classes_per_it, num_samples, iterations):
        '''
        Initialize the PrototypicalBatchSampler object
        Args:
        - labels: an iterable containing all the labels for the current dataset
        samples indexes will be infered from this iterable.
        - classes_per_it: number of random classes for each iteration
        - num_samples: number of samples for each iteration for each class (support + query)
        - iterations: number of iterations (episodes) per epoch
        '''
        super(PrototypicalBatchSampler, self).__init__()
        self.labels = labels
        self.classes_per_it = classes_per_it
        self.sample_per_class = num_samples
        self.iterations = iterations

        self.classes, self.counts = np.unique(self.labels, return_counts=True)
        self.classes = torch.LongTensor(self.classes)

        # create a matrix, indexes, of dim: classes X max(elements per class)
        # fill it with nans
        # for every class c, fill the relative row with the indices samples belonging to c
        # in numel_per_class we store the number of samples for each class/row
        self.idxs = range(len(self.labels))
        self.indexes = np.empty((len(self.classes), max(self.counts)), dtype=int) * np.nan
        self.indexes = torch.Tensor(self.indexes)
        self.numel_per_class = torch.zeros_like(self.classes)
        for idx, label in enumerate(self.labels):
            label_idx = np.argwhere(self.classes == label).item()
            self.indexes[label_idx, np.where(np.isnan(self.indexes[label_idx]))[0][0]] = idx
            self.numel_per_class[label_idx] += 1

    # def __iter__(self):
    #     '''
    #     yield a batch of indexes
    #     '''
    #     spc = self.sample_per_class
    #     cpi = self.classes_per_it

    #     for it in range(self.iterations):
    #         batch_size = spc * cpi
    #         batch = torch.LongTensor(batch_size)
    #         c_idxs = torch.randperm(len(self.classes))[:cpi]
    #         for i, c in enumerate(self.classes[c_idxs]):
    #             s = slice(i * spc, (i + 1) * spc)
    #             # FIXME when torch.argwhere will exists
    #             label_idx = torch.arange(len(self.classes)).long()[self.classes == c].item()
    #             sample_idxs = torch.randperm(self.numel_per_class[label_idx])[:spc]
    #             batch[s] = self.indexes[label_idx][sample_idxs]
    #         batch = batch[torch.randperm(len(batch))]
    #         yield batch
    
    def __iter__(self):
        '''
        Yield a batch of indexes
        '''
        spc = self.sample_per_class
        cpi = self.classes_per_it
        
        # print(f"spc: {spc}")
        # print(f"cpi: {cpi}")
        
        # exit()

        for it in range(self.iterations):
            batch_size = spc * cpi
            batch = torch.LongTensor(batch_size)
            c_idxs = torch.randperm(len(self.classes))[:cpi]
            batch_offset = 0  # Keeps track of batch index

            # for i, c in enumerate(self.classes[c_idxs]):
            #     s = slice(batch_offset, batch_offset + spc)
            #     label_idx = (self.classes == c).nonzero(as_tuple=True)[0].item()
                
            #     available_samples = self.numel_per_class[label_idx]
            #     if available_samples < spc:
            #         print(f"Warning: Class {c} has only {available_samples} samples, expected {spc}")
            #         sample_idxs = torch.randperm(available_samples)[:available_samples]  # Take all available
            #         pad_size = spc - available_samples
            #         sample_idxs = torch.cat((sample_idxs, sample_idxs[:pad_size]))  # Duplicate some if needed
            #     else:
            #         sample_idxs = torch.randperm(available_samples)[:spc]

            #     batch[s] = self.indexes[label_idx][sample_idxs]
            #     batch_offset += spc  # Move to the next slice
            
            '''
            for c in self.classes[c_idxs]:
                label_idx = (self.classes == c).nonzero(as_tuple=True)[0].item()
                available_samples = self.numel_per_class[label_idx].item()

                if available_samples < spc:
                    # Warn and sample with replacement
                    # print(f"Warning: Class {c} has only {available_samples} samples, expected {spc}. Sampling with replacement.")
                    sample_idxs = torch.randint(0, available_samples, (spc,))  # Sample with replacement
                else:
                    sample_idxs = torch.randperm(available_samples)[:spc]

                batch[batch_offset: batch_offset + spc] = self.indexes[label_idx][sample_idxs]
                batch_offset += spc

            yield batch[torch.randperm(batch_size)]  # Shuffle batch
            '''
            for i, c in enumerate(self.classes[c_idxs]):
                s = slice(i * spc, (i + 1) * spc)
                # FIXME when torch.argwhere will exists
                label_idx = torch.arange(len(self.classes)).long()[self.classes == c].item()
                sample_idxs = torch.randperm(self.numel_per_class[label_idx])[:spc]
                batch[s] = self.indexes[label_idx][sample_idxs]
            
            batch = batch[torch.randperm(len(batch))]  # Shuffle the batch
            yield batch


    def __len__(self):
        '''
        returns the number of iterations (episodes) per epoch
        '''
        return self.iterations