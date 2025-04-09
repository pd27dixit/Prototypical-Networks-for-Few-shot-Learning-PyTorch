"""# coding=utf-8
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

    def __init__(self, labels, classes_per_it, num_samples, iterations, sector_to_classlist=None, sectors=None, sectors_per_it=None):
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
        # print("self.classes: ", self.classes)

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
            
        self.sector_to_classlist = sector_to_classlist
        self.sectors = sectors
        self.sectors_per_it = sectors_per_it

    def __iter__(self):
        '''
        yield a batch of indexes
        '''
        spc = self.sample_per_class
        cpi = self.classes_per_it

        for it in range(self.iterations):
            batch_size = spc * cpi
            batch = torch.LongTensor(batch_size)
            c_idxs = torch.randperm(len(self.classes))[:cpi]
            for i, c in enumerate(self.classes[c_idxs]):
                s = slice(i * spc, (i + 1) * spc)
                # FIXME when torch.argwhere will exists
                label_idx = torch.arange(len(self.classes)).long()[self.classes == c].item()
                sample_idxs = torch.randperm(self.numel_per_class[label_idx])[:spc]
                batch[s] = self.indexes[label_idx][sample_idxs]
            batch = batch[torch.randperm(len(batch))]
            yield batch
    """
    
import numpy as np
import torch
import os
import random



class PrototypicalBatchSampler(object):
    # OBJECTIVE:
    # It creates batches of samples for the 2-level prototype compuation

    def __init__(self, labels, classes_per_it, num_samples, iterations, sector_to_classlist=None, sectors=None, sectors_per_it=None):
        # OBJECTIVE
        # create 1) A mapping from class (person) → all its sample indices.
        # create 2) A mapping from class → sector (if sectors exist).
        
        """
        eg.
        labels = ['p1', 'p1', 'p2', 'p2', 'p3', 'p3', 'p4', 'p4', 'p5']
        
        sector_to_classlist = {
            's1': ['p1', 'p2'],
            's2': ['p3', 'p4', 'p5']
        }
        """
        
        super(PrototypicalBatchSampler, self).__init__()
        self.labels = labels #list in which each elem is class label
        self.classes_per_it = classes_per_it #number of different classes to sample per episode
        self.sample_per_class = num_samples
        self.iterations = iterations #number of episodes per epoch.

        self.classes, self.counts = np.unique(self.labels, return_counts=True)
        self.classes = torch.LongTensor(self.classes)

        self.idxs = range(len(self.labels))
        self.indexes = np.empty((len(self.classes), max(self.counts)), dtype=int) * np.nan
        self.indexes = torch.Tensor(self.indexes)
        self.numel_per_class = torch.zeros_like(self.classes)
        for idx, label in enumerate(self.labels):
            label_idx = np.argwhere(self.classes == label).item()
            self.indexes[label_idx, np.where(np.isnan(self.indexes[label_idx]))[0][0]] = idx #Create a matrix indexes with rows = [p1, p2, ..., p5] and columns = sample indices.
            self.numel_per_class[label_idx] += 1
            
        self.sector_to_classlist = sector_to_classlist #dictionary mapping each sector to the list of person-classes in it.
        self.sectors = sectors # list of sector names/ids.
        self.sectors_per_it = sectors_per_it #how many sectors to sample in each episode.
        
        # Build the reverse mapping from class → sector (class_to_sector).
        self.class_to_sector = {}
        if self.sector_to_classlist:
            for sector, class_list in self.sector_to_classlist.items():
                for cls in class_list:
                    self.class_to_sector[cls] = sector

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
        
        # print("self.sector_to_classlist: ", self.sector_to_classlist)

    def __iter__(self): 
        
        spc = self.sample_per_class
        cpi = self.classes_per_it

        for it in range(self.iterations): #for each episode
            batch_size = spc * cpi
            batch = torch.LongTensor(batch_size)

            selected_classes = []

            # Step 1: Select sectors and person-classes (depending on whether sectors are provided)
            if self.sector_to_classlist and self.sectors and self.sectors_per_it:
                selected_sectors = random.sample(self.sectors, self.sectors_per_it)
                eligible_classes = []
                for sec in selected_sectors:
                    if sec in self.sector_to_classlist:
                        eligible_classes.extend(self.sector_to_classlist[sec])
                eligible_classes = list(set(eligible_classes))
                
                """
                eg.
                
                sectors_per_it = 1, and random pick sector 's2'
                Then eligible_classes = ['p3', 'p4', 'p5']
                """
                
                
                # Then we try to sample classes_per_it persons from these, if not enough, fall back to random classes from the full class list.

                if len(eligible_classes) >= cpi:
                    selected_classes = random.sample(eligible_classes, cpi)
                elif len(self.classes) >= cpi:
                    # Fallback: use all available classes
                    selected_classes = random.sample(self.classes.tolist(), cpi)
                else:
                    # Final fallback: skip iteration if not even full classes exist
                    continue
            else:
                if len(self.classes) >= cpi:
                    c_idxs = torch.randperm(len(self.classes))[:cpi]
                    selected_classes = self.classes[c_idxs].tolist()
                else:
                    continue  # skip if not enough classes at all
            
            
            
            # Step 2: For each selected class (person), sample data points
            for i, c in enumerate(selected_classes): #c: each person class
                s = slice(i * spc, (i + 1) * spc)
                label_idx = torch.arange(len(self.classes)).long()[self.classes == c].item() #Find the row in self.indexes for c.
                sample_idxs = torch.randperm(self.numel_per_class[label_idx])[:spc]  #Randomly pick spc = num_samples indices (support + query).
                batch[s] = self.indexes[label_idx][sample_idxs] #Add them to the batch.


            batch = batch[torch.randperm(len(batch))]  #The batch (of size classes_per_it × num_samples) is shuffled and returned.
            yield batch

    

    def __len__(self):
        '''
        returns the number of iterations (episodes) per epoch
        '''
        return self.iterations
    
    
    
# Batch Sampler: ensures only classes from selected sectors are sampled

"""
in training loop, batches will look something like this

Episode:
  Sectors selected: [s2]
  Classes selected: [p3, p5]
  Samples: [
    x3_1, x3_2, ..., x5_1, x5_2, ...
  ]
  
forward pass, you would:

    1. Create prototypes per class (p3 and p5).
    2. Create prototype per sector (s2) as the average of p3 and p5 prototypes.
    3. For a query sample:
        First match it to all sector prototypes (s2, s1...).
        Then match only to persons in the closest sector.
"""