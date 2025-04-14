# coding=utf-8
import torch
from torch.nn import functional as F
from torch.nn.modules import Module


class PrototypicalLoss(Module):
    '''
    Loss class deriving from Module for the prototypical loss function defined below
    '''
    def __init__(self, n_support, class_to_sector):
        super(PrototypicalLoss, self).__init__()
        self.n_support = n_support  # how many support examples per class
        self.class_to_sector = class_to_sector # maps person_id to sector_id

    def forward(self, input, target):
        return prototypical_loss(input, target, self.n_support, self.class_to_sector)


def euclidean_dist(x, y):
    '''
    Compute euclidean distance between two tensors
    '''
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


def prototypical_loss(input, target, n_support, class_to_sector):

    target_cpu = target.to('cpu')
    input_cpu = input.to('cpu')
    # input_cpu = F.normalize(input.to('cpu'), p=2, dim=1)


    
    ### 1. Get Class-wise Support Indices
    def supp_idxs(c):
        # FIXME when torch will support where as np
        return target_cpu.eq(c).nonzero()[:n_support].squeeze(1)   ### fetches the first n_support indices for each class (person).

    # FIXME when torch.unique will be available on cuda too
    classes = torch.unique(target_cpu)
    # classes += 1
    # print("classes (1-indexed):", classes)
    # print("classes: ", classes)
    
    # dataset_classes = torch.tensor([idx_to_class[c.item()] for c in classes])
    # print("Original dataset class labels:", dataset_classes)
    
    n_classes = len(classes)
    # print("classes: ", classes) #classes:  tensor([ 0,  2,  3,  4,  5,  6,  7,  8,  9, 10, 12, 14, 15, 16, 17, 18, 19, 20,
        # 21, 22, 23, 25, 26, 27, 28, 29, 30, 32, 33, 34, 35, 36, 37, 38, 39, 40,
        # 41, 42, 43, 44])
        
    # print("n_classes: ", n_classes) # n_classes: 20
    
    # FIXME when torch will support where as np
    # assuming n_query, n_target constants
    n_query = target_cpu.eq(classes[0].item()).sum().item() - n_support

    support_idxs = list(map(supp_idxs, classes))

    """prototypes = torch.stack([input_cpu[idx_list].mean(0) for idx_list in support_idxs])
    # FIXME when torch will support where as np
    query_idxs = torch.stack(list(map(lambda c: target_cpu.eq(c).nonzero()[n_support:], classes))).view(-1)

    query_samples = input.to('cpu')[query_idxs]
    dists = euclidean_dist(query_samples, prototypes)

    log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1)

    target_inds = torch.arange(0, n_classes)
    target_inds = target_inds.view(n_classes, 1, 1)
    target_inds = target_inds.expand(n_classes, n_query, 1).long()

    loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
    _, y_hat = log_p_y.max(2)
    acc_val = y_hat.eq(target_inds.squeeze(2)).float().mean()

    return loss_val,  acc_val"""
    
    # print("Hi1")
    
    
    ###  2. Compute Person Prototypes
    prototypes = torch.stack([input_cpu[idx_list].mean(0) for idx_list in support_idxs])  ### Each person prototype is the average of their support embeddings.
    ### prototypes = [proto_0, proto_1, ..., proto_5].
    
    # prototypes = torch.stack([input_cpu[idx_list].mean(0) for idx_list in support_idxs]) 
    # prototypes = F.normalize(prototypes, p=2, dim=1)
    
    
    
    class_to_idx = {c.item(): i for i, c in enumerate(classes)}
    
    
    # # Build sector_to_classlist from class_to_sector
    # sector_to_classlist = {}
    # for c in classes:
    #     sector = class_to_sector[c.item()]
    #     if sector not in sector_to_classlist:
    #         sector_to_classlist[sector] = []
    #     sector_to_classlist[sector].append(c.item())
    
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    # print("idx_to_class: ", idx_to_class)
    # print("class_to_idx: ", class_to_idx)


    ### 3. Compute Sector Prototypes
    sector_to_classlist = {}
    for c in classes:
        class_name = c.item()  # Already actual class name, e.g., 11, 19, 20...
        # print("class_name: ", class_name)
        sector = class_to_sector[class_name+1]
        if sector not in sector_to_classlist:
            sector_to_classlist[sector] = []
        sector_to_classlist[sector].append(class_name)
        
    """
    eg. 
    sector_to_classlist = {
        A: [0, 1, 2],
        B: [3, 4, 5]
    }
    """
        
    # Compute sector-level prototypes (mean of prototypes in that sector)
    sector_prototypes = {}
    for sector_id, class_ids in sector_to_classlist.items():
        idxs = [class_to_idx[cid] for cid in class_ids]
        sector_vecs = torch.stack([prototypes[i] for i in idxs])
        sector_prototypes[sector_id] = sector_vecs.mean(dim=0)  
        """ 
        eg. sector_A_proto = mean(proto_0, proto_1, proto_2)
        """

    ### 4. Get query samples
    query_idxs = torch.stack(list(map(lambda c: target_cpu.eq(c).nonzero()[n_support:], classes))).view(-1)
    # print("A!")
    query_samples = input.to('cpu')[query_idxs] ###  embeddings of query examples (those not used in support set).

    # For each query, compute sector-level distance, then within that sector, person-level distance
    all_preds = []
    all_targets = []
    
    
    ### 5. Classify Each Query
    for i, q in enumerate(query_samples):
        #Closest sector
        closest_sector = min(sector_prototypes, key=lambda sid: torch.dist(q, sector_prototypes[sid])) ###  query q is first compared to each sector prototype, using Euclidean distance.

        ### 6. Within Closest Sector, Find Closest Person
        class_ids_in_sector = sector_to_classlist[closest_sector]
        proto_idxs = [class_to_idx[cid] for cid in class_ids_in_sector]
        sector_protos = torch.stack([prototypes[j] for j in proto_idxs])

        dists = torch.stack([torch.dist(q, p) for p in sector_protos])
        pred_idx = torch.argmin(dists).item()
        pred_class = class_ids_in_sector[pred_idx]   ### Among the persons within the closest sector, find the closest person prototype → this becomes the predicted class.
        
        """
        If q was closest to sector A,

        Then it is compared only to proto_0, proto_1, proto_2,
        Closest one (say proto_1) → predicted class = 1
        
        """

        true_class = target_cpu[query_idxs[i]].item()
        all_preds.append(pred_class)
        all_targets.append(true_class)

    all_preds = torch.tensor(all_preds)
    all_targets = torch.tensor(all_targets)

    # loss_val = F.cross_entropy(
    #     -torch.stack([
    #         torch.tensor([torch.dist(query_samples[i], prototypes[class_to_idx[cid]]) for cid in classes])
    #         for i in range(query_samples.shape[0])
    #     ]),
    #     torch.tensor([class_to_idx[c] for c in all_targets])
    # )
    
    # distances will be computed using autograd-aware ops
    distances = []
    for i in range(query_samples.shape[0]):
        dists = torch.stack([
            torch.norm(query_samples[i] - prototypes[class_to_idx[int(cid)]])
            for cid in classes
        ])
        distances.append(dists)


    ### 7. Compute Cross Entropy Loss (with Autograd-safe distances)
    
    ### network still sees gradients from all class-wise distances during training 
    logits = -torch.stack(distances)  # shape: [num_queries, num_classes]
    targets = torch.tensor([class_to_idx[int(c)] for c in all_targets], dtype=torch.long, device=logits.device)

    loss_val = F.cross_entropy(logits, targets)

    ### 8. Compute Accuracy
    acc_val = all_preds.eq(all_targets).float().mean()
    
    # print("Hi2")

    return loss_val, acc_val


"""
eg. 

1)Query q = embedding of person 4’s test image

2) Check distance(q, sector_A), distance(q, sector_B)

    - Lets assume,  it’s closer to sector B


4) Then compare q with person_3, person_4, person_5


    - Say it's closest to person_4 → predicted = 4

5) Compute loss using all class distances

6) Update model using this loss

"""
