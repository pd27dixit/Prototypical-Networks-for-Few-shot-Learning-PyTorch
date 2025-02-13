# coding=utf-8
from prototypical_batch_sampler import PrototypicalBatchSampler
from prototypical_loss import prototypical_loss as loss_fn
from mtp2_pd_IITD_dataset import IITDelhiDataset
from protonet import ProtoNet
from parser_util import get_parser
from tqdm import tqdm
import numpy as np
import torch
import os

def init_seed(opt):
    torch.cuda.cudnn_enabled = False
    np.random.seed(opt.manual_seed)
    torch.manual_seed(opt.manual_seed)
    torch.cuda.manual_seed(opt.manual_seed)


# def init_dataset(opt, mode):
#     dataset = OmniglotDataset(mode=mode, root=opt.dataset_root)
#     n_classes = len(np.unique(dataset.y))
#     if n_classes < opt.classes_per_it_tr or n_classes < opt.classes_per_it_val:
#         raise(Exception('There are not enough classes in the dataset in order ' +
#                         'to satisfy the chosen classes_per_it. Decrease the ' +
#                         'classes_per_it_{tr/val} option and try again.'))
#     return dataset

def init_dataset(opt, mode):
    dataset = IITDelhiDataset(mode=mode, root=opt.dataset_root)
    n_classes = len(np.unique(dataset.labels))
    if n_classes < opt.classes_per_it_tr or n_classes < opt.classes_per_it_val:
        raise Exception("Not enough classes in dataset. Adjust classes_per_it_tr/val.")
    return dataset

def init_sampler(opt, labels, mode):
    classes_per_it = opt.classes_per_it_tr if 'train' in mode else opt.classes_per_it_val
    num_samples = opt.num_support_tr + opt.num_query_tr if 'train' in mode else opt.num_support_val + opt.num_query_val
    return PrototypicalBatchSampler(labels=labels, classes_per_it=classes_per_it, num_samples=num_samples, iterations=opt.iterations)

def init_dataloader(opt, mode):
    dataset = init_dataset(opt, mode)
    sampler = init_sampler(opt, dataset.labels, mode)
    return torch.utils.data.DataLoader(dataset, batch_sampler=sampler)

def init_protonet(opt):
    device = 'cuda:0' if torch.cuda.is_available() and opt.cuda else 'cpu'
    return ProtoNet().to(device)

def init_optim(opt, model):
    return torch.optim.Adam(params=model.parameters(), lr=opt.learning_rate)

def init_lr_scheduler(opt, optim):
    return torch.optim.lr_scheduler.StepLR(optimizer=optim, gamma=opt.lr_scheduler_gamma, step_size=opt.lr_scheduler_step)

def train(opt, tr_dataloader, model, optim, lr_scheduler, val_dataloader=None):
    device = 'cuda:0' if torch.cuda.is_available() and opt.cuda else 'cpu'
    best_acc, train_loss, train_acc, val_loss, val_acc = 0, [], [], [], []
    best_model_path = os.path.join(opt.experiment_root, 'best_model.pth')
    last_model_path = os.path.join(opt.experiment_root, 'last_model.pth')
    
    for epoch in range(opt.epochs):
        print(f'=== Epoch: {epoch} ===')
        model.train()
        for x, y in tqdm(tr_dataloader):
            x, y = x.to(device), y.to(device)
            optim.zero_grad()
            loss, acc = loss_fn(model(x), target=y, n_support=opt.num_support_tr)
            loss.backward()
            optim.step()
            train_loss.append(loss.item())
            train_acc.append(acc.item())
        
        avg_train_acc = np.mean(train_acc[-opt.iterations:])
        print(f'Avg Train Acc: {avg_train_acc}')
        lr_scheduler.step()
        
        if val_dataloader:
            model.eval()
            for x, y in val_dataloader:
                x, y = x.to(device), y.to(device)
                loss, acc = loss_fn(model(x), target=y, n_support=opt.num_support_val)
                val_loss.append(loss.item())
                val_acc.append(acc.item())
            
            avg_val_acc = np.mean(val_acc[-opt.iterations:])
            print(f'Avg Val Acc: {avg_val_acc}')
            if avg_val_acc > best_acc:
                torch.save(model.state_dict(), best_model_path)
                best_acc = avg_val_acc
    
    torch.save(model.state_dict(), last_model_path)

def test(opt, test_dataloader, model):
    device = 'cuda:0' if torch.cuda.is_available() and opt.cuda else 'cpu'
    avg_acc = []
    model.eval()
    for x, y in test_dataloader:
        x, y = x.to(device), y.to(device)
        _, acc = loss_fn(model(x), target=y, n_support=opt.num_support_val)
        avg_acc.append(acc.item())
    
    print(f'Test Acc: {np.mean(avg_acc)}')

def main():
    opt = get_parser().parse_args()
    opt.dataset_root = "/content/drive/MyDrive/MTP2/IRIS_DATABASES/IITD/IITD V1/IITD Database"
    
    os.makedirs(opt.experiment_root, exist_ok=True)
    init_seed(opt)
    
    tr_dataloader = init_dataloader(opt, 'train')
    val_dataloader = init_dataloader(opt, 'val')
    test_dataloader = init_dataloader(opt, 'test')
    
    model = init_protonet(opt)
    optim = init_optim(opt, model)
    lr_scheduler = init_lr_scheduler(opt, optim)
    
    train(opt, tr_dataloader, model, optim, lr_scheduler, val_dataloader)
    print('Testing with best model..')
    model.load_state_dict(torch.load(os.path.join(opt.experiment_root, 'best_model.pth')))
    test(opt, test_dataloader, model)

    # optim = init_optim(options, model)
    # lr_scheduler = init_lr_scheduler(options, optim)

    # print('Training on train+val set..')
    # train(opt=options,
    #       tr_dataloader=trainval_dataloader,
    #       val_dataloader=None,
    #       model=model,
    #       optim=optim,
    #       lr_scheduler=lr_scheduler)

    # print('Testing final model..')
    # test(opt=options,
    #      test_dataloader=test_dataloader,
    #      model=model)

    
if __name__ == '__main__':
    main()
