# coding=utf-8
from mtp2_pd_prototypical_batch_sampler import PrototypicalBatchSampler
from prototypical_loss import prototypical_loss as loss_fn
from mtp2_pd_IITD_dataset import IITDelhiDataset
# from protonet import ProtoNet
# from protonet import ProtoNet1
from protonet import ProtoNet2

# from parser_util import get_parser
from mtp2_pd_parser_util import get_parser

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
    # exit()
    n_classes = len(np.unique(dataset.y))
    
    # print(f"[DEBUG] Mode: {mode}, Dataset Size: {len(dataset)}")
    # print(f"[DEBUG] Class distribution: {np.bincount(dataset.y)}")

    
    
    print(f"[INFO] Number of unique classes in dataset: {n_classes}")
    if n_classes < opt.classes_per_it_tr or n_classes < opt.classes_per_it_val:
        raise Exception(f"[ERROR] Not enough classes in the dataset! Required: " +
                        f"{opt.classes_per_it_tr} (train) / {opt.classes_per_it_val} (val), " +
                        f"but found only {n_classes}. Try reducing classes_per_it.")
    
    print(f"[SUCCESS] Dataset initialized successfully with {n_classes} classes.")
    return dataset


def init_sampler(opt, labels, mode, sector_to_classlist=None, sectors=None):
    if 'train' in mode:
        classes_per_it = opt.classes_per_it_tr
        num_samples = opt.num_support_tr + opt.num_query_tr
    else:
        classes_per_it = opt.classes_per_it_val
        num_samples = opt.num_support_val + opt.num_query_val
        
    # print(f"num_samples: {num_samples }")
    # exit(0)

    return PrototypicalBatchSampler(labels=labels,
                                    classes_per_it=classes_per_it,
                                    num_samples=num_samples,
                                    iterations=opt.iterations, 
                                    sector_to_classlist=sector_to_classlist,
                                    sectors=sectors,
                                    sectors_per_it=opt.sectors_per_it)

def init_dataloader(opt, mode):
    dataset = init_dataset(opt, mode)
    sampler = init_sampler(opt, dataset.y, mode, sector_to_classlist=dataset.sector_to_classlist,sectors=dataset.sectors)
    return torch.utils.data.DataLoader(dataset, batch_sampler=sampler), dataset.class_to_sector

def init_protonet(opt):
    device = 'cuda:0' if torch.cuda.is_available() and opt.cuda else 'cpu'
    # return ProtoNet().to(device)
    # return ProtoNet1().to(device)
    return ProtoNet2().to(device)
# def init_protonet(opt):
#     device = 'cuda:0' if torch.cuda.is_available() and opt.cuda else 'cpu'
#     onnx_model_path = "/old/home/nishkal/PD/DeepIris_Recog_Drive/ResNet50_Iris.onnx"
#     return ProtoNet(onnx_model_path).to(device)

def init_optim(opt, model):
    return torch.optim.Adam(params=model.parameters(), lr=opt.learning_rate)

def init_lr_scheduler(opt, optim):
    return torch.optim.lr_scheduler.StepLR(optimizer=optim, gamma=opt.lr_scheduler_gamma, step_size=opt.lr_scheduler_step)

def save_list_to_file(path, thelist):
    with open(path, 'w') as f:
        for item in thelist:
            f.write("%s\n" % item)


def train(opt, tr_dataloader, model, optim, lr_scheduler, val_dataloader=None, class_to_sector=None):
    device = 'cuda:0' if torch.cuda.is_available() and opt.cuda else 'cpu'
    
    if val_dataloader is None:
        best_state = None
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    best_acc = 0
    
    # best_acc, train_loss, train_acc, val_loss, val_acc = 0, [], [], [], []
    best_model_path = os.path.join(opt.experiment_root, 'best_model.pth')
    last_model_path = os.path.join(opt.experiment_root, 'last_model.pth')
    
    for epoch in range(opt.epochs):
        print(f'=== Epoch: {epoch} ===')

        tr_iter = iter(tr_dataloader)
        model.train()
        
        # print("tr_iter: ", tr_iter) #tr_iter:  <torch.utils.data.dataloader._SingleProcessDataLoaderIter object at 0x7d69100afd40>
        for batch in tqdm(tr_iter):
            optim.zero_grad()
            x, y = batch
            # (x, y), _ = batch
            x, y = x.to(device), y.to(device)
            model_output = model(x)
            loss, acc = loss_fn(model_output, target=y,
                                n_support=opt.num_support_tr, class_to_sector=class_to_sector)
            loss.backward()
            optim.step()
            train_loss.append(loss.item())
            train_acc.append(acc.item())
        avg_loss = np.mean(train_loss[-opt.iterations:])
        avg_acc = np.mean(train_acc[-opt.iterations:])
        print('Avg Train Loss: {}, Avg Train Acc: {}'.format(avg_loss, avg_acc))
        
        lr_scheduler.step()
        if val_dataloader is None:
            print("\n\nNone val_dataloader")
            continue
        
        val_iter = iter(val_dataloader)
        model.eval()
        for batch in val_iter:
            x, y = batch
            # (x, y), _ = batch
            x, y = x.to(device), y.to(device)
            model_output = model(x)
            loss, acc = loss_fn(model_output, target=y,
                                n_support=opt.num_support_val, class_to_sector=class_to_sector)
            val_loss.append(loss.item())
            val_acc.append(acc.item())
        avg_loss = np.mean(val_loss[-opt.iterations:])
        avg_acc = np.mean(val_acc[-opt.iterations:])
        postfix = ' (Best)' if avg_acc >= best_acc else ' (Best: {})'.format(
            best_acc)
        print('Avg Val Loss: {}, Avg Val Acc: {}{}'.format(
            avg_loss, avg_acc, postfix))
        if avg_acc >= best_acc:
            torch.save(model.state_dict(), best_model_path)
            best_acc = avg_acc
            best_state = model.state_dict()
    
    torch.save(model.state_dict(), last_model_path)
    
    for name in ['train_loss', 'train_acc', 'val_loss', 'val_acc']:
        save_list_to_file(os.path.join(opt.experiment_root,
                                       name + '.txt'), locals()[name])

    return best_state, best_acc, train_loss, train_acc, val_loss, val_acc


# def test(opt, test_dataloader, model):
#     device = 'cuda:0' if torch.cuda.is_available() and opt.cuda else 'cpu'
#     avg_acc = []
#     model.eval()
#     for x, y in test_dataloader:
#         x, y = x.to(device), y.to(device)
#         _, acc = loss_fn(model(x), target=y, n_support=opt.num_support_val)
#         avg_acc.append(acc.item())
    
#     print(f'Test Acc: {np.mean(avg_acc)}')


def test(opt, test_dataloader, model, class_to_sector=None):
    '''
    Test the model trained with the prototypical learning algorithm
    '''
    device = 'cuda:0' if torch.cuda.is_available() and opt.cuda else 'cpu'
    avg_acc = list()
    for epoch in range(10):
        test_iter = iter(test_dataloader)
        for batch in test_iter:
            x, y = batch
            # (x, y), _ = batch
            x, y = x.to(device), y.to(device)
            model_output = model(x)
            _, acc = loss_fn(model_output, target=y,
                             n_support=opt.num_support_val, class_to_sector=class_to_sector)
            avg_acc.append(acc.item())
    avg_acc = np.mean(avg_acc)
    print('Test Acc: {}'.format(avg_acc))

    return avg_acc


def eval(opt):
    '''
    Initialize everything and train
    '''
    options = get_parser().parse_args()

    if torch.cuda.is_available() and not options.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    init_seed(options)
    test_dataloader = init_dataset(options)[-1]
    model = init_protonet(options)
    model_path = os.path.join(opt.experiment_root, 'best_model.pth')
    model.load_state_dict(torch.load(model_path))

    test(opt=options,
         test_dataloader=test_dataloader,
         model=model)



def main():
    options = get_parser().parse_args()
    options.dataset_root = "/old/home/nishkal/datasets/iris_datasets/IITD/IITD V1/IITD Database" # for server code.
    
    os.makedirs(options.experiment_root, exist_ok=True)
    init_seed(options)
    
    tr_dataloader, class_to_sector_tr = init_dataloader(options, 'train')
    
    print("\n\nval_dataloader")
    val_dataloader, class_to_sector_val = init_dataloader(options, 'val')
    
    if val_dataloader is None:
        print("\n\nNone val_dataloader")
        # continue
    else:
        print("\nval_dataloader not empty")

    
    test_dataloader, class_to_sector_test = init_dataloader(options, 'test')
    
    model = init_protonet(options)
    optim = init_optim(options, model)
    lr_scheduler = init_lr_scheduler(options, optim)
    
    
    class_to_sector = {}
    # Helper to convert keys to int before updating
    def update_with_int_keys(target_dict, source_dict):
        for k, v in source_dict.items():
            target_dict[int(k)] = v  # convert key to int

    update_with_int_keys(class_to_sector, class_to_sector_tr)
    update_with_int_keys(class_to_sector, class_to_sector_val)
    update_with_int_keys(class_to_sector, class_to_sector_test)

    # Optional: print the combined mapping
    # print("Combined class_to_sector mapping:")
    # for cls, sector in class_to_sector.items():
    #     print(f"Class {cls} --> Sector {sector}")

    # print(f"\nTotal number of unique classes after merging: {len(class_to_sector)}")
    
    
    res = train(opt=options,
                tr_dataloader=tr_dataloader,
                val_dataloader=val_dataloader,
                model=model,
                optim=optim,
                lr_scheduler=lr_scheduler, 
                class_to_sector=class_to_sector)
    best_state, best_acc, train_loss, train_acc, val_loss, val_acc = res
    print('Testing with last model..')
    test(opt=options,
         test_dataloader=test_dataloader,
         model=model,
         class_to_sector=class_to_sector)

    model.load_state_dict(best_state)
    print('Testing with best model..')
    test(opt=options,
         test_dataloader=test_dataloader,
         model=model,
         class_to_sector=class_to_sector)

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
    # new change, did 60% train, 30% test, 10% val in my_split.py (11/3/25)
    # changed epochs to 100 in mtp2_pd_parser_util.py (11/3/25)
    main()