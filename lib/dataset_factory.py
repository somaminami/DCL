
# coding: utf-8

# In[1]:


import random
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


# # Initializer for worker

# In[15]:


class worker_initializer():
    def __init__(self, manualSeed):
        self.manualSeed = manualSeed

    def worker_init_fn(self, worker_id):
        random.seed(self.manualSeed+worker_id)
        return


# # CIFAR-10

# In[17]:


def CIFAR10(args):    
    manualSeed = args.manualSeed
    args = args.dataloader
    
    
    train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), 
                                          transforms.Pad(4, padding_mode="reflect"), 
                                          transforms.RandomCrop(32, padding=0), 
                                          transforms.ToTensor(),
                                          ])
    
    test_transform = transforms.Compose([transforms.ToTensor(),])
    
    data_path = args.data_path
    
    train_dataset = torchvision.datasets.CIFAR10(data_path, 
                                                 train=True, 
                                                 download=True, 
                                                 transform=train_transform)
    
    test_dataset = torchvision.datasets.CIFAR10(data_path, 
                                                train=False, 
                                                download=True, 
                                                transform=test_transform)
    
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=args.train_shuffle,
                              num_workers=args.workers,
                              pin_memory=False,
                              drop_last=args.train_drop_last,
                              worker_init_fn=worker_initializer(manualSeed).worker_init_fn)
    
    test_loader = DataLoader(test_dataset,
                             batch_size=args.batch_size,
                             shuffle=args.test_shuffle,
                             num_workers=args.workers,
                             pin_memory=False,
                             drop_last=args.test_drop_last,
                             worker_init_fn=worker_initializer(manualSeed).worker_init_fn)
    
    return train_loader, test_loader


# # CIFAR-100

# In[8]:


def CIFAR100(args):    
    manualSeed = args.manualSeed
    args = args.dataloader
    
    
    train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), 
                                          transforms.Pad(4, padding_mode="reflect"), 
                                          transforms.RandomCrop(32, padding=0), 
                                          transforms.ToTensor(),
                                          ])
    
    test_transform = transforms.Compose([transforms.ToTensor(),])
    
    data_path = args.data_path
    
    train_dataset = torchvision.datasets.CIFAR100(data_path, 
                                                  train=True, 
                                                  download=True, 
                                                  transform=train_transform)
    
    test_dataset = torchvision.datasets.CIFAR100(data_path, 
                                                 train=False, 
                                                 download=True, 
                                                 transform=test_transform)
    
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=args.train_shuffle,
                              num_workers=args.workers,
                              pin_memory=False,
                              drop_last=args.train_drop_last,
                              worker_init_fn=worker_initializer(manualSeed).worker_init_fn)
    
    test_loader = DataLoader(test_dataset,
                             batch_size=args.batch_size,
                             shuffle=args.test_shuffle,
                             num_workers=args.workers,
                             pin_memory=False,
                             drop_last=args.test_drop_last,
                             worker_init_fn=worker_initializer(manualSeed).worker_init_fn)
    
    return train_loader, test_loader

