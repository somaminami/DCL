
# coding: utf-8

# # Import library

# In[1]:


import os
import random
import easydict
import copy
import argparse
import torch


# In[2]:


from lib import dataset_factory
from lib import models as model_fuctory
from lib import loss_func as loss_func
from lib import trainer as trainer_module
from lib import utils


# In[3]:


parser = argparse.ArgumentParser()

parser.add_argument('--target_model', type=str, default="ResNet32")
parser.add_argument('--dataset', type=str, choices=["CIFAR10", "CIFAR100"], default="CIFAR100")
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--save_dir', type=str, default="./pre-train/ResNet32/")

try:
    args = parser.parse_args()
except SystemExit:
    args = parser.parse_args(args=[
        "--target_model", "ResNet32",
        "--dataset", "CIFAR100",
        "--gpu_id", "0",
        "--save_dir", "./pre-train/ResNet32/",
    ])


# In[4]:


get_ipython().magic('env CUDA_DEVICE_ORDER=PCI_BUS_ID')
get_ipython().magic('env CUDA_VISIBLE_DEVICES=$args.gpu_id')


# # Set config

# In[5]:


manualSeed = 0

if args.dataset == "CIFAR10":
    DATA_PATH = "./dataset/CIFAR10/"
    NUM_CLASS = 10
    SCHEDULE = [60,120,180]
    EPOCHS = 200    
elif args.dataset == "CIFAR100":
    DATA_PATH = "./dataset/CIFAR100/"
    NUM_CLASS = 100
    SCHEDULE = [60,120,180]
    EPOCHS = 200
    
optim_setting = {
    "name": "SGD",
    "args":
    {
        "lr": 0.1,
        "momentum": 0.9,
        "weight_decay": 0.0005,
        "nesterov": True,
    },
    "schedule": SCHEDULE,
    "gammas": [0.1,0.1,0.1],
}

args_factory = easydict.EasyDict({
    "models": {
        "ResNet32":
        {
            "name": "resnet32",
            "args":
            {
                "num_classes": NUM_CLASS,
            },
        },
        "ResNet110":
        {
            "name": "resnet110",
            "args":
            {
                "num_classes": NUM_CLASS,
            },
        },
        "WRN28_2":
        {
            "name": "WideResNet",
            "args":
            {
                "depth": 28,
                "num_classes": NUM_CLASS,
                "widen_factor": 2,
                "dropRate": 0.0
            },
        },
    },
    "losses":
    {
        "IndepLoss":
        {
            "name": "IndependentLoss",
            "args": 
            {
                "loss_weight": 1,
                "gate": 
                {
                    "name": "ThroughGate",
                    "args": {},
                },
            },
        },
    }
})

model = args_factory.models[args.target_model]


config = easydict.EasyDict(
    {
        #------------------------------Others--------------------------------        
        "doc": "",
        "manualSeed": manualSeed,
        #------------------------------Dataloader--------------------------------
        "dataloader": 
        {
            "name": args.dataset,
            "data_path": DATA_PATH,
            "num_class": NUM_CLASS,
            "batch_size": 128,
            "workers": 10,
            "train_shuffle": True,
            "train_drop_last": True, 
            "test_shuffle": True,
            "test_drop_last": False, 
        },
        #------------------------------Trainer--------------------------------
        "trainer": 
        {
            "name": "ClassificationTrainer",
            "start_epoch": 1,
            "epochs": EPOCHS,
            "saving_interval": EPOCHS,
            "base_dir": "./",
        },
        #--------------------------Models & Optimizer-------------------------
        "models": 
        [           
            #----------------Model------------------
            {
                "name": model.name,
                "args": model.args,
                "load_weight":
                {
                    "path": None,
                    "model_id": 0,
                },
                "optim": optim_setting,
            } 
        ],
        #-----------------------------Loss_func-------------------------------
        #
        #    source node -> target node
        #    [
        #        [1->1, 2->1, 3->1],
        #        [1->2, 2->2, 3->2],
        #        [1->3, 2->3, 3->3],
        #    ]
        #
        "losses":        
        [
            [
                args_factory.losses["IndepLoss"]
            ]
        ],
        #------------------------------GPU-------------------------------- 
        "gpu":
        {
            "use_cuda": True,
            "ngpu": 1,
            "id": 0,
        },
    })

config = copy.deepcopy(config)


# # Create object

# In[ ]:


def create_object(config):
    # set seed value
    config.manualSeed = random.randint(1,10000)
    random.seed(config.manualSeed)
    torch.manual_seed(config.manualSeed)
    torch.cuda.manual_seed_all(config.manualSeed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # create dataset
    train_loader, test_loader = getattr(dataset_factory, config.dataloader.name)(config)    
    
    # define model & loss func & optimizer    
    nets = []
    criterions = []
    optimizers = []
    for model_args in config.models:
        # define model
        net = getattr(model_fuctory, model_args.name)(**model_args.args)
        net = net.cuda(config.gpu.id)
        
        # load weight
        if model_args.load_weight.path is not None:
            utils.load_model(net, model_args.load_weight.path, model_args.load_weight.model_id)
        
        nets += [net]
        
        # define loss function        
        criterions = []
        for row in config.losses:
            r = []
            for loss in row:
                criterion = getattr(loss_func, loss.name)(loss.args)
                criterion = criterion.cuda(config.gpu.id)
                r += [criterion]
            criterions += [loss_func.TotalLoss(r)]
        
        # define optimizer
        optimizer = getattr(torch.optim, model_args.optim.name)
        optimizers += [optimizer(net.parameters(), **model_args.optim.args)]
    
    # Trainer
    trainer = getattr(trainer_module, config.trainer.name)(config)

    # Logger
    logs = utils.LogManagers(len(config.models), len(train_loader.dataset),
                                config.trainer.epochs, config.dataloader.batch_size)

    return trainer, nets, criterions, optimizers, train_loader, test_loader, logs


# # Train

# In[ ]:


config.trainer.base_dir = args.save_dir
utils.make_dirs(config.trainer.base_dir+"log")
utils.make_dirs(config.trainer.base_dir+"checkpoint")

utils.save_json(config, config.trainer.base_dir+r"log/config.json")

trainer, nets, criterions, optimizers, train_loader, test_loader, logs = create_object(config)    

trainer.train(nets, criterions, optimizers, train_loader, test_loader, logs)

