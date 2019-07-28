
# coding: utf-8

# In[1]:


import os
import random
import easydict
import copy
import argparse
import torch
import optuna


# In[2]:


from lib import dataset_factory
from lib import models as model_fuctory
from lib import loss_func as loss_func
from lib import trainer as trainer_module
from lib import utils


# In[3]:


parser = argparse.ArgumentParser()

parser.add_argument('--num_nodes', type=int, default=3)
parser.add_argument('--target_model', type=str, default="ResNet32")
parser.add_argument('--dataset', type=str, choices=["CIFAR10", "CIFAR100"], default="CIFAR100")
parser.add_argument('--gpu_id', type=int, default=0)
parser.add_argument('--num_trial', type=int, default=1500)
parser.add_argument('--optuna_dir', type=str, default="./result/")


try:
    args = parser.parse_args()
except SystemExit:
    args = parser.parse_args(args=[
        "--num_nodes", "3",
        "--target_model", "ResNet32",
        "--dataset", "CIFAR100",
        "--gpu_id", "0",
        "--num_trial", "1500",
        "--optuna_dir", "./result/",
    ])

args.num_ens = 0


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
            "batch_size": 64,
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
            "saving_interval": 1000,
            "base_dir": "./",
        },
        #--------------------------Models & Optimizer-------------------------
        "models": 
        [           
            #----------------Model------------------
            {
                "name": "resnet32",
                "args":
                {
                    "num_classes": NUM_CLASS,
                },
                "load_weight":
                {
                    "path": None,
                    "model_id": 0,
                },
                "optim": optim_setting,
            } 
            for _ in range(args.num_nodes)
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
                None
                for _ in range(args.num_nodes)
            ]
            for _ in range(args.num_nodes)
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

# In[6]:


def create_object(config):
    # set seed value
    config.manualSeed = random.randint(1,10000)
    random.seed(config.manualSeed)
    torch.manual_seed(config.manualSeed)
    torch.cuda.manual_seed_all(config.manualSeed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # load dataset
    train_loader, test_loader = getattr(dataset_factory, config.dataloader.name)(config)    
    
    # model & loss func & optimizer    
    nets = []
    criterions = []
    optimizers = []
    for model_args in config.models:
        # model
        net = getattr(model_fuctory, model_args.name)(**model_args.args)
        net = net.cuda(config.gpu.id)
        
        # load weight
        if model_args.load_weight.path is not None:
            utils.load_model(net, model_args.load_weight.path, model_args.load_weight.model_id)
        
        nets += [net]
        
        # loss function        
        criterions = []
        for row in config.losses:
            r = []
            for loss in row:
                criterion = getattr(loss_func, loss.name)(loss.args)
                criterion = criterion.cuda(config.gpu.id)
                r += [criterion]
            criterions += [loss_func.TotalLoss(r)]
        
        # optimizer
        optimizer = getattr(torch.optim, model_args.optim.name)
        optimizers += [optimizer(net.parameters(), **model_args.optim.args)]
    
    # trainer
    trainer = getattr(trainer_module, config.trainer.name)(config)

    # logger
    logs = utils.LogManagers(len(config.models), len(train_loader.dataset),
                                config.trainer.epochs, config.dataloader.batch_size)

    return trainer, nets, criterions, optimizers, train_loader, test_loader, logs


# # Optuna

# ## Args Factory

# In[7]:


ckpt_path = "result/0000/checkpoint/checkpoint_epoch_200.pkl"

args_factory = easydict.EasyDict({
    "models": {
        "ResNet32":
        {
            "name": "resnet32",
            "args":
            {
                "num_classes": NUM_CLASS,
            },
            "load_weight":
            {
                "path": f"./pre-train/ResNet32/{ckpt_path}",
            },
        },
        "ResNet110":
        {
            "name": "resnet110",
            "args":
            {
                "num_classes": NUM_CLASS,
            },
            "load_weight":
            {
                "path": f"./pre-train/ResNet110/{ckpt_path}",
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
            "load_weight":
            {
                "path": f"./pre-train/WRN28_2/{ckpt_path}",
            },
        },
        "Ensemble":
        {
            "name": "Ensemble",
            "args":
            {
                "source_list": list(range(1, args.num_ens+1)),
                "detach_list": list(range(1, args.num_ens+1)),
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
                "gate": {},
            },
        },
        "KLLoss":
        {
            "name": "KLLoss",
            "args":
            {
                "T": 1,
                "loss_weight": 1,
                "gate": {},
            },        
        },
    },
    "gates":
    {
        "CutoffGate":
        {
            "name": "CutoffGate",
            "args": {},
        }, 
        "ThroughGate":
        {
            "name": "ThroughGate",
            "args": {},
        }, 
        "CorrectGate": 
        {
            "name": "CorrectGate",
            "args": {},    
        },
        "LinearGate":
        {
            "name": "LinearGate",
            "args": {},
        },
    },
})


# ## Inform function for optuna

# In[8]:


def inform_optuna(**kwargs):
    trial = kwargs["_trial"]
    logs = kwargs["_logs"]
    epoch = kwargs["_epoch"]
    
    error = 100 - logs[0]["epoch_log"][epoch]["test_accuracy"]
    trial.report(error, step=epoch)
    
    if trial.should_prune(epoch):
        raise optuna.structs.TrialPruned()
    return


# ## Hyperparameters

# In[9]:


LOSS_LISTS = [
    [
        ["IndepLoss"] if i == j else ["KLLoss"]
        for i in range(args.num_nodes)
    ] 
    for j in range(args.num_nodes)
]

GATE_LIST = [
    [
        ["ThroughGate", "CutoffGate", "CorrectGate", "LinearGate"]
        for i in range(args.num_nodes)
    ]
    for j in range(args.num_nodes)
]

if args.target_model == "Ensemble":
    GATE_LIST[0] = [
        ["CutoffGate"]
        for i in range(args.num_nodes)
    ]

MODEL_LISTS = [
    [args.target_model]
]+[
    ["ResNet32"]
    for i in range(args.num_ens)
]+[
    ["ResNet32", "ResNet110", "WRN28_2"]
    for i in range(args.num_nodes - args.num_ens - 1)
]


# ## Objective function for optuna

# In[10]:


def objective_func(trial):
    global config
        
    if type(args.num_trial) is int:
        if trial.number > args.num_trial:
            import sys
            sys.exit()
        
    # make dirs
    config.trainer.base_dir = os.path.join(args.optuna_dir, f"{trial.number:04}/")
    utils.make_dirs(config.trainer.base_dir+"log")
    utils.make_dirs(config.trainer.base_dir+"checkpoint")
    
    # change config        
    # set loss funcs & gates
    for source_id, model_losses in enumerate(config.losses):
        for target_id, _ in enumerate(model_losses):
            loss_name = trial.suggest_categorical(f'{source_id:02}_{target_id:02}_loss',
                                                  LOSS_LISTS[source_id][target_id])
            
            loss_args = copy.deepcopy(args_factory.losses[loss_name])
            if "gate" in loss_args.args:
                gate_name = trial.suggest_categorical(f'{source_id:02}_{target_id:02}_gate', GATE_LIST[source_id][target_id])
                loss_args.args.gate = copy.deepcopy(args_factory.gates[gate_name])
            config.losses[source_id][target_id] = loss_args
    
    for model_id in range(len(config.models)):
        # set model
        model_name = trial.suggest_categorical(f"model_{model_id}_name", MODEL_LISTS[model_id])
        model = copy.deepcopy(args_factory.models[model_name])
        config.models[model_id].name = model.name
        config.models[model_id].args = model.args
        
        # set model weight
        is_cutoff = all([loss.args.gate.name == "CutoffGate" for loss in config.losses[model_id]]) 
        is_ensemble = config.models[model_id].name == "Ensemble"
        if is_cutoff & (not is_ensemble):
            config.models[model_id].load_weight.path = model.load_weight.path
        else:
            config.models[model_id].load_weight.path = None
            
    
    config = copy.deepcopy(config)
    
    # save config
    utils.save_json(config, config.trainer.base_dir+r"log/config.json")
    
    # create object
    trainer, nets, criterions, optimizers, train_loader, test_loader, logs = create_object(config)    
    
    # make kwargs
    kwargs = {"_trial": trial,
              "_callback":inform_optuna}
    
    # set seed
    trial.set_user_attr("seed", config.manualSeed)
    
    # raise exception if target model is pretrained.
    if config.models[0].load_weight.path is not None:
        class BlacklistError(optuna.structs.OptunaError):
            pass
        raise BlacklistError()
    
    # start trial
    trainer.train(nets, criterions, optimizers, train_loader, test_loader, logs, trial=trial, **kwargs)
    
    acc = 100 - logs[0]["epoch_log"][config.trainer.epochs]["test_accuracy"]
    
    return acc


# ## Cteate study object

# In[ ]:


utils.make_dirs(args.optuna_dir)

sampler = optuna.samplers.RandomSampler()
pruner = optuna.pruners.SuccessiveHalvingPruner(min_resource=1,
                                                reduction_factor=2,
                                                min_early_stopping_rate=0)

db_path = os.path.join(args.optuna_dir, "optuna.db")
study = optuna.create_study(storage=f"sqlite:///{db_path}",
                            study_name='experiment01',
                            sampler=sampler,
                            pruner=pruner,
                            direction="minimize",
                            load_if_exists=True)


# ## Start optimization

# In[ ]:


study.optimize(objective_func, n_trials=None, timeout=None, n_jobs=1)

