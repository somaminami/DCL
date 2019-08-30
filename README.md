# Deep Cooperative Learning

Implementation of Deep Cooperative Learning


## Environment
 - python 3.6.8
 - ipython 7.6.0
 - jupyterlab 0.35.3
 - pytorch 1.1.0
 - torchvision 0.3.0
 - Optuna 0.13.0
 - easydict 1.9
 - graphviz 0.10.1



## Usage
1. Train pre-trained model
```
ipython ./pre-train.py -- --target_model=ResNet32 --dataset=CIFAR100 --gpu_id=0 --save_dir=./pre-train/ResNet32/
ipython ./pre-train.py -- --target_model=ResNet110 --dataset=CIFAR100 --gpu_id=0 --save_dir=./pre-train/ResNet110/
ipython ./pre-train.py -- --target_model=WRN28_2 --dataset=CIFAR100 --gpu_id=0 --save_dir=./pre-train/WRN28_2/
```

2. Optimize knowledge transfer graph in parallel distributed environment  
Run train.py mltiple times.
```
ipython ./train.py -- --num_nodes=3 --target_model=ResNet32 --dataset=CIFAR100 --gpu_id=0 --num_trial=1500 --optuna_dir=./result/
```

3. Watch result  
Open watch.ipynb on jupyterlab and run all cells.
