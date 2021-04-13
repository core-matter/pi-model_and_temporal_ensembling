# pi-model_and_temporal_ensembling
Implementation of [Pi-model&amp;temporal ensembling article](https://arxiv.org/abs/1610.02242) on PyTorch.


The article is devoted to semi-supervised problem task, to adress the issue authors provide two aprroaches
namely pi-model and temporal ensembling. We test the former approach on CIFAR 10.
## Installing dependecies:
~~~
pip install -r requirements.txt
~~~
## Training
~~~
train.py [-h]   [--dataset_path DATASET_PATH]
                [--checkpoints_path CHECKPOINTS_PATH]
                [--writer_path WRITER_PATH] [--max_lr MAX_LR]
                [--num_epochs NUM_EPOCHS] [--num_classes NUM_CLASSES]
                [--batch_size BATCH_SIZE] [--start_epoch START_EPOCH]
                [--supervised_ratio SUPERVISED_RATIO]
                [--num_workers NUM_WORKERS] [--device DEVICE]
~~~
## TODO:
~~~~
-add checkpoints

-add prediction

-add temp ens model

-add training statistics
~~~~
