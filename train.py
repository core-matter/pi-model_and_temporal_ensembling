from modules.Dataset import PiModelDataset
from modules.net import Model
from torch.utils.tensorboard import SummaryWriter
from modules.utils import *
import torch
from torch.utils.data import DataLoader
from modules.train_scripts import train
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default='.\ZCA_cifar10')
    parser.add_argument('--checkpoints_path', type=str, default='.\checkpoints')
    parser.add_argument('--writer_path', type=str, default='.\logs', help='path to save tensorboard logs')
    parser.add_argument('--max_lr', type=float, default=0.003, help='maximum learning rate')
    parser.add_argument('--num_epochs', type=int, default=10, help='total number of epochs')
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--start_epoch', type=int, default=0, help='epoch to start from when training is resumed')
    parser.add_argument('--supervised_ratio', type=float, default=1, help='ratio of supervised data')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    ###################################################################################################

    DEVICE = torch.device(args.device)
    PATH = args.dataset_path
    WRITER_PATH = args.writer_path

    train_dataset = PiModelDataset(PATH, mode='train', supervised_ratio=args.supervised_ratio)
    val_dataset = PiModelDataset(PATH, mode='test')
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    pi_model = Model(n_classes=args.num_classes).to(DEVICE)
    pi_model.apply(weights_init)

    optimizer = torch.optim.Adam(pi_model.parameters(), lr=args.max_lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: ramp_up(epoch) * ramp_down(epoch, args.num_epochs))
    criterion = nn.CrossEntropyLoss()
    writer = SummaryWriter(log_dir=WRITER_PATH)

    history = train(pi_model, train_loader, val_loader, optimizer, scheduler, criterion, epochs=args.num_epochs, start_epoch=args.start_epoch, writer=writer, supervised_ratio=args.supervised_ratio, device=DEVICE)
