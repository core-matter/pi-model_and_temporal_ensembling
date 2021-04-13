import torch
from tqdm import tqdm
from torch.nn.functional import softmax
import config
from utils import ramp_down, ramp_up


def fit_epoch(model, train_loader, criterion, optimizer, w_t, device='cpu'):
    """
    :param model:  neural network from net.py
    :param train_loader: data loader
    :param criterion: cross entropy
    :param optimizer: adam
    :param w_t: weight of unsupervised part in loss function
    :param device: device for tensors computation
    :return: train accuracy and losses of both parts of the objective as in the article
    """
    running_loss = 0.0
    running_corrects = 0
    processed_data = 0
    model.train()
    for inputs1, inputs2, labels in train_loader:
        optimizer.zero_grad()

        unlabeled_index = torch.where(labels == config.no_label)
        labeled_index = torch.where(labels != config.no_label)

        inputs1 = inputs1.to(device)
        inputs2 = inputs2.to(device)
        labels = labels.to(device)
        z1 = model(inputs1)
        z2 = model(inputs2)
        print('z1', z1)
        if len(z1[labeled_index]) != 0:
            tr_supervised_loss = criterion(z1[labeled_index], labels[labeled_index]) + \
                                 (w_t / (2 * inputs1.size(0))) * torch.sum((softmax(z1[labeled_index], dim=1) -
                                                                            softmax(z2[labeled_index], dim=1)) ** 2)
        else:
            tr_supervised_loss = torch.tensor(0)

        if len(z1[unlabeled_index]) != 0:
            tr_unsupervised_loss = (w_t / (2 * inputs1.size(0))) * torch.sum((softmax(z1[unlabeled_index], dim=1) -
                                                                              softmax(z2[unlabeled_index], dim=1)) ** 2)
        else:
            tr_unsupervised_loss = torch.tensor(0)

        loss = (tr_supervised_loss + tr_unsupervised_loss)
        loss.backward()
        optimizer.step()

        if len(z1[labeled_index]) != 0:
            preds = torch.argmax(z1[labeled_index], dim=1)
            running_corrects += torch.sum(preds == labels[labeled_index])

        processed_data += len(z1[labeled_index])
        running_loss += loss.item() * inputs1.size(0)

    train_loss = running_loss / processed_data  # TODO нужно делить на inputs.size(0)
    train_acc = running_corrects.cpu().numpy() / processed_data

    return train_loss, train_acc, tr_supervised_loss, tr_unsupervised_loss


def eval_epoch(model, val_loader, criterion, w_t, device='cpu'):
    """
    :param model:  neural network from net.py
    :param val_loader: data loader
    :param criterion: cross entropy
    :param w_t: weight of unsupervised part in loss function
    :param device: device for tensors computation
    :return: val accuracy and losses of both parts of the objective as in the article
    """
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    processed_size = 0

    for inputs1, inputs2, labels in val_loader:
        inputs1 = inputs1.to(device)
        inputs2 = inputs2.to(device)
        labels = labels.to(device)

        z1 = model(inputs1)
        z2 = model(inputs2)

        val_supervised_loss = criterion(z1, labels) + (w_t / (2 * inputs1.size(0))) * torch.sum((softmax(z1, dim=1) -
                                                                                                 softmax(z2, dim=1)) ** 2) #TODO inputs1.size(0)) поменять на букву из статьи и импортить из конфига
        val_unsupervised_loss = (w_t / (2 * inputs1.size(0))) * torch.sum((softmax(z1, dim=1) -
                                                                           softmax(z2, dim=1)) ** 2)

        loss = (val_supervised_loss + val_unsupervised_loss)
        preds = torch.argmax(z1, 1)
        running_loss += loss.item() * inputs1.size(0)
        running_corrects += torch.sum(preds == labels.data)
        processed_size += inputs1.size(0)
    val_loss = running_loss / processed_size
    val_acc = running_corrects.cpu().numpy() / processed_size
    return val_loss, val_acc, val_supervised_loss, val_unsupervised_loss


def train(model, train_loader, val_loader, optimizer, scheduler, criterion, epochs, start_epoch, writer, supervised_ratio, device):
    """
    :param model: neural net from net.py
    :param train_loader: data loader
    :param val_loader: data loader
    :param optimizer: adam
    :param scheduler:
    :param criterion: cross entropy
    :param epochs: total number of epochs
    :param start_epoch: epoch to start from when training is resumed
    :param writer: logs on tensorboard
    :param supervised_ratio: ration of labeled data
    :return: training statistics information packed in dictionary
    """
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    max_value = config.unsup_weight_max * supervised_ratio
    beta1 = config.beta1
    beta2 = config.beta2
    ramp_down_beta1_target = 0.5
    least_val_loss = float('inf')
    for epoch in tqdm(range(start_epoch, epochs)):
        if epoch != 0:
            w_t = ramp_up(epoch, epochs) * max_value
        else:
            w_t = 0
        train_loss, train_acc, tr_supervised_loss, tr_unsupervised_loss = fit_epoch(model, train_loader, criterion, optimizer, w_t, device)
        val_loss, val_acc, val_supervised_loss, val_unsupervised_loss = eval_epoch(model, val_loader, criterion, w_t, device)
        beta1 = ramp_down(epoch, epochs) * beta1 + (1.0 - ramp_down(epoch, epochs)) * ramp_down_beta1_target
        optimizer.param_groups[0]['betas'] = (beta1, beta2)
        writer.add_scalars("loss", {'train_loss': train_loss, 'val_loss': val_loss}, epoch)
        writer.add_scalars("accuracy", {'train_acc': train_acc, 'val_acc': val_acc}, epoch)
        scheduler.step()
        print(f' \n epoch: {epoch} \n train_loss:{train_loss} \n val_loss:{val_loss}\n train_acc:{train_acc}\n val_acc:{val_acc} \n')
        if supervised_ratio < 1:
            print(f' tr_supervised_loss: {tr_supervised_loss} \n tr_unsupervised_loss:{tr_unsupervised_loss}')
            print(f' val_supervised_loss: {val_supervised_loss} \n val_unsupervised_loss:{val_unsupervised_loss}')
            writer.add_scalars("supervised_losses", {'tr_supervised_loss': tr_supervised_loss, 'val_supervised_loss': val_supervised_loss}, epoch)
            writer.add_scalars("unsupervised_losses", {'tr_unsupervised_loss': tr_unsupervised_loss, 'val_unsupervised_loss': val_unsupervised_loss}, epoch)

        if val_loss < least_val_loss:
            least_val_loss = val_loss
            torch.save(model.state_dict(), './best-val-model.pt')

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        torch.save({'history': history}, './history.pth')

    return history
