import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path
import numpy as np


class ZCA(object):
    def __init__(self, epsilon=1e-5):
        self.eps = epsilon

    def fit(self, x):
        m = x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])
        self.mean = m.mean(axis=0)
        m = m - m.mean(axis=0)
        cov = np.cov(m, rowvar=False, bias=True)
        U, S, V = np.linalg.svd(cov)
        self.zca_whitening_matrix = U.dot(np.diag(1.0 / np.sqrt(S + self.eps))).dot(U.T)

    def transform(self, x):
        m = x.reshape(x.shape[0], x.shape[1] * x.shape[2] * x.shape[3])
        m = m - self.mean
        x_zca = np.dot(np.array(m), self.zca_whitening_matrix)
        return x_zca


if __name__ == '__main__':
    path = '.\cifar10\cifar10'

    train_files = list((Path(path + '\\train').rglob('*.png')))
    to_tensor = transforms.ToTensor()
    to_pil = transforms.ToPILImage()
    train_tensors = [to_tensor(Image.open(file)) for file in train_files]
    stack_tensors = torch.stack(train_tensors, dim=0)
    zca = ZCA()
    zca.fit(stack_tensors)
    train_zca = zca.transform(stack_tensors)
    train_zca_rescaled = ((train_zca - train_zca.min()) / (train_zca.max() - train_zca.min())).astype(np.float32)
    train_zca_min = train_zca.min()
    train_zca_max = train_zca.max()
    whitened_train = [to_pil(torch.tensor(array.reshape(3, 32, 32))) for array in train_zca_rescaled]
    train_image_path = list(zip(whitened_train, train_files))


    Path('.\ZCA_cifar10').mkdir()
    Path('.\ZCA_cifar10\\train').mkdir()
    Path('.\ZCA_cifar10\\test').mkdir()

    unique_labels = list(set([Path(file).parent.name for file in train_files]))

    for label in unique_labels:
        Path('.\ZCA_cifar10\\train' + label).mkdir()
        Path('.\ZCA_cifar10\\test' + label).mkdir()
    for element in train_image_path:

        element[0].save('.\ZCA_cifar10\\train' + element[1].parent.name + '\\' + element[1].name)


