import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder
import random
import config


class PiModelDataset(Dataset):
    def __init__(self, path, mode='train', supervised_ratio=config.supervised_ratio):
        super().__init__()
        self.paths = list((Path(path).rglob('*.png')))
        self.files = [dirname for dirname in self.paths if dirname.parent.parent.name == mode]
        self.labels = [Path(path).parent.name for path in self.files]
        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(self.labels)
        self.mode = mode
        self.supervised_ratio = supervised_ratio

        if self.mode == 'train':
            self.supervised_part = set(random.sample(range(len(self.files)), int(supervised_ratio * len(self.files))))
        else:
            self.supervised_part = set(range(len(self.files)))

        self.transform = transforms.Compose([
            transforms.RandomAffine(0, translate=(config.augment_translation, config.augment_translation)),  # appl
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor()])
        self.to_pil = transforms.ToPILImage(mode=None)
        self.to_tensor = transforms.ToTensor()

    def __getitem__(self, idx):
        image = Image.open(self.files[idx])
        if self.mode == 'train':
            image1 = self.transform(image)
            image2 = self.transform(image)
            image1 = image1 * (config.X_ZCAmax - config.X_ZCAmin) + config.X_ZCAmin
            image2 = image2 * (config.X_ZCAmax - config.X_ZCAmin) + config.X_ZCAmin
            image1 += config.noise_stddev * torch.randn(image1.shape)
            image2 += config.noise_stddev * torch.randn(image2.shape)
        elif self.mode == 'test':
            image1 = self.to_tensor(image)
            image2 = self.to_tensor(image)
            image1 = image1 * (config.X_ZCAmax - config.X_ZCAmin) + config.X_ZCAmin
            image2 = image2 * (config.X_ZCAmax - config.X_ZCAmin) + config.X_ZCAmin
        else:
            print('Wrong mode: available modes - train, test')  # TODO кидать исключение или оставить так
        if idx in self.supervised_part:
            label = self.labels[idx]
        else:
            label = config.no_label

        if self.mode == 'train' or self.mode == 'test':  # TODO переделать
            return image1, image2, label
        else:
            return image, label

    def __len__(self):
        return len(self.files)
