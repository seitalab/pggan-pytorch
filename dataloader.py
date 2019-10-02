import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image


class dataloader:
    def __init__(self, config):
        # self.root = config.train_data_root
        self.dataset_name = config.dataset
        self.cfg = config
        self.batch_table = {4:32, 8:32, 16:32, 32:16, 64:16, 128:16, 256:12, 512:3, 1024:1} # change this according to available gpu memory.
        self.batchsize = int(self.batch_table[pow(2,2)])        # we start from 2^2=4
        self.imsize = int(pow(2,2))
        self.num_workers = 4

    def renew(self, resl):
        # print('[*] Renew dataloader configuration, load data from {}.'.format(self.root))

        self.batchsize = int(self.batch_table[pow(2,resl)])
        self.imsize = int(pow(2,resl))
        # self.dataset = ImageFolder(
        #             root=self.root,
        #             transform=transforms.Compose(   [
        #                                             transforms.Scale(size=(self.imsize,self.imsize), interpolation=Image.NEAREST),
        #                                             transforms.ToTensor(),
        #                                             ]))       
        transform = transforms.Compose([
                        transforms.Resize((self.imsize, self.imsize)),
                        transforms.ToTensor(),
                        ])
        if self.dataset_name == 'chest-xray':
            self.dataset = ChestXrayDataset(root='./data/nih-chest-xrays/images',
                                            transform=transform,
                                            image_list_file='./data/nih-chest-xrays/labels/train_list.txt')
        elif self.dataset_name == 'emarie':
            self.dataset = EmarieDataset(root_dir='../dataset/emarie/raw',
                                         mix=self.cfg.mix, transform=transform)
        elif self.dataset_name == 'emarie_rose':
            # vertically flip rose images
            rose_transform = transforms.Compose([
                        transforms.Resize((self.imsize, self.imsize)),
                        transforms.RandomVerticalFlip(p=1.0),
                        transforms.ToTensor(),
                        ])
            self.dataset = EmarieRoseDataset(root_dir='../dataset/emarie/raw',
                                             skirt_transform=transform,
                                             rose_transform=rose_transform)

        self.dataloader = DataLoader(
            dataset=self.dataset,
            batch_size=self.batchsize,
            shuffle=True,
            num_workers=self.num_workers
        )

    def __iter__(self):
        return iter(self.dataloader)

    def __next__(self):
        return next(self.dataloader)

    def __len__(self):
        return len(self.dataloader.dataset)

    def get_batch(self):
        dataIter = iter(self.dataloader)
        return next(dataIter)[0].mul(2).add(-1)         # pixel range [-1, 1]


class EmarieDataset(Dataset):
    def __init__(self, root_dir, mix=True, transform=None):
        self.image_names = []
        skirt_dir = os.path.join(root_dir, 'skirt')
        for f in os.listdir(skirt_dir):
            fpath = os.path.join(skirt_dir, f)
            if os.path.isfile(fpath):
                self.image_names.append(fpath)
        if mix:
            shell_dir = os.path.join(root_dir, 'shell')
            for f in os.listdir(shell_dir):
                fpath = os.path.join(shell_dir, f)
                if os.path.isfile(fpath):
                    self.image_names.append(fpath)
        self.transform = transform

    def __getitem__(self, index):
        image_name = self.image_names[index]
        image = Image.open(image_name).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.FloatTensor([1])  # label is dummy

    def __len__(self):
        return len(self.image_names)


class EmarieRoseDataset(Dataset):
    def __init__(self, root_dir, skirt_transform=None, rose_transform=None):
        self.image_names = []
        skirt_dir = os.path.join(root_dir, 'skirt_trimmed_512')
        rose_dir = os.path.join(root_dir, 'rose_512')
        for f in os.listdir(skirt_dir):
            fpath = os.path.join(skirt_dir, f)
            if os.path.isfile(fpath):
                self.image_names.append(fpath)
        for f in os.listdir(rose_dir):
            fpath = os.path.join(rose_dir, f)
            if os.path.isfile(fpath):
                self.image_names.append(fpath)
        self.skirt_transform = skirt_transform
        self.rose_transform = rose_transform

    def __getitem__(self, index):
        image_name = self.image_names[index]
        image = Image.open(image_name).convert('RGB')
        if 'rose' in image_name:
            if self.rose_transform is not None:
                image = self.rose_transform(image)
        elif 'skirt' in image_name:
            if self.skirt_transform is not None:
                image = self.skirt_transform(image)
        return image, torch.FloatTensor([1])  # label is dummy

    def __len__(self):
        return len(self.image_names)


class ChestXrayDataset(Dataset):
    def __init__(self, root, image_list_file, transform=None):
        image_names = []
        labels = []
        with open(image_list_file, "r") as f:
            for line in f:
                items = line.split()
                image_name = items[0]
                label = items[1:]
                label = [int(i) for i in label]
                if all([True if e == 0 else True for e in label]):
                    label.append(1)
                else:
                    label.append(0)
                image_name = os.path.join(root, image_name)
                image_names.append(image_name)
                labels.append(label)

        self.image_names = image_names
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        image_name = self.image_names[index]
        image = Image.open(image_name).convert('RGB')
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.FloatTensor(label)

    def __len__(self):
        return len(self.image_names)
