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
            self.dataset = EmarieDataset(root_dir='./data/20181018_Square/',
                                         transform=transform)
        elif self.dataset_name == 'emarie-skirt-shell':
            self.dataset = EmarieSkirtShellDataset(root_dir='./data/skirt_shell/', transform=transform)

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
    def __init__(self, root_dir, transform=None):
        files = os.listdir(root_dir)
        files_file = [os.path.join(root_dir, f) for f in files if os.path.isfile(os.path.join(root_dir, f))]
        self.image_names = files_file
        self.transform = transform

    def __getitem__(self, index):
        image_name = self.image_names[index]
        image = Image.open(image_name).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, torch.FloatTensor([1])  # label is dummy

    def __len__(self):
        return len(self.image_names)


class EmarieSkirtShellDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        files_file = []
        for root, _, filenames in os.walk(root_dir):
            files = [os.path.join(root, f) for f in filenames if os.path.isfile(os.path.join(root, f))]
            files_file.extend(files)
        self.image_names = files_file
        self.transform = transform

    def __getitem__(self, index):
        image_name = self.image_names[index]
        image = Image.open(image_name).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
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
