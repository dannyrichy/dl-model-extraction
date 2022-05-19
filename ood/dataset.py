from __future__ import print_function

import os
import os.path
import pickle

import cv2
import numpy as np
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image

CLASS_NAMES = ['air_plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


class OODDataset(data.Dataset):
    """`DownsampleImageNet`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again
    """
    map_dict = {
        404: 0,
        436: 1,
        734: 1,
        8: 2,
        18: 2,
        284: 3,
        285: 3,
        207: 5,
        208: 5,
        30: 6,
        32: 6,
        510: 8,
        724: 8,
        569: 9,
        867: 9
    }
    train_list = [
        ['train_data_batch_1'],
        ['train_data_batch_2'],
        ['train_data_batch_3'],
        ['train_data_batch_4'],
        ['train_data_batch_5'],
        ['train_data_batch_6'],
        ['train_data_batch_7'],
        ['train_data_batch_8'],
        ['train_data_batch_9'],
        ['train_data_batch_10']
        #         ['val_data']
    ]
    test_list = [
    ]

    def __init__(self, root="/home/shared/ood", img_folder="/home/shared/extended_dataset",
                 transform=transforms.ToTensor(), target_transform=None):
        self.root = os.path.expanduser(root)
        self.transform1 = transforms.ToTensor()
        self.transform2 = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ])
        self.target_transform = target_transform
        self.img_folder = img_folder
        self.image_names = list()
        self.labels = list()
        for i in os.listdir(self.img_folder):
            for j in os.listdir(os.path.join(self.img_folder, i)):
                self.image_names.append(os.path.join(i, j))
                self.labels.append(CLASS_NAMES.index(i))
        # now load the picked numpy arrays
        self.train_data = []
        self.train_labels = []
        for fentry in self.train_list:
            f = fentry[0]
            file = os.path.join(self.root, f)
            fo = open(file, 'rb')
            entry = pickle.load(fo)
            tmp = list()
            for ix, lbl in enumerate(entry['labels']):
                if lbl - 1 in self.map_dict.keys():
                    tmp.append(entry['data'][ix, :])
                    self.train_labels += [self.map_dict[lbl - 1]]
            self.train_data.append(np.stack(tmp))
            fo.close()

        self.train_data = np.concatenate(self.train_data)
        [picnum, pixel] = self.train_data.shape
        pixel = int(np.sqrt(pixel / 3))
        self.train_data = self.train_data.reshape((picnum, 3, pixel, pixel))
        self.train_data = self.train_data.transpose((0, 2, 3, 1))  # convert to HWC

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if index < 18938:
            img, target = self.train_data[index], self.train_labels[index]
            img = Image.fromarray(img)

            img = self.transform1(img)

            if self.target_transform is not None:
                target = self.target_transform(target)

            return img, target
        elif index < 24027:
            image = cv2.imread(os.path.join(self.img_folder, self.image_names[index - 18938]))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            image = self.transform2(image)
            targets = self.labels[index - 18938]

            # sample = {'image': image, 'labels': targets}

            return image, targets

    def __len__(self):
        return len(self.train_data) + len(self.image_names)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        return fmt_str

