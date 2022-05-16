import os

import cv2
import torchvision.transforms as transforms
from fastbook import download_url, search_images_ddg
from torch.utils.data import Dataset, random_split
from tqdm import tqdm

CLASS_NAMES = ['air_plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
MAX_IMGS = 999


def download_images():
    for class_name in CLASS_NAMES:
        urls = search_images_ddg(class_name if "_" not in class_name else class_name.replace("_", ""), max_images=MAX_IMGS)
        for ix, url in tqdm(enumerate(urls)):
            download_url(url, f'images/{class_name}/{ix}.jpg')


class ImageDataset(Dataset):
    def __init__(self, img_folder=os.path.join(os.getcwd(), "images")):
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ])
        self.img_folder = img_folder

        self.image_names = [os.path.join(i, j) for i in os.listdir('images') for j in os.listdir(os.path.join('images', i))]
        self.labels = [i for i in os.listdir('images') for _ in os.listdir(os.path.join('images', i))]

    # The __len__ function returns the number of samples in our dataset.
    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, index):
        image = cv2.imread(os.path.join(self.img_folder, self.image_names[index]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = self.transform(image)
        targets = self.labels[index]

        sample = {'image': image, 'labels': targets}

        return sample


def fetch_data_set(train_set_perc=0.75, download=True):
    if download:
        if (not os.path.exists(os.path.join(os.getcwd(), "images"))) or len(os.listdir(os.path.join(os.getcwd(), "images"))) == 0:
            download_images()

    dataset = ImageDataset()
    size = train_set_perc * len(dataset)
    train_set, valid_set = random_split(dataset, [train_set_perc * size, size - train_set_perc * size])
    return train_set, valid_set


fetch_data_set()
