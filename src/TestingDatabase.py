import random

from torch.utils.data import Dataset
from torchvision import datasets
import os
from PIL import Image

class TestingDatabase(datasets.DatasetFolder):
    def __init__(self, path,transform = None):
        exclude = set(['code','tmp','dataStyleGAN2'])
        self.classes = ["real", "generated"]
        self.samples = []
        self.transform = transform
        real_images = []
        fake_images = []
        fake_count = 0
        real_count = 0
        for entry in os.listdir(path):
            data_folder = os.path.join(path, entry)
            if entry == 'Whatsapp' or entry == "Slack" or entry=="WhichFaceIsReal_Real":
                for root, dirs, files in os.walk(data_folder):
                    for file in files:
                        if file.endswith(".png") or file.endswith(".jpeg") or file.endswith(".jpg"):
                            #if real_count > 1020:
                            #    break
                            real_count += 1
                            item = os.path.join(root, file), 0
                            real_images.append(item)
            if entry == 'ThisPersonDoesNotExist' or entry == 'This-Person-Does-Not-Exist' or entry =="WhichFaceIsReal_Fake":
                for root, dirs, files in os.walk(data_folder, topdown=True):
                    dirs[:] = [d for d in dirs if d not in exclude]
                    for file in files:
                        if file.endswith(".png") or file.endswith(".jpeg") or file.endswith(".jpg"):
                            #if fake_count > 5132:
                            #    break
                            fake_count += 1
                            item = os.path.join(root, file), 1
                            fake_images.append(item)

        real_images.sort()
        fake_images.sort()

        #res = []

        #res.extend(real_images)

        #res.append(fake_images[0])
        #for i in range(4,len(fake_images),5):
        #    res.append(fake_images[i])

        self.samples = real_images + fake_images

    def __len__(self):
        return len(self.samples)

    def find_classes(self, directory):
        classes_mapping = {"real": 0, "generated": 1}
        return self.classes, classes_mapping

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = Image.open(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target
