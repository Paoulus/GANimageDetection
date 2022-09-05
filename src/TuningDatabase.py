import random

from torch.utils.data import Dataset
from torchvision import datasets
import os
from PIL import Image

class TuningDatabase(datasets.DatasetFolder):
    def __init__(self, path,transform = None):
        exclude = set(['code','tmp','dataStyleGAN2'])
        self.classes = ["real", "generated"]
        self.samples = []
        self.transform = transform
        self.downsamplervariable = 0
        self.real_images_count = 0
        self.fake_images_count = 0

        for root_1, dirs_1, files_1 in os.walk(path, topdown=True):
            for entry in dirs_1:
                data_folder = os.path.join(root_1, entry)
                if entry == 'FFHQ' or entry == 'Real' or entry == '0_Real':
                    for root, dirs, files in os.walk(data_folder, topdown=True):
                        for file in sorted(files):
                            if (file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg"))  and (self.downsamplervariable % 1 == 0) and self.real_images_count < 50:
                                item = os.path.join(root, file), 0
                                self.samples.append(item)
                                self.real_images_count += 1
                            self.downsamplervariable += 1
                elif (((entry == 'StyleGAN' or entry == 'StyleGAN2') and os.path.basename(path) == 'forensicsDatasets') or entry=='Fake' or entry=='1_Fake'):
                    exclude = set(['code', 'tmp', 'dataStyleGAN2'])
                    # exclude = set(['StyleGAN', 'StyleGAN2'])
                    for root, dirs, files in os.walk(data_folder, topdown=True):
                        dirs[:] = [d for d in dirs if d not in exclude]
                        self.downsamplervariable += 0
                        for file in sorted(files):
                            if (file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg")) and (self.downsamplervariable % 5 == 0) and self.fake_images_count < 50:
                                item = os.path.join(root, file), 1
                                self.samples.append(item)
                                self.fake_images_count += 1
                            self.downsamplervariable += 1
        pass

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

class TuningDatabaseWithRandomSampling(datasets.DatasetFolder):
    def __init__(self, path,transform = None,real_amount=50,fake_amount=50,seed=451):
        exclude = set(['code','tmp','dataStyleGAN2'])
        self.classes = ["real", "generated"]
        self.samples = []
        self.transform = transform
        self.downsamplervariable = 0
        self.real_images_count = 0
        self.fake_images_count = 0
        self.fake_images = []
        self.real_images = []

        rand_generator = random.Random(seed)

        for root_1, dirs_1, files_1 in os.walk(path, topdown=True):
            for entry in dirs_1:
                data_folder = os.path.join(root_1, entry)
                if entry == 'FFHQ' or entry == 'Real' or entry == '0_Real':
                    for root, dirs, files in os.walk(data_folder, topdown=True):
                        for file in sorted(files):
                            if (file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg"))  and (self.downsamplervariable % 1 == 0) and self.real_images_count < real_amount:
                                item = os.path.join(root, file), 0
                                self.real_images.append(item)
                                self.real_images_count += 1
                            self.downsamplervariable += 1
                elif (((entry == 'StyleGAN' or entry == 'StyleGAN2') and os.path.basename(path) == 'forensicsDatasets') or entry=='Fake' or entry=='1_Fake'):
                    exclude = set(['code', 'tmp', 'dataStyleGAN2'])
                    files_in_folder = []
                    # exclude = set(['StyleGAN', 'StyleGAN2'])
                    for root, dirs, files in os.walk(data_folder, topdown=True):
                        dirs[:] = [d for d in dirs if d not in exclude]
                        self.downsamplervariable += 0
                        for file in sorted(files):
                            if (file.endswith(".png") or file.endswith(".jpg") or file.endswith(".jpeg")) and (self.downsamplervariable % 5 == 0):
                                item = os.path.join(root, file), 1
                                files_in_folder.append(item)
                                self.fake_images_count += 1
                            self.downsamplervariable += 1
                    self.fake_images.extend(rand_generator.sample(files_in_folder,fake_amount//2))
            
        self.samples = self.real_images + self.fake_images



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