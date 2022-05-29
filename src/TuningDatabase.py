from torch.utils.data import Dataset
from torchvision import datasets
import os
from PIL import Image

class TuningDatabase(datasets.DatasetFolder):
    def __init__(self, path,transform = None):
        exclude = set(['code'])
        self.classes = ["real", "generated"]
        self.samples = []
        self.transform = transform
        for entry in os.listdir(path):
            data_folder = os.path.join(path, entry)
            if entry == 'FFHQ':
                for root, dirs, files in os.walk(data_folder):
                    for file in files:
                        if file.endswith(".png"):
                            item = os.path.join(root, file), 0
                            self.samples.append(item)
            if entry == 'styleGAN' or entry == 'StyleGAN2':
                for root, dirs, files in os.walk(data_folder, topdown=True):
                    dirs[:] = [d for d in dirs if d not in exclude]
                    for file in files:
                        if file.endswith(".png"):
                            item = os.path.join(root, file), 1
                            self.samples.append(item)

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
