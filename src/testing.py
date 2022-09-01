import argparse

from verdeolivaFineTuning import testModel
from resnet50nodown import *
from TestingDatabase import TestingDatabase
from torch.utils.data import DataLoader
from torchvision import transforms

batch_size = 2
num_classes = 2

device = "cuda:0"

if __name__ == '__main__':
	parser = argparse.ArgumentParser(
		description="This script fine-tunes the default pre-trained resnet50 model.",
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)

	parser.add_argument('--input_folder', '-i', type=str, default='./fake_dataset',
						help='input folder with PNG and JPEG images')

	parser.add_argument('--weights_path', '-m', type=str, default='./weights/gandetection_resnet50nodown_stylegan2.pth',
						help='weights path of the network')

	config = parser.parse_args()
	data_dir = config.input_folder
	weights_path = config.weights_path

	starting_model = resnet50nodown(device, weights_path)

	starting_model.change_output(2)
	starting_model.fc.to(device)

	starting_model.load_state_dict(torch.load("finetuned_model_weights.pth"))
	starting_model.eval()

	transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
		transforms.Resize([1024,1024])
    ])

	testing_db = TestingDatabase(data_dir,transforms)

	dataloader = DataLoader(testing_db,batch_size=batch_size,shuffle=True)

	dictionary_dataloader = {"testing":dataloader}

	testModel(starting_model,dictionary_dataloader,device)
