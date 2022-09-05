import argparse

from verdeolivaFineTuning import testModel
from resnet50nodown import *
from TestingDatabase import TestingDatabase
from TuningDatabase import TuningDatabase,TuningDatabaseWithRandomSampling
from torch.utils.data import DataLoader
from torchvision import transforms

batch_size = 1
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
	web_dir = "/home/paolochiste/Samples-to-test/Samples-To-Test-Finetuned/Samples/"
	data_dir = "/media/antoniostefani/c73e5e73-45ea-4b43-8dbb-cebfdacf033d/truebees/forensicsDatasets"
	weights_path = config.weights_path

	starting_model = resnet50nodown(device, weights_path)

	starting_model.change_output(2)
	starting_model.fc.to(device)

	starting_model.load_state_dict(torch.load("/home/paolochiste/Repos/GANimageDetection/logs/09_04_23_36_29/finetuned_weights.pth"))
	starting_model.eval()

	transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

	testing_db = TuningDatabaseWithRandomSampling(data_dir,transforms)

	dataloader = DataLoader(testing_db,batch_size=batch_size,shuffle=True)

	dictionary_dataloader = {"testing":dataloader}

	print("Going to test on: ")
	for image_path, _ in testing_db.samples:
		print(image_path)

	testModel(starting_model,dictionary_dataloader,device)
