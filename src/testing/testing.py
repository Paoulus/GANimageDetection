import argparse
import datetime

from main import testModel
from verdeolivaNetwork import *
from TestingDatabase import TestingDatabase
from Databases import TuningDatabaseWithRandomSampling
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

	script_args = parser.parse_args()
	web_dir = "/home/paolochiste/Samples-to-test/Samples-To-Test-Finetuned/Samples/"
	data_dir = "/media/mmlab/c73e5e73-45ea-4b43-8dbb-cebfdacf033d/truebees/forensicsDatasets"
	weights_path = script_args.weights_path

	starting_model = resnet50nodown(device, weights_path)

	starting_model.change_output(2)
	starting_model.fc.to(device)

	finetuned_weights_path = "/home/paolochiste/Repos/GANimageDetection/logs/09_06_17_25_30/finetuned_weights.pth"
	starting_model.load_state_dict(torch.load(finetuned_weights_path))
	starting_model.eval()

	transforms = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
	])

	# here is using the default seed, meaning that we are testing on the same data we used for training (as a security check)
	testing_db = TuningDatabaseWithRandomSampling(data_dir,transforms,seed=75089)
	#testing_db = TestingDatabase(web_dir,transform=transforms)

	real_images_count = 0
	fake_images_count = 0
	for path, label in testing_db:
		if label == 0 : 
			real_images_count += 1
		else:
			fake_images_count += 1

	dataloader = DataLoader(testing_db,batch_size=batch_size,shuffle=True)

	dictionary_dataloader = {"testing":dataloader}

	print("Testing model with finetuned weights: {} on day {} ".format(finetuned_weights_path,datetime.now().strftime("%b %a %d at %H:%M:%S")))
	print("Number of real images: {} fake images: {}",real_images_count,fake_images_count)

	print("Going to test on: ")
	for image_path, _ in testing_db.samples:
		print(image_path)

	testModel(starting_model,dictionary_dataloader,device)
