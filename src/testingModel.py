import argparse
import datetime
import os

from torch.utils.data import DataLoader
from torchvision import transforms

from verdeolivaNetwork import *
from testing.testingDatabase import TestingDatabase
from testing.testingUtils import test_on_folder
from tuningDatabase import TuningDatabaseWithRandomSampling,FolderDataset

batch_size = 1
num_classes = 2

device = "cpu"

if __name__ == '__main__':
	parser = argparse.ArgumentParser(
		description="This script tests the desired model with some data.",
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)

	parser.add_argument('--input_folder', '-i', type=str, default='./fake_dataset',
						help='input folder with PNG and JPEG images')

	parser.add_argument('--weights_path', '-m', type=str, default='./weights/gandetection_resnet50nodown_stylegan2.pth',
						help='weights path of the network')

	script_args = parser.parse_args()
	postsocial_dir = "postsocial-samples/Samples/"
	weights_path = script_args.weights_path

	starting_model = resnet50nodown(device, weights_path)

	starting_model.change_output(2)
	starting_model.fc.to(device)

	finetuned_weights_path = "logs/11_30_21_24_09/ft-postsoc-from-presoc-linear-lr.pth"
	starting_model.load_state_dict(torch.load(finetuned_weights_path))
	starting_model.eval()

	transforms = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
		transforms.Resize([1024,1024])
	])

	# here is using the default seed, meaning that we are testing on the same data we used for training (as a security check)
	#testing_db = TuningDatabaseWithRandomSampling(data_dir,transforms,seed=9842765,real_amount=300,fake_amount=300)
	testing_db = TestingDatabase(postsocial_dir,transform=transforms)

	real_images_count = 0
	fake_images_count = 0
	for path, label in testing_db:
		if label == 0 : 
			real_images_count += 1
		else:
			fake_images_count += 1

	dataloader = DataLoader(testing_db,batch_size=batch_size,shuffle=True)

	dictionary_dataloader = {"testing":dataloader}

	print("Testing model with finetuned weights: {} on day {} ".format(finetuned_weights_path,datetime.datetime.now().strftime("%b %a %d at %H:%M:%S")))
	print("Number of real images: {} fake images: {}",real_images_count,fake_images_count)

	print("Going to test on: ")
	for image_path, _ in testing_db.samples:
		print(image_path)

	folder_dataloaders = []

	folders_real = ["WhichFaceIsReal_Real","Facebook","Whatsapp","Slack","Telegram/WhichFaceIsReal_Real","TelegramProfilo"]
	folders_fake = ["This-Person-Does-Not-Exist","ThisPersonDoesNotExist","Telegram/WhichFaceIsReal_Fake","WhichFaceIsReal_Fake","Telegram/This-Person-Does-Not-Exist","Telegram/ThisPersonDoesNotExist","Telegram/WhichFaceIsReal_Fake"]
	
	for folder in folders_real:
		folder = os.path.join(postsocial_dir,folder)
		folder_dataloaders.append(DataLoader(FolderDataset(folder,transforms,0),batch_size=batch_size))

	for folder in folders_fake:
		folder = os.path.join(postsocial_dir,folder)
		folder_dataloaders.append(DataLoader(FolderDataset(folder,transforms,1),batch_size=batch_size))
	
	
	folder_dataloaders.append(DataLoader(TuningDatabaseWithRandomSampling("/media/mmlab/c73e5e73-45ea-4b43-8dbb-cebfdacf033d/truebees/forensicsDatasets",transform=transforms,real_amount=60,fake_amount=60),batch_size=1))
	folder_dataloaders.append(DataLoader(TuningDatabaseWithRandomSampling("/media/mmlab/c73e5e73-45ea-4b43-8dbb-cebfdacf033d/truebees/Shared_Dataset",transform=transforms,real_amount=60,fake_amount=60),batch_size=1))
	
	folder_dataloaders.append(DataLoader(TuningDatabaseWithRandomSampling("/media/mmlab/c73e5e73-45ea-4b43-8dbb-cebfdacf033d/truebees/Telegram",transform=transforms,real_amount=60,fake_amount=60),batch_size=1))


	output_file = open("testing-post-social-from-pre-2.txt",'w')

	for folder_dataloader in folder_dataloaders:
		if len(folder_dataloader.dataset.samples) > 0:
			accuracy_on_folder = test_on_folder(starting_model,folder_dataloader,transforms,device)
			name = str(folder_dataloader.dataset)
			line = name + " " + str(accuracy_on_folder * 100) + "\n--------------\n" 
			print(line)
			output_file.write(line)
		else:
			print ("Dataloader" + str(folder_dataloader) + "has zero samples. Check folder path.")
	
	output_file.close()