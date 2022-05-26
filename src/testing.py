import argparse

from resnet50fineTuning import *

batch_size = 16
num_classes = 2

device = "cpu"

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

	starting_model.load_state_dict(torch.load("trained_model_weights.pth"))
	starting_model.eval()

	testing_db = TuningDatabase(data_dir)

	for index in range(0,len(testing_db)):
		sample, label = testing_db[index]
		output = starting_model.apply(sample)
		print(f"output is {output}")
		print(f"label is {label}")
