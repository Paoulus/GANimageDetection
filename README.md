## Description
This repo contains ongoing work on the finetuning of the detector originally presented in 

**Are GAN generated images easy to detect? A critical analysis of the state-of-the-art**  
Diego Gragnaniello, Davide Cozzolino, Francesco Marra, Giovanni Poggi and Luisa Verdoliva.
<br />In IEEE International Conference on Multimedia and Expo (ICME), 2021.

The original source code can be found [here](https://github.com/grip-unina/GANimageDetection).

## Warning
This version of GANimageDetectiom  has been adapted to work with the [TrueFace dataset](https://mmlab.disi.unitn.it/resources/published-datasets#h.4bwcjdyr0h5i), respecting its folder structure. Please refer to the [original repository](https://github.com/grip-unina/GANimageDetection) if you want to use this with more general purpose datasets.

## Installation
To install, run `pip install -r requirements.txt` to install all the required libraries. Use of `venv` is reccomended.

## Running finetuning
Launch the finetuning script with:

    python -u main.py -c config.json

The `config.json` contains the configuration for a run. A sample is located at `samples/config.json`. A description of the main parameters can be found below.

    "datasetPath":path to dataset to use for finetuning,
    "loadCheckpoint":if true, resume finetuning from desired path,
    "checkpointPath":path to checkpoint to load, if we're resuming from one,
    "logsPath":path where logs should be saved,
    "finetunedWeightsFilename":filename for the finetuned weights file,
    "testDatasetCatalogue":path to a catalogue of test samples,
    "validationDatasetCatalogue":path to a catalogue of validation samples,
    "performValidation":false,
    "performTesting":false,
    "learningRate":0.01,
    "epochs":30,
    "exportToONNX":if true, will export the finetuned weights to ONNX file. Used for inspecting the finetuned model


## Testing finetuned weights
Launch the testing script with:

    python -u testingModel.py -i /folder/with/dataset -m /file/with/finetuned/weights 

## License
Project distributed under GRIP license. Please have a look at the [license file](LICENSE.md)