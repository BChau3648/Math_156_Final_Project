# Math 156: Classifying Geographical Land Use and Land Coverage

## Dataset

The dataset comes from Helber et al. [1]. The instructions to download the RGB images can be found on their [GitHub page](https://github.com/phelber/EuroSAT). This folder containing the RGB satellite images separated by their land use type should replace the empty [EuroSAT_RGB folder](EuroSAT_RGB/).


## Training the Model

The main code to run is the [training.ipynb](training/training_cnn.ipynb) notebook inside the training folder. It imports the convolutional neural network (CNN) model from the [model.py](model/cnn.py) module and the dataset from [preprocessing.py](preprocessing/preprocessing.py). The notebook can be run directly, after ensuring that the paths to the modules, the paths to the data, and the hyperparameters are changed appropriately.

## References

[1] Eurosat: A novel dataset and deep learning benchmark for land use and land cover classification. Patrick Helber, Benjamin Bischke, Andreas Dengel, Damian Borth. IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing, 2019.
