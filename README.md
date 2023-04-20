# EXPERT Model Convert

This script can be used to convert a model trained by EXPERT version 0.X, which is based on TensorFlow Keras, to a model that is compatible with EXPERT version 1.X, which is based on PyTorch Lightning.

## Installation

To use this script, you need to have Python 3 installed. Clone the repository and navigate to the root directory of the project. Then install the required packages by running:

```shell
pip install -r requirements.txt
```

## Usage

The script takes the following command line arguments:

- `--input_model` or `-i`: The path to the model to be converted.
- `--output_model` or `-o`: The path to save the converted model.
For example, to convert general_model trained by EXPERT version 0.X to a model that is compatible with EXPERT version 1.X, run the following command:


```shell
python general_model.py -igeneral_model -ol general_model_lightning
```
Note that the script assumes that the phylogeny file, which is required for converting the model, is located at src/phylogeny.csv relative to the root directory of the project.

## How it works

The convert_model.py script defines a function conver_keras_to_torch() that takes three arguments:

keras_model_path: The path to the Keras model to be converted.
phylogeny: A pandas DataFrame containing the phylogeny information.
torch_model_path: The path to save the converted PyTorch model.
The function first loads the Keras model using the KerasModel class defined in src/keras_model.py, and creates a new PyTorch model using the TorchModel class defined in src/lightning_model.py. The PyTorch model is initialized with the phylogeny information and the ontology information of the Keras model.

Then, the function converts the weights of each layer of the Keras model to the corresponding PyTorch layer. It first converts the base layer, which is the first layer of the model, followed by the inter layers, integ layers, and output layers.

Finally, the function saves the converted PyTorch model to the specified path using the save_blocks() method defined in src/lightning_model.py.