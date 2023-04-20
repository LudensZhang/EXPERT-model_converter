from src.keras_model import Model as KerasModel
from src.lightning_model import Model as TorchModel
import torch
import pandas as pd
import warnings 
import argparse

warnings.filterwarnings("ignore")

def conver_keras_to_torch(keras_model_path, phylogeny, torch_model_path):
    print('Loading keras model...')
    keras_model = KerasModel(restore_from=keras_model_path)
    
    torch_model = TorchModel(phylogeny=phylogeny, 
                  ontology=keras_model.ontology,
                  open_set=False)
    torch_model.statistics = keras_model.statistics
    
    print('Converting base layer...')
    torch_model.base[1].weight.data = torch.from_numpy(keras_model.base.get_weights()[0]).float().T
    torch_model.base[1].bias.data = torch.from_numpy(keras_model.base.get_weights()[1]).float()
    torch_model.base[3].weight.data = torch.from_numpy(keras_model.base.get_weights()[2]).float().T
    torch_model.base[3].bias.data = torch.from_numpy(keras_model.base.get_weights()[3]).float()
    
    print('Converting inter layers...')
    for i in range(keras_model.n_layers):
        torch_model.spec_inters[i][0].weight.data = torch.from_numpy(keras_model.spec_inters[i].get_weights()[0]).float().T
        torch_model.spec_inters[i][0].bias.data = torch.from_numpy(keras_model.spec_inters[i].get_weights()[1]).float()
        torch_model.spec_inters[i][2].weight.data = torch.from_numpy(keras_model.spec_inters[i].get_weights()[2]).float().T
        torch_model.spec_inters[i][2].bias.data = torch.from_numpy(keras_model.spec_inters[i].get_weights()[3]).float()
        torch_model.spec_inters[i][4].weight.data = torch.from_numpy(keras_model.spec_inters[i].get_weights()[4]).float().T
        torch_model.spec_inters[i][4].bias.data = torch.from_numpy(keras_model.spec_inters[i].get_weights()[5]).float()
        
    print('Converting integ layer...')
    for i in range(keras_model.n_layers):
        torch_model.spec_integs[i][0].weight.data = torch.from_numpy(keras_model.spec_integs[i].get_weights()[0]).float().T
        torch_model.spec_integs[i][0].bias.data = torch.from_numpy(keras_model.spec_integs[i].get_weights()[1]).float()
        
    print('Converting output layer...')
    for i in range(keras_model.n_layers):
        torch_model.spec_outputs[i][0].weight.data = torch.from_numpy(keras_model.spec_outputs[i].get_weights()[0]).float().T
        torch_model.spec_outputs[i][0].bias.data = torch.from_numpy(keras_model.spec_outputs[i].get_weights()[1]).float()
        
    print('Saving torch model...')
    torch_model.save_blocks(torch_model_path)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_model', '-i', type=str, help='Path to the model to be converted')
    parser.add_argument('--output_model', '-o', type=str, help='Path to save the converted model')
    args = parser.parse_args()
    
    phylogeny = pd.read_csv('src/phylogeny.csv', index_col=0)
    conver_keras_to_torch(keras_model_path=args.input_model,
                          phylogeny=phylogeny,
                          torch_model_path=args.output_model)
