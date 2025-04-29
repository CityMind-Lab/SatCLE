import torch
import numpy as np
import pandas as pd
from torch import nn

def get_population_data(pred="temp", norm_y=True, norm_x=True):
    inc = pd.read_csv('data/downstream/labels_population.csv')
    inc = np.array(inc)
    y = inc[:,0].reshape(-1)
    coords = inc[:,1:]
    coords = coords[:,::-1]
    y_mean = np.mean(y)
    y_std = np.std(y)
    y = (y - y_mean) / y_std

    return torch.tensor(coords.copy()), torch.tensor(y), y_mean, y_std

def get_elevation_data(pred="temp",
                      norm_y=True,
                      norm_x=True
                      ):

    inc = pd.read_csv('data/downstream/labels_elevation.csv')
    inc = inc.dropna(subset=[inc.columns[0]])
    inc = np.array(inc)
    y = inc[:,0].reshape(-1)
    coords = inc[:,1:]
    coords = coords[:,::-1]
    y_mean = np.mean(y)
    y_std = np.std(y)
    y = (y - y_mean) / y_std
    return torch.tensor(coords.copy()),  torch.tensor(y), y_mean, y_std

def get_carbon_data(pred="temp",
                      norm_y=True,
                      norm_x=True
                      ):
    inc = pd.read_csv('data/downstream/carbon/carbon.csv')
    inc = inc.dropna(subset=[inc.columns[0]])
    inc = np.array(inc)
    y = inc[:,0].reshape(-1)
    y = np.log(y)
    coords = inc[:,1:]
    y_mean = np.mean(y)
    y_std = np.std(y)
    y = (y - y_mean) / y_std

    return torch.tensor(coords.copy()),  torch.tensor(y), y_mean, y_std


def get_carbon_data_specific_region(
                        file_train,
                        file_test
                        ):
    
    inc_train = np.array(pd.read_csv(f'data/downstream/carbon/{file_train}'))
    inc_test = np.array(pd.read_csv(f'data/downstream/carbon/{file_test}'))
    
    sample_size = int(0.01 * len(inc_test))
    random_indices = np.random.choice(len(inc_test), size=sample_size, replace=False)
    sampled_inc_test = inc_test[random_indices]

    inc_train = np.concatenate((inc_train, sampled_inc_test), axis=0)

    inc_test = np.delete(inc_test, random_indices, axis=0)

    y_train = inc_train[:,0].reshape(-1)
    coords_train = inc_train[:,1:]
    
    y_test = inc_test[:,0].reshape(-1)
    coords_test = inc_test[:,1:]

    return torch.tensor(coords_train.copy()),  torch.tensor(y_train), torch.tensor(coords_test.copy()), torch.tensor(y_test)


def get_countrycode_data(pred="temp",
                      norm_y=True,
                      norm_x=True
                      ):
    inc = pd.read_csv('data/downstream/lat_lon_country.csv')
    inc = inc.dropna(subset=[inc.columns[0]])
    inc = np.array(inc)
    y = inc[:,2].reshape(-1)
    coords = inc[:,:2]
    coords = coords[:,::-1]

    return torch.tensor(coords.copy()),  torch.tensor(y)



def get_landvegetation_data(pred="temp",
                      norm_y=True,
                      norm_x=True
                      ):
    inc = pd.read_csv('data/downstream/landuse/landvegetation.csv')
    inc = inc.dropna(subset=[inc.columns[0]])
    inc = np.array(inc)
    y = inc[:,0].reshape(-1)
    coords = inc[:,1:]
    coords = coords[:,::-1]

    return torch.tensor(coords.copy()),  torch.tensor(y)



class MLP(nn.Module):
    def __init__(self, input_dim, dim_hidden, num_layers, out_dims):
        super(MLP, self).__init__()

        layers = []
        layers += [nn.Linear(input_dim, dim_hidden, bias=True), nn.ReLU()] # Input layer
        layers += [nn.Linear(dim_hidden, dim_hidden, bias=True), nn.ReLU()] * num_layers # Hidden layers
        layers += [nn.Linear(dim_hidden, out_dims, bias=True)] # Output layer

        self.features = nn.Sequential(*layers)

    def forward(self, x):
        return self.features(x)