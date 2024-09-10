from DotA_Data import DotA_Data
from DotA_NN import DotA_NN
from pprint import pprint
import json
import pandas as pd
import numpy as np

def main():
    features = ["match_id", "radiant_win"]
    no_matches = 1

    matches = DotA_Data(features)
    '''
    temp_matches = matches.getData(no_matches)
    with open('DotA_data.json', 'w') as json_file:
        json.dump(temp_matches, json_file, indent=4)   
    
    full_matches_df = matches.flattenDataframe()
    with open('flattened_dota_data.json', 'w') as json_file:
        json.dump(full_matches_df, json_file, indent=4)
    '''
    json_file_path = 'C:/Users/Reece Lazarus/DOTA2_Match_Prediction/flattened_dota_data.json'
    matches_df = matches.jsonToDataframe(json_file_path)


    ################## Hyperparameters ###################
    train_size = 0.6
    epochs = 2
    batch_size = 24
    
    ######################################################
    nn = DotA_NN()
    nn.train(matches_df, train_size, epochs, batch_size)

if __name__ == "__main__":
    main()