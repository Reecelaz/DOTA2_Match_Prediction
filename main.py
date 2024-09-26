from DotA_Data import DotA_Data
from DotA_NN import DotA_NN
from pprint import pprint
import json
import pandas as pd
import numpy as np

def data():
    no_matches = 200000

    matches = DotA_Data()
    
    temp_matches = matches.getData(no_matches)
    with open('DotA_data.json', 'w') as json_file:
        json.dump(temp_matches, json_file, indent=4)   
    
    full_matches_df = matches.flattenDataframe()
    with open('flattened_dota_data2.json', 'w') as json_file:
        json.dump(full_matches_df, json_file, indent=4)

def train():
    features = ["match_id", "radiant_win"]
    matches = DotA_Data(features)

    json_file_path = 'C:/Users/Reece Lazarus/DOTA2_Match_Prediction/flattened_dota_data.json'
    matches_df = matches.jsonToDataframe(json_file_path)
    print("Total matches in dataset: ", matches_df.shape[0])

    ################## Hyperparameters ###################
    train_size = 0.65
    epochs = 20
    batch_size = 24
    learning_rate = 0.001
    
    ######################################################
    nn = DotA_NN()
    nn.train(matches_df, train_size, epochs, batch_size, learning_rate)
    
def main():
    data()
    #train()   
    
    
if __name__ == "__main__":
    main()