from DotA_Data import DotA_Data
from DotA_NN import DotA_NN
from DotA_Postgre_Data import DotA_Postgre_Data
from pprint import pprint
import json
import pandas as pd
import numpy as np

def data():
    no_matches = 40560

    matches = DotA_Data()
    
    temp_matches = matches.getData(no_matches)  
    
    full_matches = matches.flattenDataframe()

def train():
    matches = DotA_Data()

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

def postgreSqlExplorer():
    no_matches = 40560

    sql_matches = DotA_Postgre_Data()

    #match_ids_data = sql_matches.getMatchIDs(no_matches)

    
    
def main():
    postgreSqlExplorer()
    #data()
    #train()   
    
    
if __name__ == "__main__":
    main()