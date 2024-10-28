from DotA_Data import DotA_Data
from DotA_NN import DotA_NN
from DotA_Stacked_BiLSTM import DotA_Stacked_BiLSTM
from DotA_Postgre_Data import DotA_Postgre_Data
from pprint import pprint
import json
import pandas as pd
import numpy as np

def data():
    matches = DotA_Data()
    
    #no_matches = 40560
    #temp_matches = matches.getData(no_matches) 
    
    #full_matches = matches.flattenDataframe()

def train_NN():
    matches = DotA_Data()

    #json_file_path = '/home-mscluster/rlazarus/flattened_postgre_dota_data.json'
    #json_file_path = 'C:/Users/Reece Lazarus/DOTA2_Match_Prediction/flattened_postgre_dota_data.json'
    json_file_path = 'C:/Users/Reece Lazarus/DOTA2_Match_Prediction/flattened_dota_data.json'
    chunksize = 500

    input_variables = ['radiant_win', 'players_0_hero_id', 'players_1_hero_id', 'players_2_hero_id', 'players_3_hero_id', 'players_4_hero_id', 'players_5_hero_id', 'players_6_hero_id', 'players_7_hero_id', 'players_8_hero_id', 'players_9_hero_id', 'players_0_isRadiant', 'players_1_isRadiant', 'players_2_isRadiant', 'players_3_isRadiant', 'players_4_isRadiant', 'players_5_isRadiant', 'players_6_isRadiant', 'players_7_isRadiant', 'players_8_isRadiant', 'players_9_isRadiant']

    dataset, total_rows = matches.getDataset(json_file_path=json_file_path, chunksize=chunksize, input_variables=input_variables)

    print(f"Total matches in the dataset: {total_rows}")

    ################## NN Hyperparameters ################
    train_size = 0.1
    epochs = 20
    batch_size = 64
    learning_rate = 0.001
    
    ######################################################
    nn = DotA_NN()
    nn.train(dataset, train_size, epochs, batch_size, learning_rate)

def train_BiLSTM():
    matches = DotA_Data()

    #json_file_path = '/home-mscluster/rlazarus/flattened_postgre_dota_data.json'
    #json_file_path = 'C:/Users/Reece Lazarus/DOTA2_Match_Prediction/flattened_postgre_dota_data.json'
    json_file_path = 'C:/Users/Reece Lazarus/DOTA2_Match_Prediction/flattened_dota_data.json'
    chunksize = 500

    input_variables = ['radiant_win', 'players_0_hero_id', 'players_1_hero_id', 'players_2_hero_id', 'players_3_hero_id', 'players_4_hero_id', 'players_5_hero_id', 'players_6_hero_id', 'players_7_hero_id', 'players_8_hero_id', 'players_9_hero_id', 'players_0_isRadiant', 'players_1_isRadiant', 'players_2_isRadiant', 'players_3_isRadiant', 'players_4_isRadiant', 'players_5_isRadiant', 'players_6_isRadiant', 'players_7_isRadiant', 'players_8_isRadiant', 'players_9_isRadiant']

    dataset, total_rows = matches.getDataset(json_file_path=json_file_path, chunksize=chunksize, input_variables=input_variables)

    print(f"Total matches in the dataset: {total_rows}")

    ################## NN Hyperparameters ################
    train_size = 0.6
    epochs = 20
    batch_size = 20
    
    ######################################################
    lstm = DotA_Stacked_BiLSTM()
    lstm.train(dataset, train_size, epochs, batch_size)

def postgreSqlExplorer():
    no_matches = 40560
    rate_limit = 1200
    backup_frequency = 100

    sql_matches = DotA_Postgre_Data()

    #match_ids_data = sql_matches.getMatchIDs(no_matches)

    match_data = sql_matches.getData(no_matches, rate_limit, backup_frequency)

def evaluateModel():
    model = tf.keras.models.load_model('dota_nn_model.keras')   
    # Assume test_dataset is a tf.data.Dataset object with batched test data
    test_loss, test_accuracy = model.evaluate(test_dataset, verbose=1)
    print(f'Test Accuracy: {test_accuracy}')
    print(f'Test Loss: {test_loss}')



    
def main():
    #postgreSqlExplorer()
    #data()
    #train_NN()
    train_BiLSTM()
    #evaluateModel()
    
    
if __name__ == "__main__":
    main()