from DotA_Data import DotA_Data
from DotA_NN import DotA_NN
from DotA_Stacked_BiLSTM import DotA_Stacked_BiLSTM
from DotA_Postgre_Data import DotA_Postgre_Data
from pprint import pprint
import json
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

#json_file_path = '/home-mscluster/rlazarus/flattened_postgre_dota_data.json'
#json_file_path = '/home-mscluster/rlazarus/flattened_dota_data.json'
json_file_path = 'C:/Users/Reece Lazarus/DOTA2_Match_Prediction/flattened_postgre_dota_data.json'
#json_file_path = 'C:/Users/Reece Lazarus/DOTA2_Match_Prediction/flattened_dota_data.json'

hero_id_columns = [
    'players_0_hero_id', 'players_1_hero_id', 'players_2_hero_id', 'players_3_hero_id', 'players_4_hero_id',
    'players_5_hero_id', 'players_6_hero_id', 'players_7_hero_id', 'players_8_hero_id', 'players_9_hero_id'
    ]

is_radiant_columns = [
        'players_0_isRadiant', 'players_1_isRadiant', 'players_2_isRadiant', 'players_3_isRadiant', 'players_4_isRadiant',
        'players_5_isRadiant', 'players_6_isRadiant', 'players_7_isRadiant', 'players_8_isRadiant', 'players_9_isRadiant'
    ]

extra_columns = [
    # Gold and XP per minute
    'players_0_gold_per_min', 'players_0_xp_per_min', 'players_1_gold_per_min', 'players_1_xp_per_min',
    'players_2_gold_per_min', 'players_2_xp_per_min', 'players_3_gold_per_min', 'players_3_xp_per_min',
    'players_4_gold_per_min', 'players_4_xp_per_min', 'players_5_gold_per_min', 'players_5_xp_per_min',
    'players_6_gold_per_min', 'players_6_xp_per_min', 'players_7_gold_per_min', 'players_7_xp_per_min',
    'players_8_gold_per_min', 'players_8_xp_per_min', 'players_9_gold_per_min', 'players_9_xp_per_min',
    
    # Additional fields for each player
    'players_0_player_slot', 'players_0_kills', 'players_0_deaths', 'players_0_assists', 'players_0_last_hits',
    'players_0_denies', 'players_0_level', 'players_0_net_worth', 'players_0_aghanims_scepter', 'players_0_aghanims_shard',
    'players_0_moonshard', 'players_0_hero_damage', 'players_0_tower_damage', 'players_0_hero_healing',
    'players_0_gold', 'players_0_gold_spent', 'players_0_benchmarks_gold_per_min_raw', 'players_0_benchmarks_xp_per_min_raw',
    'players_0_benchmarks_kills_per_min_raw', 'players_0_benchmarks_last_hits_per_min_raw',
    'players_0_benchmarks_hero_damage_per_min_raw', 'players_0_benchmarks_hero_healing_per_min_raw',
    'players_0_benchmarks_tower_damage_raw',
    
    'players_1_player_slot', 'players_1_kills', 'players_1_deaths', 'players_1_assists', 'players_1_last_hits',
    'players_1_denies', 'players_1_level', 'players_1_net_worth', 'players_1_aghanims_scepter', 'players_1_aghanims_shard',
    'players_1_moonshard', 'players_1_hero_damage', 'players_1_tower_damage', 'players_1_hero_healing',
    'players_1_gold', 'players_1_gold_spent', 'players_1_benchmarks_gold_per_min_raw', 'players_1_benchmarks_xp_per_min_raw',
    'players_1_benchmarks_kills_per_min_raw', 'players_1_benchmarks_last_hits_per_min_raw',
    'players_1_benchmarks_hero_damage_per_min_raw', 'players_1_benchmarks_hero_healing_per_min_raw',
    'players_1_benchmarks_tower_damage_raw',
    
    'players_2_player_slot', 'players_2_kills', 'players_2_deaths', 'players_2_assists', 'players_2_last_hits',
    'players_2_denies', 'players_2_level', 'players_2_net_worth', 'players_2_aghanims_scepter', 'players_2_aghanims_shard',
    'players_2_moonshard', 'players_2_hero_damage', 'players_2_tower_damage', 'players_2_hero_healing',
    'players_2_gold', 'players_2_gold_spent', 'players_2_benchmarks_gold_per_min_raw', 'players_2_benchmarks_xp_per_min_raw',
    'players_2_benchmarks_kills_per_min_raw', 'players_2_benchmarks_last_hits_per_min_raw',
    'players_2_benchmarks_hero_damage_per_min_raw', 'players_2_benchmarks_hero_healing_per_min_raw',
    'players_2_benchmarks_tower_damage_raw',
    
    'players_3_player_slot', 'players_3_kills', 'players_3_deaths', 'players_3_assists', 'players_3_last_hits',
    'players_3_denies', 'players_3_level', 'players_3_net_worth', 'players_3_aghanims_scepter', 'players_3_aghanims_shard',
    'players_3_moonshard', 'players_3_hero_damage', 'players_3_tower_damage', 'players_3_hero_healing',
    'players_3_gold', 'players_3_gold_spent', 'players_3_benchmarks_gold_per_min_raw', 'players_3_benchmarks_xp_per_min_raw',
    'players_3_benchmarks_kills_per_min_raw', 'players_3_benchmarks_last_hits_per_min_raw',
    'players_3_benchmarks_hero_damage_per_min_raw', 'players_3_benchmarks_hero_healing_per_min_raw',
    'players_3_benchmarks_tower_damage_raw',

    'players_4_player_slot', 'players_4_kills', 'players_4_deaths', 'players_4_assists', 'players_4_last_hits',
    'players_4_denies', 'players_4_level', 'players_4_net_worth', 'players_4_aghanims_scepter', 'players_4_aghanims_shard',
    'players_4_moonshard', 'players_4_hero_damage', 'players_4_tower_damage', 'players_4_hero_healing',
    'players_4_gold', 'players_4_gold_spent', 'players_4_benchmarks_gold_per_min_raw', 'players_4_benchmarks_xp_per_min_raw',
    'players_4_benchmarks_kills_per_min_raw', 'players_4_benchmarks_last_hits_per_min_raw',
    'players_4_benchmarks_hero_damage_per_min_raw', 'players_4_benchmarks_hero_healing_per_min_raw',
    'players_4_benchmarks_tower_damage_raw',

    'players_5_player_slot', 'players_5_kills', 'players_5_deaths', 'players_5_assists', 'players_5_last_hits',
    'players_5_denies', 'players_5_level', 'players_5_net_worth', 'players_5_aghanims_scepter', 'players_5_aghanims_shard',
    'players_5_moonshard', 'players_5_hero_damage', 'players_5_tower_damage', 'players_5_hero_healing',
    'players_5_gold', 'players_5_gold_spent', 'players_5_benchmarks_gold_per_min_raw', 'players_5_benchmarks_xp_per_min_raw',
    'players_5_benchmarks_kills_per_min_raw', 'players_5_benchmarks_last_hits_per_min_raw',
    'players_5_benchmarks_hero_damage_per_min_raw', 'players_5_benchmarks_hero_healing_per_min_raw',
    'players_5_benchmarks_tower_damage_raw',

    'players_6_player_slot', 'players_6_kills', 'players_6_deaths', 'players_6_assists', 'players_6_last_hits',
    'players_6_denies', 'players_6_level', 'players_6_net_worth', 'players_6_aghanims_scepter', 'players_6_aghanims_shard',
    'players_6_moonshard', 'players_6_hero_damage', 'players_6_tower_damage', 'players_6_hero_healing',
    'players_6_gold', 'players_6_gold_spent', 'players_6_benchmarks_gold_per_min_raw', 'players_6_benchmarks_xp_per_min_raw',
    'players_6_benchmarks_kills_per_min_raw', 'players_6_benchmarks_last_hits_per_min_raw',
    'players_6_benchmarks_hero_damage_per_min_raw', 'players_6_benchmarks_hero_healing_per_min_raw',
    'players_6_benchmarks_tower_damage_raw',

    'players_7_player_slot', 'players_7_kills', 'players_7_deaths', 'players_7_assists', 'players_7_last_hits',
    'players_7_denies', 'players_7_level', 'players_7_net_worth', 'players_7_aghanims_scepter', 'players_7_aghanims_shard',
    'players_7_moonshard', 'players_7_hero_damage', 'players_7_tower_damage', 'players_7_hero_healing',
    'players_7_gold', 'players_7_gold_spent', 'players_7_benchmarks_gold_per_min_raw', 'players_7_benchmarks_xp_per_min_raw',
    'players_7_benchmarks_kills_per_min_raw', 'players_7_benchmarks_last_hits_per_min_raw',
    'players_7_benchmarks_hero_damage_per_min_raw', 'players_7_benchmarks_hero_healing_per_min_raw',
    'players_7_benchmarks_tower_damage_raw',

    'players_8_player_slot', 'players_8_kills', 'players_8_deaths', 'players_8_assists', 'players_8_last_hits',
    'players_8_denies', 'players_8_level', 'players_8_net_worth', 'players_8_aghanims_scepter', 'players_8_aghanims_shard',
    'players_8_moonshard', 'players_8_hero_damage', 'players_8_tower_damage', 'players_8_hero_healing',
    'players_8_gold', 'players_8_gold_spent', 'players_8_benchmarks_gold_per_min_raw', 'players_8_benchmarks_xp_per_min_raw',
    'players_8_benchmarks_kills_per_min_raw', 'players_8_benchmarks_last_hits_per_min_raw',
    'players_8_benchmarks_hero_damage_per_min_raw', 'players_8_benchmarks_hero_healing_per_min_raw',
    'players_8_benchmarks_tower_damage_raw',

    'players_9_player_slot', 'players_9_kills', 'players_9_deaths', 'players_9_assists', 'players_9_last_hits',
    'players_9_denies', 'players_9_level', 'players_9_net_worth', 'players_9_aghanims_scepter', 'players_9_aghanims_shard',
    'players_9_moonshard', 'players_9_hero_damage', 'players_9_tower_damage', 'players_9_hero_healing',
    'players_9_gold', 'players_9_gold_spent', 'players_9_benchmarks_gold_per_min_raw', 'players_9_benchmarks_xp_per_min_raw',
    'players_9_benchmarks_kills_per_min_raw', 'players_9_benchmarks_last_hits_per_min_raw',
    'players_9_benchmarks_hero_damage_per_min_raw', 'players_9_benchmarks_hero_healing_per_min_raw',
    'players_9_benchmarks_tower_damage_raw'
    ]

# extra_columns = [
#     # Gold and XP per minute
#     'players_0_gold_per_min', 'players_1_gold_per_min', 'players_2_gold_per_min', 'players_3_gold_per_min',
#     'players_4_gold_per_min', 'players_5_gold_per_min', 'players_6_gold_per_min', 'players_7_gold_per_min',
#     'players_8_gold_per_min', 'players_9_gold_per_min',
#     'players_0_xp_per_min', 'players_1_xp_per_min', 'players_2_xp_per_min', 'players_3_xp_per_min',
#     'players_4_xp_per_min', 'players_5_xp_per_min', 'players_6_xp_per_min', 'players_7_xp_per_min',
#     'players_8_xp_per_min', 'players_9_xp_per_min',

#     # Player slot
#     'players_0_player_slot', 'players_1_player_slot', 'players_2_player_slot', 'players_3_player_slot',
#     'players_4_player_slot', 'players_5_player_slot', 'players_6_player_slot', 'players_7_player_slot',
#     'players_8_player_slot', 'players_9_player_slot',

#     # Kills, deaths, assists
#     'players_0_kills', 'players_1_kills', 'players_2_kills', 'players_3_kills', 'players_4_kills',
#     'players_5_kills', 'players_6_kills', 'players_7_kills', 'players_8_kills', 'players_9_kills',
#     'players_0_deaths', 'players_1_deaths', 'players_2_deaths', 'players_3_deaths', 'players_4_deaths',
#     'players_5_deaths', 'players_6_deaths', 'players_7_deaths', 'players_8_deaths', 'players_9_deaths',
#     'players_0_assists', 'players_1_assists', 'players_2_assists', 'players_3_assists', 'players_4_assists',
#     'players_5_assists', 'players_6_assists', 'players_7_assists', 'players_8_assists', 'players_9_assists',

#     # Last hits and denies
#     'players_0_last_hits', 'players_1_last_hits', 'players_2_last_hits', 'players_3_last_hits', 'players_4_last_hits',
#     'players_5_last_hits', 'players_6_last_hits', 'players_7_last_hits', 'players_8_last_hits', 'players_9_last_hits',
#     'players_0_denies', 'players_1_denies', 'players_2_denies', 'players_3_denies', 'players_4_denies',
#     'players_5_denies', 'players_6_denies', 'players_7_denies', 'players_8_denies', 'players_9_denies',

#     # Level and net worth
#     'players_0_level', 'players_1_level', 'players_2_level', 'players_3_level', 'players_4_level',
#     'players_5_level', 'players_6_level', 'players_7_level', 'players_8_level', 'players_9_level',
#     'players_0_net_worth', 'players_1_net_worth', 'players_2_net_worth', 'players_3_net_worth', 'players_4_net_worth',
#     'players_5_net_worth', 'players_6_net_worth', 'players_7_net_worth', 'players_8_net_worth', 'players_9_net_worth',

#     # Items: Aghanim's Scepter, Shard, Moonshard
#     'players_0_aghanims_scepter', 'players_1_aghanims_scepter', 'players_2_aghanims_scepter', 'players_3_aghanims_scepter',
#     'players_4_aghanims_scepter', 'players_5_aghanims_scepter', 'players_6_aghanims_scepter', 'players_7_aghanims_scepter',
#     'players_8_aghanims_scepter', 'players_9_aghanims_scepter',
#     'players_0_aghanims_shard', 'players_1_aghanims_shard', 'players_2_aghanims_shard', 'players_3_aghanims_shard',
#     'players_4_aghanims_shard', 'players_5_aghanims_shard', 'players_6_aghanims_shard', 'players_7_aghanims_shard',
#     'players_8_aghanims_shard', 'players_9_aghanims_shard',
#     'players_0_moonshard', 'players_1_moonshard', 'players_2_moonshard', 'players_3_moonshard', 'players_4_moonshard',
#     'players_5_moonshard', 'players_6_moonshard', 'players_7_moonshard', 'players_8_moonshard', 'players_9_moonshard',

#     # Hero, tower, and healing damage
#     'players_0_hero_damage', 'players_1_hero_damage', 'players_2_hero_damage', 'players_3_hero_damage', 'players_4_hero_damage',
#     'players_5_hero_damage', 'players_6_hero_damage', 'players_7_hero_damage', 'players_8_hero_damage', 'players_9_hero_damage',
#     'players_0_tower_damage', 'players_1_tower_damage', 'players_2_tower_damage', 'players_3_tower_damage', 'players_4_tower_damage',
#     'players_5_tower_damage', 'players_6_tower_damage', 'players_7_tower_damage', 'players_8_tower_damage', 'players_9_tower_damage',
#     'players_0_hero_healing', 'players_1_hero_healing', 'players_2_hero_healing', 'players_3_hero_healing', 'players_4_hero_healing',
#     'players_5_hero_healing', 'players_6_hero_healing', 'players_7_hero_healing', 'players_8_hero_healing', 'players_9_hero_healing',

#     # Gold and gold spent
#     'players_0_gold', 'players_1_gold', 'players_2_gold', 'players_3_gold', 'players_4_gold',
#     'players_5_gold', 'players_6_gold', 'players_7_gold', 'players_8_gold', 'players_9_gold',
#     'players_0_gold_spent', 'players_1_gold_spent', 'players_2_gold_spent', 'players_3_gold_spent', 'players_4_gold_spent',
#     'players_5_gold_spent', 'players_6_gold_spent', 'players_7_gold_spent', 'players_8_gold_spent', 'players_9_gold_spent',

#     # Benchmarks: gold per minute, XP per minute, kills, last hits, hero damage, hero healing, tower damage
#     'players_0_benchmarks_gold_per_min_raw', 'players_1_benchmarks_gold_per_min_raw', 'players_2_benchmarks_gold_per_min_raw',
#     'players_3_benchmarks_gold_per_min_raw', 'players_4_benchmarks_gold_per_min_raw', 'players_5_benchmarks_gold_per_min_raw',
#     'players_6_benchmarks_gold_per_min_raw', 'players_7_benchmarks_gold_per_min_raw', 'players_8_benchmarks_gold_per_min_raw',
#     'players_9_benchmarks_gold_per_min_raw',
#     'players_0_benchmarks_xp_per_min_raw', 'players_1_benchmarks_xp_per_min_raw', 'players_2_benchmarks_xp_per_min_raw',
#     'players_3_benchmarks_xp_per_min_raw', 'players_4_benchmarks_xp_per_min_raw', 'players_5_benchmarks_xp_per_min_raw',
#     'players_6_benchmarks_xp_per_min_raw', 'players_7_benchmarks_xp_per_min_raw', 'players_8_benchmarks_xp_per_min_raw',
#     'players_9_benchmarks_xp_per_min_raw',
#     'players_0_benchmarks_kills_per_min_raw', 'players_1_benchmarks_kills_per_min_raw', 'players_2_benchmarks_kills_per_min_raw',
#     'players_3_benchmarks_kills_per_min_raw', 'players_4_benchmarks_kills_per_min_raw', 'players_5_benchmarks_kills_per_min_raw',
#     'players_6_benchmarks_kills_per_min_raw', 'players_7_benchmarks_kills_per_min_raw', 'players_8_benchmarks_kills_per_min_raw',
#     'players_9_benchmarks_kills_per_min_raw',
#     'players_0_benchmarks_last_hits_per_min_raw', 'players_1_benchmarks_last_hits_per_min_raw', 'players_2_benchmarks_last_hits_per_min_raw',
#     'players_3_benchmarks_last_hits_per_min_raw', 'players_4_benchmarks_last_hits_per_min_raw', 'players_5_benchmarks_last_hits_per_min_raw',
#     'players_6_benchmarks_last_hits_per_min_raw', 'players_7_benchmarks_last_hits_per_min_raw', 'players_8_benchmarks_last_hits_per_min_raw',
#     'players_9_benchmarks_last_hits_per_min_raw',
#     'players_0_benchmarks_hero_damage_per_min_raw', 'players_1_benchmarks_hero_damage_per_min_raw', 'players_2_benchmarks_hero_damage_per_min_raw',
#     'players_3_benchmarks_hero_damage_per_min_raw', 'players_4_benchmarks_hero_damage_per_min_raw', 'players_5_benchmarks_hero_damage_per_min_raw',
#     'players_6_benchmarks_hero_damage_per_min_raw', 'players_7_benchmarks_hero_damage_per_min_raw', 'players_8_benchmarks_hero_damage_per_min_raw',
#     'players_9_benchmarks_hero_damage_per_min_raw',
#     'players_0_benchmarks_hero_healing_per_min_raw', 'players_1_benchmarks_hero_healing_per_min_raw', 'players_2_benchmarks_hero_healing_per_min_raw',
#     'players_3_benchmarks_hero_healing_per_min_raw', 'players_4_benchmarks_hero_healing_per_min_raw', 'players_5_benchmarks_hero_healing_per_min_raw',
#     'players_6_benchmarks_hero_healing_per_min_raw', 'players_7_benchmarks_hero_healing_per_min_raw', 'players_8_benchmarks_hero_healing_per_min_raw',
#     'players_9_benchmarks_hero_healing_per_min_raw',
#     'players_0_benchmarks_tower_damage_raw', 'players_1_benchmarks_tower_damage_raw', 'players_2_benchmarks_tower_damage_raw',
#     'players_3_benchmarks_tower_damage_raw', 'players_4_benchmarks_tower_damage_raw', 'players_5_benchmarks_tower_damage_raw',
#     'players_6_benchmarks_tower_damage_raw', 'players_7_benchmarks_tower_damage_raw', 'players_8_benchmarks_tower_damage_raw',
#     'players_9_benchmarks_tower_damage_raw'
# ]

def data():
    matches = DotA_Data()
    
    #no_matches = 40560
    #temp_matches = matches.getData(no_matches) 
    
    #full_matches = matches.flattenDataframe()
'''
def Plot_Metric_vs_Features(num_features, accuracies_nn, accuracies_lstm, ylabel, title, fileName):
    plt.plot(num_features, accuracies_nn, label='Neural Network', color='teal')
    plt.plot(num_features, accuracies_lstm, label='LSTM', color='orange')

    plt.xlabel('Number of Features')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()

    plt.savefig(fileName)

    plt.show()
'''
def Plot_Metric_vs_Features(num_features, accuracies_nn, accuracies_lstm, ylabel, title, fileName):
    # Set the figure size with reduced height for a slimmer plot
    plt.figure(figsize=(8, 4))

    # Use a clearer, contrasting color scheme
    plt.plot(num_features, accuracies_nn, label='Neural Network', color='#1f77b4')  # Blue
    plt.plot(num_features, accuracies_lstm, label='Stacked BiLSTM', color='#ff7f0e')  # Orange

    plt.xlabel('Number of Features')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()

    # Save the plot to the specified filename
    plt.savefig(fileName, bbox_inches='tight')  # Remove unnecessary whitespace
    plt.show()

def train_NN():
    chunksize = 10

    # Load and prepare the dataset
    dota_data = DotA_Data()
    one_hot_encoded_dataset = dota_data.prepare_dataset(
        dataset_path=json_file_path,
        hero_id_columns=hero_id_columns,
        is_radiant_columns=is_radiant_columns,
        extra_columns=extra_columns,
        chunksize=chunksize
    )

    ################## NN Baseline Hyperparameters ################
    train_size = 0.6
    epochs = 150
    batch_size = 64
    learning_rate = 0.005
    
    ###############################################################
    print("Neural Network with all extra features and metadata")
    nn = DotA_NN()
    test_accuracy, roc_auc, precision, test_loss = nn.train(one_hot_encoded_dataset, train_size, epochs, batch_size, learning_rate)
    
def train_BiLSTM():
    chunksize = 10

    # Load and prepare the dataset
    dota_data = DotA_Data()
    one_hot_encoded_dataset = dota_data.prepare_dataset(
        dataset_path=json_file_path,
        hero_id_columns=hero_id_columns,
        is_radiant_columns=is_radiant_columns,
        extra_columns=extra_columns,
        chunksize=chunksize
    )

    ################## Stacked BiLSTM Baseline Hyperparameters ################
    train_size = 0.8
    epochs = 200
    batch_size = 64
    
    ######################################################
    print("Stacked BiLSTM with all features")
    lstm = DotA_Stacked_BiLSTM()
    lstm.train(one_hot_encoded_dataset, train_size, epochs, batch_size)


def postgreSqlExplorer():
    no_matches = 40560
    rate_limit = 1200
    backup_frequency = 100

    sql_matches = DotA_Postgre_Data()

    #match_ids_data = sql_matches.getMatchIDs(no_matches)

    match_data = sql_matches.getData(no_matches, rate_limit, backup_frequency)

def evaluateModels():
    num_features_list = []

    accuracies_nn = []
    accuracies_lstm = []

    loss_nn = []
    loss_lstm = []

    precision_nn = []
    precision_lstm = []

    roc_nn = []
    roc_lstm = []

    for i in range(10, len(extra_columns) + 1, 10):
        feature_subset = extra_columns[:i]
        num_features = len(feature_subset)
        print("Features: ", num_features, " out of ", len(extra_columns))
        num_features_list.append(num_features)

        chunksize = 10

        dota_data = DotA_Data()
        one_hot_encoded_dataset = dota_data.prepare_dataset(
            dataset_path=json_file_path,
            hero_id_columns=hero_id_columns,
            is_radiant_columns=is_radiant_columns,
            extra_columns=feature_subset,
            chunksize=chunksize
        )

        ####################### NN Evaluation ########################
        train_size = 0.6
        epochs = 150
        batch_size = 64
        learning_rate = 0.005
        
        nn = DotA_NN()
        
        test_accuracy, roc_auc, precision, test_loss = nn.train(one_hot_encoded_dataset, train_size, epochs, batch_size, learning_rate)
        accuracies_nn.append(test_accuracy)
        loss_nn.append(test_loss)
        precision_nn.append(precision)
        roc_nn.append(roc_auc)

        ####################### Stacked BiLSTM Evaluation ########################
        train_size = 0.8
        epochs = 200
        batch_size = 64
        
        lstm = DotA_Stacked_BiLSTM()
        
        test_accuracy, roc_auc, precision, test_loss = lstm.train(one_hot_encoded_dataset, train_size, epochs, batch_size)
        accuracies_lstm.append(test_accuracy)
        loss_lstm.append(test_loss)
        precision_lstm.append(precision)
        roc_lstm.append(roc_auc)

    np.save("num_features_list.npy", np.array(num_features_list))
    np.save("accuracies_nn.npy", np.array(accuracies_nn))
    np.save("accuracies_lstm.npy", np.array(accuracies_lstm))
    np.save("loss_nn.npy", np.array(loss_nn))
    np.save("loss_lstm.npy", np.array(loss_lstm))
    np.save("precision_nn.npy", np.array(precision_nn))
    np.save("precision_lstm.npy", np.array(precision_lstm))
    np.save("roc_nn.npy", np.array(roc_nn))
    np.save("roc_lstm.npy", np.array(roc_lstm))

    Plot_Metric_vs_Features(num_features_list, accuracies_nn, accuracies_lstm, ylabel='Accuracy', title='Comparison of Accuracies: NN vs LSTM', fileName='Comparison_of_Accuracies_NN_vs_LSTM.png')
    Plot_Metric_vs_Features(num_features_list, loss_nn, loss_lstm, ylabel='Loss', title='Comparison of Losses: NN vs LSTM', fileName='Comparison_of_Losses_NN_vs_LSTM.png')
    Plot_Metric_vs_Features(num_features_list, precision_nn, precision_lstm, ylabel='Precision', title='Comparison of Precisions: NN vs LSTM', fileName='Comparison_of_Precisions_NN_vs_LSTM.png')
    Plot_Metric_vs_Features(num_features_list, roc_nn, roc_lstm, ylabel='ROC AUC', title='Comparison of ROC AUC: NN vs LSTM', fileName='Comparison_of_ROC_AUC_NN_vs_LSTM.png')

    print("Model Evaluation Complete.")

def evaluateTrainedModels():
    # num_features_list = np.load("num_features_list.npy")
    # accuracies_nn = np.load("accuracies_nn.npy")
    # accuracies_lstm = np.load("accuracies_lstm.npy")
    # loss_nn = np.load("loss_nn.npy")
    # loss_lstm = np.load("loss_lstm.npy")
    # precision_nn = np.load("precision_nn.npy")
    # precision_lstm = np.load("precision_lstm.npy")
    # roc_nn = np.load("roc_nn.npy")
    # roc_lstm = np.load("roc_lstm.npy")

    num_features_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130]
    '''
    accuracies_nn = [0.6438, 0.5227, 0.5227, 0.7231, 0.5227, 0.5227, 0.5227, 0.8492, 0.5227, 0.5227]
    accuracies_lstm = [0.6289, 0.6967, 0.7235, 0.7197, 0.7185, 0.7232, 0.8008, 0.8217, 0.8943, 0.9365]
    #loss_nn = [0.6921, 0.6921, 0.6921, 0.5497, 0.3308, 0.6924, 0.6924, 0.3308, 0.6924, 0.6921]
    #loss_lstm = [0.7696, 0.5859, 0.5765, 0.3964, 0.4095, 0.2607, 0.2138, 0.1705, 0.1408, 0.1204]
    precision_nn = [0.6620, 0.5227, 0.5227, 0.7729, 0.5227, 0.5227, 0.5227, 0.8778, 0.5227, 0.5227]
    precision_lstm = [0.6302, 0.6563, 0.7179, 0.6878, 0.7247, 0.7501, 0.7958, 0.8986, 0.9033, 0.9479]
    roc_nn = [0.7159, 0.5, 0.5, 0.8054, 0.5, 0.5, 0.5, 0.9361, 0.5, 0.5]
    roc_lstm = [0.6726, 0.7594, 0.7806, 0.7840, 0.7884, 0.7972, 0.8942, 0.9210, 0.9593, 0.9873]
    '''

    accuracies_nn = [0.643105149269104, 0.7090773582458496, 0.7092013955116272, 0.7065972089767456, 0.7198660969734192, 0.5226934552192688, 0.8020833134651184, 0.8483383059501648, 0.5226934552192688, 0.5226934552192688, 0.9150545597076416, 0.5226934552192688, 0.5226934552192688]

    accuracies_lstm = [0.6009424328804016, 0.6721230149269104, 0.7160218358039856, 0.7073412537574768, 0.7299107313156128, 0.7132936716079712, 0.7926587462425232, 0.8323412537574768, 0.8826884627342224, 0.902529776096344, 0.8851686716079712, 0.9211309552192688, 0.9397321343421936]

    Plot_Metric_vs_Features(num_features_list, accuracies_nn, accuracies_lstm, ylabel='Accuracy', title='Comparison of Accuracies: NN vs LSTM', fileName='Comparison_of_Accuracies_NN_vs_LSTM.png')
    #Plot_Metric_vs_Features(num_features_list, loss_nn, loss_lstm, ylabel='Loss', title='Comparison of Losses: NN vs LSTM', fileName='Comparison_of_Losses_NN_vs_LSTM.png')
    #Plot_Metric_vs_Features(num_features_list, precision_nn, precision_lstm, ylabel='Precision', title='Comparison of Precisions: NN vs LSTM', fileName='Comparison_of_Precisions_NN_vs_LSTM.png')
    #Plot_Metric_vs_Features(num_features_list, roc_nn, roc_lstm, ylabel='ROC AUC', title='Comparison of ROC AUC: NN vs LSTM', fileName='Comparison_of_ROC_AUC_NN_vs_LSTM.png')
    
def main():
    #postgreSqlExplorer()
    #data()
    #train_NN()
    #train_BiLSTM()
    #evaluateModels()
    evaluateTrainedModels()
    
if __name__ == "__main__":
    main()