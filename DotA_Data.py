import pandas as pd
import requests
import json
import os
import numpy as np
import tensorflow as tf

#from config import API_KEY

# Use the API key in a URL
#base_url = f"https://api.example.com/data?apikey={API_KEY}"

class DotA_Data:
    def getMatchIDs(self):
        min_rank = 70
        public_matches_endpoint = f"http://api.opendota.com/api/publicMatches?min_rank={min_rank}&api_key={API_KEY}"

        try:
            response = requests.get(public_matches_endpoint)
            response.raise_for_status()  # Will raise an HTTPError for bad responses
            public_matches_data = response.json()
            return public_matches_data
        except requests.RequestException as e:
            print(f"Error retrieving match IDs: {e}")
            return None

    def process_teams_and_one_hot_encode(self, features, hero_id_columns, is_radiant_columns, num_heroes=140):
        # Separate Radiant and Dire teams
        radiant_hero_ids = []
        dire_hero_ids = []

        for i in range(len(hero_id_columns)):
            # Get the actual index for the hero ID and isRadiant columns
            hero_id = features[i]  # Get the hero_id for the current player
            is_radiant = features[len(hero_id_columns) + i]  # Get the corresponding isRadiant value

            # If the player is Radiant
            if is_radiant == 1:
                radiant_hero_ids.append(hero_id)
            else:
                dire_hero_ids.append(hero_id)

        # One-hot encode hero_ids for both teams
        radiant_one_hot = tf.one_hot(radiant_hero_ids, depth=num_heroes)
        dire_one_hot = tf.one_hot(dire_hero_ids, depth=num_heroes)

        # Combine Radiant and Dire one-hot encodings
        one_hot_combined = tf.concat([radiant_one_hot, dire_one_hot], axis=0)
        
        return one_hot_combined

    def create_one_hot_encoded_dataset(self, dataset, num_heroes=140):

        hero_id_columns = ['players_0_hero_id', 'players_1_hero_id', 'players_2_hero_id', 'players_3_hero_id', 'players_4_hero_id', 
        'players_5_hero_id', 'players_6_hero_id', 'players_7_hero_id', 'players_8_hero_id', 'players_9_hero_id']

        is_radiant_columns = ['players_0_isRadiant', 'players_1_isRadiant', 'players_2_isRadiant', 'players_3_isRadiant', 
        'players_4_isRadiant', 'players_5_isRadiant', 'players_6_isRadiant', 'players_7_isRadiant', 'players_8_isRadiant', 
        'players_9_isRadiant']

        # Define a function to map and process each example in the dataset
        def process_example(features, target):
            one_hot_combined = self.process_teams_and_one_hot_encode(features, hero_id_columns, is_radiant_columns, num_heroes)
            return one_hot_combined, target

        # Apply the processing function to each batch in the dataset
        encoded_dataset = dataset.map(lambda features, target: process_example(features, target))
        
        return encoded_dataset
    
    def getMatchData(self, match_id):
        matches_endpoint = f"https://api.opendota.com/api/matches/{match_id}?api_key={API_KEY}"
        try:
            response = requests.get(matches_endpoint)
            response.raise_for_status()
            matches_data = response.json()
            return matches_data
        except requests.RequestException as e:
            print(f"Error retrieving matches data: {e}")
            return None

    def listToJson(self, full_matches, fileName='DotA_data.json'):
        with open(fileName, 'w') as json_file:
            json.dump(full_matches, json_file, indent=4)
    

    def getData(self, no_matches):
        full_matches = []
        processed_match_ids = set()  # To store already processed match_id values
        match_api_calls = 0  # To count how many matches were skipped consecutively
        max_match_api_calls = 200  # Threshold to break the loop after 1000 calls without updating match data
        
        totalMatches = 0
        while totalMatches < no_matches:
            temp_match_ids = self.getMatchIDs()
            match_api_calls += 1
            if temp_match_ids is None:
                break

            df_matches = pd.DataFrame(temp_match_ids)

            for match_id in df_matches.match_id:
                if match_api_calls >= max_match_api_calls:
                    print(f"Terminating loop after {match_api_calls} without new matches found")
                    print("Matches in dataset: ", totalMatches)
                    self.listToJson(full_matches=full_matches, fileName='DotA_data.json')
                    return full_matches                    
                
                # Check if match_id is already processed
                if match_id in processed_match_ids:
                    continue  # Skip this match_id as it's already processed
                

                # Fetch the match data
                df_temp_match = self.getMatchData(match_id)
                if df_temp_match is not None:
                    if df_temp_match['game_mode'] in [22, 1]:
                        if df_temp_match['lobby_type'] in [7, 6, 5]:
                            full_matches.append(df_temp_match)
                            processed_match_ids.add(match_id)  # Mark match_id as processed
                            match_api_calls = 0  # Reset call counter if a new match_id is found

                            totalMatches += 1
                            if(totalMatches%1000 == 0):
                                print("Current matches found", totalMatches)
                else:
                    break

            if(no_matches - totalMatches < 20):
                self.listToJson(full_matches=full_matches, fileName='DotA_data.json')
                print("Matches in dataset: ", totalMatches)
                return full_matches

        self.listToJson(full_matches=full_matches, fileName='DotA_data.json')
        print("Matches in dataset: ", totalMatches)
        
        return full_matches

    def flatten_dictionary(self, d, parent_key='', sep='_'):
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            
            # Skip flattening 'radiant_win' for players, but keep it at the match level
            if parent_key.startswith('players') and k == 'radiant_win':
                continue
            
            if isinstance(v, dict):
                items.extend(self.flatten_dictionary(v, new_key, sep=sep).items())
            elif isinstance(v, list) and all(isinstance(i, dict) for i in v):
                for idx, item in enumerate(v):
                    items.extend(self.flatten_dictionary(item, f"{new_key}{sep}{idx}", sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def flattenDataframe(self):
        with open('postgre_data.json') as f:
            data = json.load(f)
        
        # Process each match and flatten it
        flattened_matches = []
        for match in data:
            flattened_match = self.flatten_dictionary(match)
            flattened_matches.append(flattened_match)

        self.listToJson(flattened_matches, 'flattened_postgre_dota_data.json')
        return flattened_matches


    def jsonToDataframe(self, path):
        data_df = pd.read_json(path)
        return data_df



    def process_chunk(self, chunk, input_variables):
        # Create a copy of input_variables to avoid modifying the original list
        variables = input_variables.copy()

        # Extract necessary columns from the DataFrame chunk
        selected_columns = chunk[variables]
        
        # Remove 'radiant_win' from the variables list for features
        variables.remove('radiant_win')

        # Convert to NumPy arrays (ensure proper numeric type)
        features = selected_columns[variables].apply(pd.to_numeric, errors='coerce').fillna(0).astype('float32').values
        target = selected_columns['radiant_win'].astype('int32').values  # Ensure target is integer

        # Convert to TensorFlow dataset
        dataset = tf.data.Dataset.from_tensor_slices((features, target))
        return dataset


        '''
        # Extract necessary columns from the DataFrame chunk
        selected_columns = chunk[input_variables]
        
        input_variables.remove('radiant_win')

        # Convert to NumPy arrays (ensure proper numeric type)
        features = selected_columns[input_variables].apply(pd.to_numeric, errors='coerce').fillna(0).astype('float32').values
        target = selected_columns['radiant_win'].astype('int32').values  # Ensure target is integer

        # Convert to TensorFlow dataset
        dataset = tf.data.Dataset.from_tensor_slices((features, target))
        return dataset
        '''

    def getDataset(self, json_file_path, chunksize=10000, input_variables=[]):
        # Initialize an empty dataset to store concatenated datasets
        final_dataset = None
        total_rows = 0

        print("Loading data from json file")
        with open(json_file_path, 'r') as f:
            data = json.load(f)  # Load the entire JSON as a list of dictionaries (matches)

        print("Processing data in chunks to build TensorFlow dataset")
        for chunk in np.array_split(data, len(data) // chunksize + 1):
            # Normalize the chunk of JSON data into a DataFrame
            chunk_df = pd.json_normalize(chunk)

            # Process the chunk and extract necessary columns
            chunk_dataset = self.process_chunk(chunk_df, input_variables)
            total_rows += chunk_df.shape[0]
            
            if final_dataset is None:
                final_dataset = chunk_dataset  # Initialize with the first chunk
            else:
                final_dataset = final_dataset.concatenate(chunk_dataset)  # Concatenate new chunk
        
        print(f"final_dataset size: {final_dataset.cardinality().numpy()}")

        # Shuffle, batch, and prefetch the dataset to optimize for training
        #final_dataset = final_dataset.shuffle(buffer_size=total_rows).batch(32).prefetch(tf.data.experimental.AUTOTUNE)
        print("Final dataset completed and shuffled")
        return final_dataset, total_rows
