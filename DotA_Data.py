import pandas as pd
import requests
import json
import os

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



    def process_chunk(chunk, input_variables):
        # Extract necessary columns from the DataFrame chunk
        selected_columns = chunk[input_variables]
        
        input_variables.remove('radiant_win')

        # Convert to NumPy arrays
        features = selected_columns[input_variables].values
        target = selected_columns['radiant_win'].values
        
        # Convert to TensorFlow dataset
        dataset = tf.data.Dataset.from_tensor_slices((features, target))
        return dataset

    def getDataset(json_file_path, chunksize=10000, input_variables=[]):
        # Initialize an empty dataset to store concatenated datasets
        final_dataset = None

        # Load and process the JSON data in chunks
        json_file = json_file_path
        chunks = pd.read_json(json_file, lines=True, chunksize=chunksize)  # Adjust chunksize as needed

        for chunk in chunks:
            chunk_dataset = process_chunk(chunk)
            
            if final_dataset is None:
                final_dataset = chunk_dataset  # Initialize with the first chunk
            else:
                final_dataset = final_dataset.concatenate(chunk_dataset)  # Concatenate new chunk

        # Shuffle, batch, and prefetch the dataset to optimize for training
        final_dataset = final_dataset.shuffle(buffer_size=chunksize).batch(32).prefetch(tf.data.experimental.AUTOTUNE)
        return final_dataset