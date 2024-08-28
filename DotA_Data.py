import pandas as pd
import requests
import json

class DotA_Data:
    def __init__(self, features):
        self.features = features

    def getMatchIDs(self):
        min_rank = 70
        public_matches_endpoint = f"http://api.opendota.com/api/publicMatches?min_rank={min_rank}"

        response = requests.get(public_matches_endpoint)

        # Check if the request was successful
        if response.status_code == 200:
            # Parse the JSON response
            public_matches_data = response.json()
            return public_matches_data
        else:
            print(f"Failed to retrieve data: {response.status_code}")
            return

    def getData(self, no_matches):
        full_matches = []
        
        totalMatches = 0
        while(totalMatches < no_matches):
            temp_match_ids = self.getMatchIDs()

            df_matches = pd.DataFrame(temp_match_ids)
            for match_id in df_matches.match_id:
                df_temp_match = self.getMatchData(match_id)
                if df_temp_match is not None:
                    if(df_temp_match['game_mode'] == 22 or df_temp_match['game_mode'] == 1):
                        if(df_temp_match['lobby_type'] == 7 or df_temp_match['lobby_type'] == 6 or df_temp_match['lobby_type'] == 5):
                            full_matches.append(df_temp_match)
                            totalMatches += 1

        with open('DotA_data.json', 'w') as json_file:
            json.dump(full_matches, json_file, indent=4)
        return full_matches
    
    def getMatchData(self, match_id):
        url = f"https://api.opendota.com/api/matches/{match_id}"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            return None

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
        with open('DotA_data.json') as f:
            data = json.load(f)
        
        # Process each match and flatten it
        flattened_matches = []
        for match in data:
            flattened_match = self.flatten_dictionary(match)
            flattened_matches.append(flattened_match)

        # Convert the list of flattened matches into a DataFrame (if needed)
        df = pd.DataFrame(flattened_matches)
        return flattened_matches


    def jsonToDataframe(self, path):
        data_df = pd.read_json(path)
        return data_df