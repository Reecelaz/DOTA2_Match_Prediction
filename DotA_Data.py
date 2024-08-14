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
        json_file_path = 'C:/Users/Reece Lazarus/DOTA2_Match_Prediction/data.json'
        df_full_matches = []
        
        totalMatches = 0
        while(totalMatches < no_matches):
            temp_match_ids = self.getMatchIDs()

            df_matches = pd.DataFrame(temp_match_ids)
            for match_id in df_matches.match_id:
                df_temp_match = self.getMatchData(match_id)
                if df_temp_match is not None:
                    df_full_matches.append(df_temp_match)
                    totalMatches += 1

        with open('DotA_data.json', 'w') as json_file:
            json.dump(df_full_matches, json_file, indent=4)
        return df_full_matches
    
    def getMatchData(self, match_id):
        url = f"https://api.opendota.com/api/matches/{match_id}"
        response = requests.get(url)
        if response.status_code == 200:
            return response.json()
        else:
            return None

    def flattenDataframe(self):
        with open('DotA_data.json') as f:
            data = json.load(f)
        
        flattened_data = []

        for match in data:
            # Start with the match-level data
            match_flat = {k: v for k, v in match.items() if k not in ['players', 'picks_bans', 'od_data']}
            
            # Flatten picks_bans
            if 'picks_bans' in match:
                picks_bans = match['picks_bans']
                for i, pick_ban in enumerate(picks_bans):
                    for key, value in pick_ban.items():
                        match_flat[f'picks_bans_{i}_{key}'] = value

            # Flatten od_data
            if 'od_data' in match and isinstance(match['od_data'], dict):
                for key, value in match['od_data'].items():
                    match_flat[f'od_data_{key}'] = value
            
            # Flatten players
            if 'players' in match:
                players = match['players']
                for i, player in enumerate(players):
                    player_flat = match_flat.copy()  # Copy match-level data
                    for key, value in player.items():
                        if isinstance(value, list):
                            for j, item in enumerate(value):
                                player_flat[f'{key}_{j}'] = item
                        elif isinstance(value, dict):
                            for sub_key, sub_value in value.items():
                                if isinstance(sub_value, dict):
                                    for sub_sub_key, sub_sub_value in sub_value.items():
                                        player_flat[f'{key}_{sub_key}_{sub_sub_key}'] = sub_sub_value
                                else:
                                    player_flat[f'{key}_{sub_key}'] = sub_value
                        else:
                            player_flat[f'{key}'] = value
                    
                    flattened_data.append(player_flat)

        # Convert to DataFrame
        data_df = pd.DataFrame(flattened_data)
        return data_df


    def jsonToDataframe(self):
        data_df = pd.read_json('C:/Users/Reece Lazarus/DOTA2_Match_Prediction/DotA_data.json')
        return data_df