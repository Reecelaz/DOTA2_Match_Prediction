import pandas as pd
import requests

class DotA_Data:
    def __init__(self, features):
        self.features = features

    def getData(self):
        json_file_path = 'C:/Users/Reece Lazarus/DOTA2_Match_Prediction/data.json'
        df_full_matches = pd.DataFrame()
        df_matches = pd.read_json(json_file_path, orient='records')
        for match_id in df_matches.match_id:
            df_temp_match = self.getMatchData(match_id)
            if df_temp_match is not None:
                df_full_matches = pd.concat([df_full_matches, df_temp_match], ignore_index=True)
        return df_full_matches
    
    def getMatchData(self, match_id):
        url = f"https://api.opendota.com/api/matches/{match_id}"
        response = requests.get(url)
        if response.status_code == 200:
            match_data = response.json()
            return pd.DataFrame([match_data])
        else:
            return None