import pandas as pd
import requests
import json
import os

#from config import API_KEY

class DotA_Postgre_Data:
    def getMatchIDs(self, no_matches=500000):
        print("Postgre Sql Query Submitted")
        url = "https://api.opendota.com/api/explorer"
        params = {
            "sql": f"SELECT * FROM public_matches WHERE (game_mode = 1 OR game_mode = 22) AND (lobby_type = 7 OR lobby_type = 6 OR lobby_type = 5) AND duration > 600 LIMIT {no_matches}"
        }

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            matches_data = response.json()
            print("Match data collected")

            with open('postgre_data.json', 'w') as json_file:
                json.dump(matches_data, json_file, indent=4)
            print("Json dump created")
            return matches_data
        except requests.RequestException as e:
            print(f"Error retrieving matches data: {e}")
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

    # Write function to loop through match_ids and get the data for each.
    # Dont forget rate limiting. 
    # Handle errors to that if it breaks you don't lose the data. 
    # Keep the code to check for duplicate match_ids.