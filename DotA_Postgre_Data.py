import pandas as pd
import requests
import json
import os
import time

from config import API_KEY

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
        #matches_endpoint = f"https://api.opendota.com/api/matches/{match_id}?api_key={API_KEY}"
        matches_endpoint = f"https://api.opendota.com/api/matches/{match_id}"
        try:
            response = requests.get(matches_endpoint)
            response.raise_for_status()
            matches_data = response.json()
            return matches_data
        except requests.RequestException as e:
            print(f"Error retrieving matches data: {e}")
            return None
    
    def jsonToDataframe(self, path):
        data_df = pd.read_json(path)
        return data_df
    
    def listToJson(self, full_matches, fileName):
        with open(fileName, 'w') as json_file:
            json.dump(full_matches, json_file, indent=4)

    def getData(self, no_matches=1, rate_limit=60):
        
        print("Fetching match_ids")
        json_file_path = 'C:/Users/Reece Lazarus/DOTA2_Match_Prediction/postgre_match_id_data.json'
        #json_file_path = '/home-mscluster/rlazarus/DotA_data.json/postgre_match_id_data.json'
        match_ids_df = self.jsonToDataframe(json_file_path)

        full_matches = []
        unique_match_ids = set()
        matches_processed = 0
        requests_made = 0

        print("Collecting individual match data")

        for index, row in match_ids_df.iterrows():
            match_id = row['match_id']

            # Skip if match_id is already processed
            if match_id in unique_match_ids:
                continue

            print(f"Processing match_id: {match_id+1}", f"; {matches_processed} / {no_matches}")
            temp_match = self.getMatchData(match_id)
            if temp_match is not None:
                full_matches.append(temp_match)
                requests_made += 1

                # Rate limit check
                if requests_made >= rate_limit:
                    print(f"Reached {rate_limit} API calls. Sleeping for 60 seconds...")
                    time.sleep(60)  # Sleep for 60 seconds before making more requests
                    requests_made = 0  # Reset the counter after sleeping
            
            # Add match_id to the set to ensure uniqueness
            unique_match_ids.add(match_id)
            
            # Increment the processed match counter
            matches_processed += 1
            
            # Stop after processing the specified number of unique matches
            if matches_processed >= no_matches:
                break

        self.listToJson(full_matches, fileName='postgre_data.json')

        print("Individual match data collection complete. File name: postgre_data.json")
        print(f"Total Matches in dataset: {len(full_matches)}")

        return full_matches

    # Write function to loop through match_ids and get the data for each.
    # Dont forget rate limiting. 
    # Handle errors to that if it breaks you don't lose the data. 
    # Keep the code to check for duplicate match_ids.