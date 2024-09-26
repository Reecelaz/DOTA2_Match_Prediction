import opendota
import json
import pandas as pd

'''
#print(client.get_schema()) # Lists all tables
# print("##########################")
print(client.get_schema('matches'))

matches = client.explorer("select * from public_matches")
'''
# Initialize the API-connection object
client = opendota.OpenDota()

matches = client.explorer("select * from matches where game_mode = 1 or game_mode = 22 and lobby_type = 7 or lobby_type = 6 OR lobby_type = 5 and duration > 600") # AND (duration > 600)
#matches = client.explorer("select * from matches where game_mode = 1 OR game_mode = 22 AND lobby_type = 7")

with open('opendota_matches.json', 'w') as json_file:
    json.dump(matches, json_file, indent=4)

path = 'C:/Users/Reece Lazarus/DOTA2_Match_Prediction/opendota_matches.json'
matches_df = pd.read_json(path)
print("Total matches in dataset: ", matches_df.shape[0])
