from DotA_Data import DotA_Data
from DotA_NN import DotA_NN
from pprint import pprint
import json
import pandas as pd

def main():
    features = ["match_id", "radiant_win"]
    no_matches = 1

    matches = DotA_Data(features)
    
    temp_matches = matches.getData(no_matches)
    with open('DotA_data.json', 'w') as json_file:
        json.dump(temp_matches, json_file, indent=4)   
    
    full_matches_df = matches.flattenDataframe()
    with open('flattened_dota_data.json', 'w') as json_file:
        json.dump(full_matches_df, json_file, indent=4)
    
    nn = DotA_NN()
    #nn.train(full_matches_df)
    
    '''
    json_file_path = 'C:/Users/Reece Lazarus/DOTA2_Match_Prediction/flattened_dota_data.json'
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    # Convert the JSON data to a pandas DataFrame
    df = pd.DataFrame(data)
    print(len(df))
    '''

if __name__ == "__main__":
    main()