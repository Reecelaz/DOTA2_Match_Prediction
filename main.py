from DotA_Data import DotA_Data
from DotA_NN import DotA_NN
from pprint import pprint
import json

def main():
    features = ["match_id", "radiant_win"]
    no_matches = 2

    matches = DotA_Data(features)
    full_matches_df = matches.flattenDataframe()
    '''
    with open('flattened_dota_data.json', 'w') as json_file:
        json.dump(full_matches_df, json_file, indent=4)
    '''
    nn = DotA_NN()
    nn.train(full_matches_df)
    
if __name__ == "__main__":
    main()