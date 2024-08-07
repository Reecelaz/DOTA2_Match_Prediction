from DotA_Data import DotA_Data
from DotA_NN import DotA_NN
from pprint import pprint

def main():
    features = ["match_id", "radiant_win"]
    no_matches = 2

    matches = DotA_Data(features)
    full_matches_df = matches.jsonToDataframe()
    nn = DotA_NN()
    nn.train(full_matches_df)

if __name__ == "__main__":
    main()