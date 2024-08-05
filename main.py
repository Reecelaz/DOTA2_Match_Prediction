from DotA_Data import DotA_Data

def main():
    features = ["match_id", "radiant_win"]
    matches = DotA_Data(features)
    full_matches = matches.getData()
    print(full_matches)

if __name__ == "__main__":
    main()