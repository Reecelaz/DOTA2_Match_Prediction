import pandas as pd
import requests
import json
import os

from config import API_KEY

class DotA_Postgre_Data:
    def getData(self, no_matches):
        matches_endpoint = f"https://api.opendota.com/api/matches/{match_id}"
        try:
            response = requests.get(matches_endpoint)
            response.raise_for_status()
            matches_data = response.json()
            return matches_data
        except requests.RequestException as e:
            print(f"Error retrieving matches data: {e}")
            return None