import ijson
import pandas as pd
import tensorflow as tf
import numpy as np
from collections import defaultdict

class DotA_Data:
    def __init__(self):
        # Initialize dictionaries to store synergy and countering stats
        self.synergy_dict = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # [win_count, total_count]
        self.countering_dict = defaultdict(lambda: defaultdict(lambda: [0, 0]))  # [win_count, total_count]

    def calculate_synergy_countering_features(self, team_heroes, match_won):
        # Update synergy and countering statistics
        for i, hero_i in enumerate(team_heroes):
            for j in range(i + 1, len(team_heroes)):
                hero_j = team_heroes[j]
                self.synergy_dict[hero_i][hero_j][1] += 1
                if match_won:
                    self.synergy_dict[hero_i][hero_j][0] += 1

    def get_synergy_score(self, team_heroes):
        # Calculate synergy score for a team based on synergy_dict
        score = 0
        for i, hero_i in enumerate(team_heroes):
            for j in range(i + 1, len(team_heroes)):
                hero_j = team_heroes[j]
                wins, total = self.synergy_dict[hero_i][hero_j]
                if total > 0:
                    score += wins / total
        return score

    def get_countering_score(self, radiant_heroes, dire_heroes):
        # Calculate countering score between Radiant and Dire teams based on countering_dict
        score = 0
        for hero_radiant in radiant_heroes:
            for hero_dire in dire_heroes:
                wins, total = self.countering_dict[hero_radiant][hero_dire]
                if total > 0:
                    score += wins / total
        return score

    def calculate_items_and_activations(self, player_data):
        # Calculate the total items completed and activations used
        items_completed = sum(1 for item in range(6) if player_data[f'item_{item}'] > 0)
        item_activations = player_data.get('item_activation_count', 0)
        return items_completed, item_activations

    def process_teams_and_one_hot_encode(self, features, hero_id_columns, is_radiant_columns, extra_column_indices, num_heroes=140):
        # Initialize zero-filled vectors for Radiant and Dire teams
        radiant_vector = np.zeros(num_heroes)
        dire_vector = np.zeros(num_heroes)
        radiant_heroes = []
        dire_heroes = []

        for i, hero_id in enumerate(hero_id_columns):
            hero_id_val = int(features[i])  # Ensure hero ID is an integer
            is_radiant = int(features[len(hero_id_columns) + i])

            # Populate hero team vectors
            if is_radiant == 1:
                radiant_vector[hero_id_val] = 1
                radiant_heroes.append(hero_id_val)
            else:
                dire_vector[hero_id_val] = 1
                dire_heroes.append(hero_id_val)

        match_vector = np.concatenate([radiant_vector, dire_vector])

        # Calculate synergy and countering scores
        synergy_score = self.get_synergy_score(radiant_heroes)
        countering_score = self.get_countering_score(radiant_heroes, dire_heroes)

        # Append synergy and countering scores
        match_vector = np.concatenate([match_vector, [synergy_score, countering_score]])

        # Collect extra features (e.g., items and activations)
        extra_features = []
        for extra_col in extra_column_indices:
            if extra_col < len(features):
                extra_features.append(float(features[extra_col]))
            else:
                extra_features.append(0.0)

        # Add extra features to match_vector
        match_vector = np.concatenate([match_vector, extra_features])
        return match_vector

    def create_one_hot_encoded_chunk(self, chunk_df, hero_id_columns, is_radiant_columns, extra_columns, target_column='radiant_win', num_heroes=140):
        extra_column_indices = [chunk_df.columns.get_loc(col) for col in extra_columns]
        feature_vectors = []
        targets = []

        for _, row in chunk_df.iterrows():
            features = row[hero_id_columns + is_radiant_columns + extra_columns].values
            target = row[target_column]

            # Process one-hot encoding and additional features
            feature_vector = self.process_teams_and_one_hot_encode(features, hero_id_columns, is_radiant_columns, extra_column_indices, num_heroes)
            feature_vectors.append(feature_vector)
            targets.append(target)

        feature_vectors = np.array(feature_vectors, dtype=np.float32)
        targets = np.array(targets, dtype=np.int32)

        return tf.data.Dataset.from_tensor_slices((feature_vectors, targets))

    def prepare_dataset(self, dataset_path, hero_id_columns, is_radiant_columns, extra_columns, target_column='radiant_win', chunksize=500):
        final_dataset = None
        print("Processing data in chunks to build TensorFlow dataset")

        with open(dataset_path, 'r') as f:
            objects = ijson.items(f, 'item')
            chunk = []

            for i, obj in enumerate(objects):
                chunk.append(obj)

                if len(chunk) == chunksize:
                    chunk_df = pd.json_normalize(chunk)
                    chunk_dataset = self.create_one_hot_encoded_chunk(chunk_df, hero_id_columns, is_radiant_columns, extra_columns, target_column)

                    if final_dataset is None:
                        final_dataset = chunk_dataset
                    else:
                        final_dataset = final_dataset.concatenate(chunk_dataset)
                    
                    chunk = []

            if chunk:
                chunk_df = pd.json_normalize(chunk)
                chunk_dataset = self.create_one_hot_encoded_chunk(chunk_df, hero_id_columns, is_radiant_columns, extra_columns, target_column)

                if final_dataset is None:
                    final_dataset = chunk_dataset
                else:
                    final_dataset = final_dataset.concatenate(chunk_dataset)

        return final_dataset


# import ijson
# import pandas as pd
# import tensorflow as tf
# import numpy as np

# class DotA_Data:
    
#     def process_teams_and_one_hot_encode(self, features, hero_id_columns, is_radiant_columns, extra_column_indices, num_heroes=140):
#         # Initialize zero-filled vectors for Radiant and Dire teams
#         radiant_vector = np.zeros(num_heroes)
#         dire_vector = np.zeros(num_heroes)

#         for i in range(len(hero_id_columns)):
#             hero_id = int(features[i])  # Ensure hero ID is an integer
#             is_radiant = int(features[len(hero_id_columns) + i])
            
#             # Set the corresponding position in the vector based on team
#             if is_radiant == 1:
#                 radiant_vector[hero_id] = 1
#             else:
#                 dire_vector[hero_id] = 1

#         # Concatenate the two vectors to form a 280-length vector
#         match_vector = np.concatenate([radiant_vector, dire_vector])

#         # Collect extra features, ensuring consistency in length
#         extra_features = [float(features[extra_col]) if extra_col < len(features) else 0.0 for extra_col in extra_column_indices]

#         # Convert extra_features to numpy array and concatenate with match_vector
#         extra_features = np.array(extra_features, dtype=np.float32)
#         match_vector = np.concatenate([match_vector, extra_features])

#         return match_vector



#     def create_one_hot_encoded_chunk(self, chunk_df, hero_id_columns, is_radiant_columns, extra_columns, target_column='radiant_win', num_heroes=140):
#         extra_column_indices = [chunk_df.columns.get_loc(col) for col in extra_columns]

#         feature_vectors = []
#         targets = []

#         for _, row in chunk_df.iterrows():
#             features = row[hero_id_columns + is_radiant_columns + extra_columns].values
#             target = row[target_column]
            
#             # Ensure consistent length by using the modified function
#             feature_vector = self.process_teams_and_one_hot_encode(features, hero_id_columns, is_radiant_columns, extra_column_indices, num_heroes)
#             feature_vectors.append(feature_vector)
#             targets.append(target)

#         # Convert lists to numpy arrays with consistent shape for TensorFlow
#         feature_vectors = np.array(feature_vectors, dtype=np.float32)
#         targets = np.array(targets, dtype=np.int32)

#         return tf.data.Dataset.from_tensor_slices((feature_vectors, targets))



#     def prepare_dataset(self, dataset_path, hero_id_columns, is_radiant_columns, extra_columns, target_column='radiant_win', chunksize=500):
#         final_dataset = None
#         print("Processing data in chunks to build TensorFlow dataset")

#         with open(dataset_path, 'r') as f:
#             # Use ijson to stream the JSON data, this avoids loading the entire file into memory at once
#             objects = ijson.items(f, 'item')  # Adjust 'item' based on your JSON structure
            
#             chunk = []
#             for i, obj in enumerate(objects):
#                 chunk.append(obj)
                
#                 if len(chunk) == chunksize:
#                     # Once chunk size is met, process and clear the chunk
#                     chunk_df = pd.json_normalize(chunk)
#                     chunk_dataset = self.create_one_hot_encoded_chunk(chunk_df, hero_id_columns, is_radiant_columns, extra_columns, target_column)
                    
#                     # Concatenate the chunk dataset with the final dataset
#                     if final_dataset is None:
#                         final_dataset = chunk_dataset
#                     else:
#                         final_dataset = final_dataset.concatenate(chunk_dataset)
                    
#                     chunk = []  # Clear the chunk for the next batch

#             # Process any remaining data after the loop
#             if chunk:
#                 chunk_df = pd.json_normalize(chunk)
#                 chunk_dataset = self.create_one_hot_encoded_chunk(chunk_df, hero_id_columns, is_radiant_columns, extra_columns, target_column)
                
#                 if final_dataset is None:
#                     final_dataset = chunk_dataset
#                 else:
#                     final_dataset = final_dataset.concatenate(chunk_dataset)

#         return final_dataset