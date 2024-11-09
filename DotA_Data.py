import ijson
import pandas as pd
import tensorflow as tf
import numpy as np

class DotA_Data:
    
    def process_teams_and_one_hot_encode(self, features, hero_id_columns, is_radiant_columns, extra_column_indices, num_heroes=140):
        # Initialize zero-filled vectors for Radiant and Dire teams
        radiant_vector = np.zeros(num_heroes)
        dire_vector = np.zeros(num_heroes)

        for i in range(len(hero_id_columns)):
            hero_id = int(features[i])  # Ensure hero ID is an integer
            is_radiant = int(features[len(hero_id_columns) + i])
            
            # Set the corresponding position in the vector based on team
            if is_radiant == 1:
                radiant_vector[hero_id] = 1
            else:
                dire_vector[hero_id] = 1

        # Concatenate the two vectors to form a 280-length vector
        match_vector = np.concatenate([radiant_vector, dire_vector])

        # Collect extra features, ensuring consistency in length
        extra_features = [float(features[extra_col]) if extra_col < len(features) else 0.0 for extra_col in extra_column_indices]

        # Convert extra_features to numpy array and concatenate with match_vector
        extra_features = np.array(extra_features, dtype=np.float32)
        match_vector = np.concatenate([match_vector, extra_features])

        return match_vector



    def create_one_hot_encoded_chunk(self, chunk_df, hero_id_columns, is_radiant_columns, extra_columns, target_column='radiant_win', num_heroes=140):
        extra_column_indices = [chunk_df.columns.get_loc(col) for col in extra_columns]

        feature_vectors = []
        targets = []

        for _, row in chunk_df.iterrows():
            features = row[hero_id_columns + is_radiant_columns + extra_columns].values
            target = row[target_column]
            
            # Ensure consistent length by using the modified function
            feature_vector = self.process_teams_and_one_hot_encode(features, hero_id_columns, is_radiant_columns, extra_column_indices, num_heroes)
            feature_vectors.append(feature_vector)
            targets.append(target)

        # Convert lists to numpy arrays with consistent shape for TensorFlow
        feature_vectors = np.array(feature_vectors, dtype=np.float32)
        targets = np.array(targets, dtype=np.int32)

        return tf.data.Dataset.from_tensor_slices((feature_vectors, targets))



    def prepare_dataset(self, dataset_path, hero_id_columns, is_radiant_columns, extra_columns, target_column='radiant_win', chunksize=500):
        final_dataset = None
        print("Processing data in chunks to build TensorFlow dataset")

        with open(dataset_path, 'r') as f:
            # Use ijson to stream the JSON data, this avoids loading the entire file into memory at once
            objects = ijson.items(f, 'item')  # Adjust 'item' based on your JSON structure
            
            chunk = []
            for i, obj in enumerate(objects):
                chunk.append(obj)
                
                if len(chunk) == chunksize:
                    # Once chunk size is met, process and clear the chunk
                    chunk_df = pd.json_normalize(chunk)
                    chunk_dataset = self.create_one_hot_encoded_chunk(chunk_df, hero_id_columns, is_radiant_columns, extra_columns, target_column)
                    
                    # Concatenate the chunk dataset with the final dataset
                    if final_dataset is None:
                        final_dataset = chunk_dataset
                    else:
                        final_dataset = final_dataset.concatenate(chunk_dataset)
                    
                    chunk = []  # Clear the chunk for the next batch

            # Process any remaining data after the loop
            if chunk:
                chunk_df = pd.json_normalize(chunk)
                chunk_dataset = self.create_one_hot_encoded_chunk(chunk_df, hero_id_columns, is_radiant_columns, extra_columns, target_column)
                
                if final_dataset is None:
                    final_dataset = chunk_dataset
                else:
                    final_dataset = final_dataset.concatenate(chunk_dataset)

        return final_dataset