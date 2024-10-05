import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

import pandas as pd
import numpy as np

class DotA_NN:

    input_variables = ['players_0_hero_id', 'players_1_hero_id', 'players_2_hero_id', 'players_3_hero_id', 'players_4_hero_id', 'players_5_hero_id', 'players_6_hero_id', 'players_7_hero_id', 'players_8_hero_id', 'players_9_hero_id', 'players_0_isRadiant', 'players_1_isRadiant', 'players_2_isRadiant', 'players_3_isRadiant', 'players_4_isRadiant', 'players_5_isRadiant', 'players_6_isRadiant', 'players_7_isRadiant', 'players_8_isRadiant', 'players_9_isRadiant']

    def __init__(self):
        # Set up the strategy for multi-GPU support
        self.strategy = tf.distribute.MirroredStrategy()
        print('Number of devices: {}'.format(self.strategy.num_replicas_in_sync))

    
    def one_hot_encode_heroes(self, X):
        '''
        hero_ids_radiant = ['players_0_hero_id', 'players_1_hero_id', 'players_2_hero_id', 'players_3_hero_id', 'players_4_hero_id']
        hero_ids_dire = ['players_5_hero_id', 'players_6_hero_id', 'players_7_hero_id', 'players_8_hero_id', 'players_9_hero_id']

        encoded_features = pd.DataFrame(index=X.index)

        # Iterate over each hero_id column
        for hero_id_col in hero_id_columns:
            # Extract the column and reshape it to a 2D array
            hero_id_values = X[[hero_id_col]]
            
            # Create and fit a new encoder for this column
            encoder = OneHotEncoder(categories=[range(1, 139)], sparse_output=False, handle_unknown='ignore')
            one_hot_encoded = encoder.fit_transform(hero_id_values)
            
            # Create a DataFrame with the encoded values
            one_hot_encoded_df = pd.DataFrame(one_hot_encoded, index=X.index)
            
            # Rename columns to identify which hero ID they belong to
            one_hot_encoded_df.columns = [f"{hero_id_col}_hero_{i+1}" for i in range(one_hot_encoded_df.shape[1])]
            
            # Concatenate the encoded features with the original DataFrame
            encoded_features = pd.concat([encoded_features, one_hot_encoded_df], axis=1)
        
        '''
        hero_id_columns = [col for col in self.input_variables if 'hero_id' in col]
        
        # Initialize an empty DataFrame to store encoded features
        encoded_features = pd.DataFrame(index=X.index)

        # Iterate over each hero_id column
        for hero_id_col in hero_id_columns:
            # Extract the column and reshape it to a 2D array
            hero_id_values = X[[hero_id_col]]
            
            # Create and fit a new encoder for this column
            encoder = OneHotEncoder(categories=[range(1, 139)], sparse_output=False, handle_unknown='ignore')
            one_hot_encoded = encoder.fit_transform(hero_id_values)
            
            # Create a DataFrame with the encoded values
            one_hot_encoded_df = pd.DataFrame(one_hot_encoded, index=X.index)
            
            # Rename columns to identify which hero ID they belong to
            one_hot_encoded_df.columns = [f"{hero_id_col}_hero_{i+1}" for i in range(one_hot_encoded_df.shape[1])]
            
            # Concatenate the encoded features with the original DataFrame
            encoded_features = pd.concat([encoded_features, one_hot_encoded_df], axis=1)
        

        # Include the non-hero ID features (e.g., 'isRadiant' columns)
        non_hero_columns = [col for col in self.input_variables if 'hero_id' not in col]
        X_encoded = pd.concat([encoded_features, X[non_hero_columns].reset_index(drop=True)], axis=1)
        
        return X_encoded



    def split_data(self, dataset, train_size=0.6):
        # Get the total number of elements in the dataset
        dataset_size = dataset.cardinality().numpy()

        if dataset_size == tf.data.experimental.INFINITE_CARDINALITY:
            raise ValueError("Dataset has infinite cardinality. Make sure it is finite to split.")

        # Compute sizes for train, validation, and test sets
        train_size = int(train_size * dataset_size)
        val_test_size = dataset_size - train_size
        val_size = test_size = val_test_size // 2

        # Shuffle the dataset first to randomize
        dataset = dataset.shuffle(buffer_size=dataset_size, seed=42)

        # Split the dataset: train, validation, and test
        train_dataset = dataset.take(train_size)
        remaining_dataset = dataset.skip(train_size)
        val_dataset = remaining_dataset.take(val_size)
        test_dataset = remaining_dataset.skip(val_size)

        return train_dataset, val_dataset, test_dataset

    def build_model(self, input_size, learning_rate=0.001):
        hidden_layer_fn = 'relu'
        with self.strategy.scope():
            model = Sequential([
                Dense(64, input_dim=input_size, activation=hidden_layer_fn),
                Dense(64, activation=hidden_layer_fn),
                Dense(64, activation=hidden_layer_fn),
                Dense(64, activation=hidden_layer_fn),
                Dense(1, activation='sigmoid')
            ])
            
            model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])

        return model
    
    def plot_accuracy_vs_epochs(self, model_history):
        plt.plot(model_history.history['accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.show()

    def train(self, dataset, train_size, epochs, batch_size, learning_rate):
        print("Splitting data")
        train_dataset, val_dataset, test_dataset = self.split_data(dataset, train_size)
        
        # Prepare datasets for batching and prefetching
        train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        val_dataset = val_dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
        
        model = self.build_model(input_size=next(iter(train_dataset))[0].shape[1], learning_rate=learning_rate)
        
        print("Begin model training")
        model_history = model.fit(train_dataset, epochs=epochs, validation_data=val_dataset)
        
        print("Model training complete")
        # Save the model
        model.save('dota_nn_model.keras')
        print("Model saved as 'dota_nn_model.keras'")

        test_loss, test_accuracy = model.evaluate(test_dataset, verbose=0)
        print('\nTest accuracy:', test_accuracy)
        print('\nLoss:', test_loss)

        self.plot_accuracy_vs_epochs(model_history)