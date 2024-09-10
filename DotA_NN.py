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



    def preprocess_data(self, data_df, train_size = 0.6):
        X = data_df[self.input_variables]
        y = data_df['radiant_win']

        # Convert boolean variables to int for use with TensorFlow
        X = X.astype({col: 'int' for col in X.select_dtypes(include='bool').columns})
        y = y.astype('int')

        X = self.one_hot_encode_heroes(X)
        
        # First split: train_size and temp set
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=1-train_size, stratify=y, random_state=42)
        
        # Second split: 50% val, 50% test from the temp set
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

        # Convert to numpy arrays
        X_train = X_train.values
        X_val = X_val.values
        X_test = X_test.values
        y_train = y_train.values
        y_val = y_val.values
        y_test = y_test.values
        
        return X_train, y_train, X_val, y_val, X_test, y_test

    def build_model(self, input_size):
        # Build the model within the strategy scope
        with self.strategy.scope():
            model = Sequential([
                Dense(64, input_dim=input_size, activation='relu'),
                Dense(64, activation='relu'),
                Dense(64, activation='relu'),
                Dense(64, activation='relu'),
                Dense(1, activation='sigmoid')
            ])
            
            model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

        return model
    
    def plot_accuracy_vs_epochs(self, model_history):
        plt.plot(model_history.history['accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.show()

    def train(self, data_df, train_size, epochs, batch_size):
        X_train, y_train, X_val, y_val, X_test, y_test = self.preprocess_data(data_df, train_size)
        
        model = self.build_model(X_train.shape[1])
        
        model_history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val))
        #model.summary()

        # Save the model
        model.save('dota_nn_model.keras')
        print("Model training complete and saved as 'dota_nn_model.keras'")

        test_loss, test_accuracy = model.evaluate(X_test,  y_test, verbose = 0)

        print('\nTest accuracy:', test_accuracy)
        print('\nLoss:', test_loss)

        self.plot_accuracy_vs_epochs(model_history)