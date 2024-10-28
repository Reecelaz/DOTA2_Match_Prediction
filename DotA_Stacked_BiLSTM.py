import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Dropout, TimeDistributed

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class DotA_Stacked_BiLSTM:

    def __init__(self):
        # Set up the strategy for multi-GPU support
        self.strategy = tf.distribute.MirroredStrategy()
        print('Number of devices: {}'.format(self.strategy.num_replicas_in_sync))
    '''
    def build_model(self, input_size):
        with self.strategy.scope():
            # Build the Stacked Bidirectional LSTM Model with dynamic input dimensions
            model = Sequential()

            # Add first Bidirectional LSTM layer (timesteps is dynamic)
            model.add(Bidirectional(LSTM(140, return_sequences=True), input_shape=(None, input_size)))
            model.add(Dropout(0.2))

            # Add second Bidirectional LSTM layer
            model.add(Bidirectional(LSTM(140, return_sequences=True)))  # Keep return_sequences=True to retain sequence data
            model.add(Dropout(0.2))

            # Use TimeDistributed to apply Dense to each time step in the sequence
            model.add(TimeDistributed(Dense(64, activation='relu')))  # Apply Dense layer to each time step

            # Add a Dense output layer for binary classification
            model.add(TimeDistributed(Dense(1, activation='sigmoid')))  # Sigmoid for binary output across timesteps

            # Compile the model
            model.compile(optimizer='adam',
                        loss='binary_crossentropy',  # Binary classification loss
                        metrics=['accuracy'])

        # Print the model summary
        model.summary()

        return model
    '''
    def build_model(self, input_shape):
        '''
        with self.strategy.scope():
            model = Sequential()

            # Add a Dense layer to reduce input dimensions (e.g., 64 units)
            model.add(Dense(64, activation='relu', input_shape=input_shape))

            # Add first Bidirectional LSTM layer
            model.add(Bidirectional(LSTM(64, return_sequences=True)))
            model.add(Dropout(0.2))

            # Add second Bidirectional LSTM layer, with return_sequences=False for single output
            model.add(Bidirectional(LSTM(64, return_sequences=False)))  # No need to keep sequence data here
            model.add(Dropout(0.2))

            # Final Dense layer for binary classification
            model.add(Dense(1, activation='sigmoid'))  # Single sigmoid output

            # Compile the model
            model.compile(optimizer='adam',
                        loss='binary_crossentropy',
                        metrics=['accuracy'])

        # Print the model summary
        model.summary()

        return model
        '''
        with self.strategy.scope():
            model = Sequential()

            # Use LSTM units that match the input feature size dynamically
            input_feature_size = input_shape[-1]  # Get the feature dimension from the input shape

            # Add first Bidirectional LSTM layer with units matching input feature size
            model.add(Bidirectional(LSTM(input_feature_size, return_sequences=True), input_shape=input_shape))
            model.add(Dropout(0.01))

            # Add second Bidirectional LSTM layer
            model.add(Bidirectional(LSTM(input_feature_size, return_sequences=False)))  # No need to keep sequence data here
            model.add(Dropout(0.01))

            # Final Dense layer for binary classification
            model.add(Dense(1, activation='sigmoid'))  # Single sigmoid output for binary classification

            # Compile the model
            model.compile(optimizer='adam',
                        loss='binary_crossentropy',
                        metrics=['accuracy'])

        # Print the model summary
        model.summary()

        return model


    def split_data(self, dataset, train_size=0.6):
        print(f"Original dataset size: {dataset.cardinality().numpy()}")
        # Get the total number of elements in the dataset
        dataset_size = dataset.cardinality().numpy()

        if dataset_size == 0:
            raise ValueError("The dataset is empty")

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

        # Print sizes for verification
        print(f"Train dataset size: {train_dataset.cardinality().numpy()}")
        print(f"Val dataset size: {val_dataset.cardinality().numpy()}")
        print(f"Test dataset size: {test_dataset.cardinality().numpy()}")

        return train_dataset, val_dataset, test_dataset

    def process_teams_and_one_hot_encode(self, features, hero_id_columns, is_radiant_columns, num_heroes=140):
        # Separate Radiant and Dire teams
        radiant_hero_ids = []
        dire_hero_ids = []

        for i in range(len(hero_id_columns)):
            # Get the actual index for the hero ID and isRadiant columns
            hero_id = features[i]  # Get the hero_id for the current player
            is_radiant = features[len(hero_id_columns) + i]  # Get the corresponding isRadiant value

            # If the player is Radiant
            if is_radiant == 1:
                radiant_hero_ids.append(hero_id)
            else:
                dire_hero_ids.append(hero_id)

        # One-hot encode hero_ids for both teams
        radiant_one_hot = tf.one_hot(radiant_hero_ids, depth=num_heroes)
        dire_one_hot = tf.one_hot(dire_hero_ids, depth=num_heroes)

        # Combine Radiant and Dire one-hot encodings
        one_hot_combined = tf.concat([radiant_one_hot, dire_one_hot], axis=0)
        
        return one_hot_combined

    def create_one_hot_encoded_dataset(self, dataset, hero_id_columns, is_radiant_columns, num_heroes=140):
        # Define a function to map and process each example in the dataset
        def process_example(features, target):
            one_hot_combined = self.process_teams_and_one_hot_encode(features, hero_id_columns, is_radiant_columns, num_heroes)
            return one_hot_combined, target

        # Apply the processing function to each batch in the dataset
        encoded_dataset = dataset.map(lambda features, target: process_example(features, target))
        
        return encoded_dataset

    def plot_accuracy_vs_epochs(self, model_history):
        plt.plot(model_history.history['accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.savefig('LSTM accuracy_vs_epochs.png')

    def train(self, dataset, train_size, epochs, batch_size):
        print("Encoding hero columns")

        hero_id_columns = ['players_0_hero_id', 'players_1_hero_id', 'players_2_hero_id', 'players_3_hero_id', 'players_4_hero_id', 
        'players_5_hero_id', 'players_6_hero_id', 'players_7_hero_id', 'players_8_hero_id', 'players_9_hero_id']

        is_radiant_columns = ['players_0_isRadiant', 'players_1_isRadiant', 'players_2_isRadiant', 'players_3_isRadiant', 
        'players_4_isRadiant', 'players_5_isRadiant', 'players_6_isRadiant', 'players_7_isRadiant', 'players_8_isRadiant', 
        'players_9_isRadiant']

        dataset = self.create_one_hot_encoded_dataset(dataset, hero_id_columns, is_radiant_columns, 140)

        print("Splitting data")
        train_dataset, val_dataset, test_dataset = self.split_data(dataset, train_size)

        # Check if the dataset is empty
        print(f"Train dataset size: {train_dataset.cardinality().numpy()}")
        if train_dataset.cardinality().numpy() == 0:
            raise ValueError("Train dataset is empty!")

        print(f"Val dataset size: {val_dataset.cardinality().numpy()}")
        if val_dataset.cardinality().numpy() == 0:
            raise ValueError("Train dataset is empty!")

        print(f"Test dataset size: {test_dataset.cardinality().numpy()}")
        if test_dataset.cardinality().numpy() == 0:
            raise ValueError("Train dataset is empty!")
        
        # Prepare datasets for batching and prefetching
        train_dataset = train_dataset.batch(batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
        val_dataset = val_dataset.batch(batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)
        test_dataset = test_dataset.batch(batch_size, drop_remainder=True).prefetch(tf.data.experimental.AUTOTUNE)

        input_shape = next(iter(train_dataset))[0].shape[1:]
        print(f"Dynamically inferred input shape: {input_shape}")
        model = self.build_model(input_shape=input_shape)

        model_history = model.fit(train_dataset, validation_data=val_dataset, epochs=epochs)

        print("Stacked BiLSTM Model training complete")
        # Save the model
        model.save('dota_lstm_model.keras')
        print("Model saved as 'dota_lstm_model.keras'")

        test_loss, test_accuracy = model.evaluate(test_dataset, verbose=0)
        print('\nTest accuracy:', test_accuracy)
        print('\nLoss:', test_loss)

        precision_metric = tf.keras.metrics.Precision()
        for features, labels in test_dataset:
            predictions = model.predict(features)
            precision_metric.update_state(labels, predictions)
        precision = precision_metric.result().numpy()
        print('\nPrecision:', precision)

        auc_metric = tf.keras.metrics.AUC()
        for features, labels in test_dataset:
            predictions = model.predict(features)
            auc_metric.update_state(labels, predictions)
        roc_auc = auc_metric.result().numpy()
        print('\nROC AUC:', roc_auc)

        self.plot_accuracy_vs_epochs(model_history)