import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
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
    
    def split_data(self, dataset, train_ratio=0.6, seed=42):
        val_ratio = test_ratio = (1 - train_ratio) / 2
        # Ensure ratios add up to 1.0
        assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1.0"
        
        # Calculate dataset size
        dataset_size = dataset.cardinality().numpy()
        
        # Compute sizes based on ratios
        train_size = int(train_ratio * dataset_size)
        val_size = int(val_ratio * dataset_size)

        #dataset = dataset.shuffle(buffer_size=dataset_size, seed=42)
        
        # Shuffle with a fixed seed for reproducibility
        #dataset = dataset.shuffle(buffer_size=dataset_size, seed=seed, reshuffle_each_iteration=True)
        
        # Create splits
        train_dataset = dataset.take(train_size)
        val_dataset = dataset.skip(train_size).take(val_size)
        test_dataset = dataset.skip(train_size + val_size)
        
        return train_dataset, val_dataset, test_dataset
    '''
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
    '''

    def build_model(self, input_size, learning_rate=0.001):
        hidden_layer_fn = 'relu'
        with self.strategy.scope():
            model = Sequential([
                Flatten(),  # Dynamically flatten the input (no need to specify input_shape)
                Dense(256, activation=hidden_layer_fn),
                Dense(256, activation=hidden_layer_fn),
                Dense(256, activation=hidden_layer_fn),
                Dense(256, activation=hidden_layer_fn),
                Dense(256, activation=hidden_layer_fn),
                Dense(128, activation=hidden_layer_fn),
                Dense(64, activation=hidden_layer_fn),
                Dense(32, activation=hidden_layer_fn),
                Dense(1, activation='sigmoid')
            ])
            
            model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])

        return model
    
    def plot_accuracy_vs_epochs(self, model_history):
        plt.plot(model_history.history['accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.savefig('NN accuracy_vs_epochs.png')

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

    def inspect_first_row(self, dataset, output_file='first_row.txt'):
        # Take the first element from the dataset
        first_row = next(iter(dataset.take(1)))
        
        features, label = first_row  # Unpack the features and the label
        
        # Convert the Tensors to NumPy arrays for easier reading
        features_np = features.numpy()
        label_np = label.numpy()

        # Write the values to a text file
        with open(output_file, 'w') as f:
            f.write("Features (X):\n")
            np.savetxt(f, features_np, fmt='%s')
            f.write("\nLabel (y):\n")
            f.write(f"{label_np}\n")

        print(f"First row saved to {output_file}")

    def train(self, dataset, train_size, epochs, batch_size, learning_rate):
        print("Encoding hero columns")

        hero_id_columns = ['players_0_hero_id', 'players_1_hero_id', 'players_2_hero_id', 'players_3_hero_id', 'players_4_hero_id', 
        'players_5_hero_id', 'players_6_hero_id', 'players_7_hero_id', 'players_8_hero_id', 'players_9_hero_id']

        is_radiant_columns = ['players_0_isRadiant', 'players_1_isRadiant', 'players_2_isRadiant', 'players_3_isRadiant', 
        'players_4_isRadiant', 'players_5_isRadiant', 'players_6_isRadiant', 'players_7_isRadiant', 'players_8_isRadiant', 
        'players_9_isRadiant']

        print("Splitting data")
        train_dataset, val_dataset, test_dataset = self.split_data(dataset, train_size)

        #dataset = self.create_one_hot_encoded_dataset(dataset, hero_id_columns, is_radiant_columns, 140)
        train_dataset = self.create_one_hot_encoded_dataset(train_dataset, hero_id_columns, is_radiant_columns, 140)
        val_dataset = self.create_one_hot_encoded_dataset(val_dataset, hero_id_columns, is_radiant_columns, 140)
        test_dataset = self.create_one_hot_encoded_dataset(test_dataset, hero_id_columns, is_radiant_columns, 140)
        
        # Storing first row in text file
        self.inspect_first_row(train_dataset, output_file='first_row.txt')

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
        
        model = self.build_model(input_size=next(iter(train_dataset))[0].shape[1], learning_rate=learning_rate)
        
        print("Begin model training")
        model_history = model.fit(train_dataset, epochs=epochs, validation_data=val_dataset, verbose=0)
        
        print("Model training complete")
        # Save the model
        model.save('dota_nn_model.keras')
        print("Model saved as 'dota_nn_model.keras'")

        print("Hyperparameter settings: \n", "train_size: ", train_size, "\n", "epochs: ", epochs, "\n", "batch_size: ", batch_size, "\n", "learning rate: ", learning_rate)
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