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

        #Split dataset using sizes
        train_dataset = dataset.take(train_size)
        val_dataset = dataset.skip(train_size).take(val_size)
        test_dataset = dataset.skip(train_size + val_size)
        
        return train_dataset, val_dataset, test_dataset
    
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
        self.inspect_first_row(dataset, output_file='first_row.txt')
        print("Encoding hero columns")

        hero_id_columns = ['players_0_hero_id', 'players_1_hero_id', 'players_2_hero_id', 'players_3_hero_id', 'players_4_hero_id', 
        'players_5_hero_id', 'players_6_hero_id', 'players_7_hero_id', 'players_8_hero_id', 'players_9_hero_id']

        is_radiant_columns = ['players_0_isRadiant', 'players_1_isRadiant', 'players_2_isRadiant', 'players_3_isRadiant', 
        'players_4_isRadiant', 'players_5_isRadiant', 'players_6_isRadiant', 'players_7_isRadiant', 'players_8_isRadiant', 
        'players_9_isRadiant']

        print("Splitting data")
        train_dataset, val_dataset, test_dataset = self.split_data(dataset, train_size)

        self.inspect_first_row(train_dataset, output_file='first_row_after_split.txt')

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