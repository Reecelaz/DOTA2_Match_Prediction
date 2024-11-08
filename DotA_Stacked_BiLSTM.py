import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Dropout
import numpy as np
import matplotlib.pyplot as plt


class DotA_Stacked_BiLSTM:

    def __init__(self):
        # Set up the strategy for multi-GPU support
        self.strategy = tf.distribute.MirroredStrategy()
        print('Number of devices: {}'.format(self.strategy.num_replicas_in_sync))
        print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    def build_model(self, input_shape):
        with self.strategy.scope():
            model = Sequential()

            # Use LSTM units that match the input feature size dynamically
            input_feature_size = input_shape[-1]  # Get the feature dimension from the input shape

            # Add first Bidirectional LSTM layer with units matching input feature size
            model.add(Bidirectional(LSTM(input_feature_size, return_sequences=True), input_shape=(None, input_feature_size)))
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

        return model

    def split_data(self, dataset, train_ratio=0.6, seed=42):
        val_ratio = test_ratio = (1 - train_ratio) / 2
        # Ensure ratios add up to 1.0
        assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1.0"
        
        # Calculate dataset size
        dataset_size = dataset.cardinality().numpy()
        
        # Compute sizes based on ratios
        train_size = int(train_ratio * dataset_size)
        val_size = int(val_ratio * dataset_size)
        
        # Create splits
        train_dataset = dataset.take(train_size)
        val_dataset = dataset.skip(train_size).take(val_size)
        test_dataset = dataset.skip(train_size + val_size)
        
        return train_dataset, val_dataset, test_dataset

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

    def plot_accuracy_vs_epochs(self, model_history):
        plt.plot(model_history.history['accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.savefig('LSTM accuracy_vs_epochs.png')

    def train(self, dataset, train_size, epochs, batch_size):
        print("Splitting data")
        train_dataset, val_dataset, test_dataset = self.split_data(dataset, train_size)

        self.inspect_first_row(train_dataset, output_file='first_row_after_split_lstm.txt')

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

        input_shape = next(iter(train_dataset))[0].shape[1:]  # Get input shape from the first batch
        print(f"Dynamically inferred input shape: {input_shape}")

        # Ensure the input data has the sequence dimension (adding a dummy sequence dimension if needed)
        train_dataset = train_dataset.map(lambda x, y: (tf.expand_dims(x, 1), y))  # Adding sequence length dimension (1)
        val_dataset = val_dataset.map(lambda x, y: (tf.expand_dims(x, 1), y))  # Same for validation data
        test_dataset = test_dataset.map(lambda x, y: (tf.expand_dims(x, 1), y))  # Same for test data

        model = self.build_model(input_shape)

        print("Begin Training")
        model_history = model.fit(train_dataset, validation_data=val_dataset, epochs=epochs, verbose=0)

        print("Stacked BiLSTM Model training complete")
        # Save the model
        model.save('dota_lstm_model.keras')
        print("Model saved as 'dota_lstm_model.keras'")

        print("Hyperparameter settings: \n", "train_size: ", train_size, "\n", "epochs: ", epochs, "\n", "batch_size: ", batch_size)
        test_loss, test_accuracy = model.evaluate(test_dataset, verbose=0)
        print('\nTest accuracy:', test_accuracy)
        print('\nLoss:', test_loss)

        precision_metric = tf.keras.metrics.Precision()
        for features, labels in test_dataset:
            predictions = model.predict(features, verbose=0)
            precision_metric.update_state(labels, predictions)
        precision = precision_metric.result().numpy()
        print('\nPrecision:', precision)

        auc_metric = tf.keras.metrics.AUC()
        for features, labels in test_dataset:
            predictions = model.predict(features, verbose=0)
            auc_metric.update_state(labels, predictions)
        roc_auc = auc_metric.result().numpy()
        print('\nROC AUC:', roc_auc)

        self.plot_accuracy_vs_epochs(model_history)

        return test_accuracy, roc_auc, precision, test_loss 