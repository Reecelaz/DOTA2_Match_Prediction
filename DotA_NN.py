import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import pandas as pd
import numpy as np

class DotA_NN:
    def __init__(self):
        # Set up the strategy for multi-GPU support
        self.strategy = tf.distribute.MirroredStrategy()
        print('Number of devices: {}'.format(self.strategy.num_replicas_in_sync))

    def preprocess_data(self, data_df):
        # Assuming 'radiant_win' is the label and the rest are features
        X = data_df.drop(columns=['radiant_win'])
        y = data_df['radiant_win']
        
        # Convert to numpy arrays
        X = X.values
        y = y.values

        # Normalize feature data (if needed)
        X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        
        return X, y

    def build_model(self):
        # Build the model within the strategy scope
        with self.strategy.scope():
            model = Sequential([
                Dense(64, input_dim=X.shape[1], activation='relu'),
                Dense(32, activation='relu'),
                Dense(1, activation='sigmoid')
            ])
            
            model.compile(optimizer=Adam(),
                          loss='binary_crossentropy',
                          metrics=['accuracy'])
        return model

    def train(self, data_df):
        X, y = self.preprocess_data(data_df)
        model = self.build_model()
        
        # Train the model
        model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)

        # Save the model
        model.save('dota_nn_model.h5')
        print("Model training complete and saved as 'dota_nn_model.h5'")

# Example usage
def main():
    # Load data (assuming the JSON file is already created and valid)
    data_df = pd.read_json('C:/Users/Reece Lazarus/DOTA2_Match_Prediction/DotA_data.json')
    
    nn = DotA_NN()
    nn.train(data_df)

if __name__ == "__main__":
    main()
