import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import pandas as pd
import numpy as np

class DotA_NN:

    input_variables = ['players_0_hero_id', 'players_1_hero_id', 'players_2_hero_id', 'players_3_hero_id', 'players_4_hero_id', 'players_5_hero_id', 'players_6_hero_id', 'players_7_hero_id', 'players_8_hero_id', 'players_9_hero_id', 'players_0_isRadiant', 'players_1_isRadiant', 'players_2_isRadiant', 'players_3_isRadiant', 'players_4_isRadiant', 'players_5_isRadiant', 'players_6_isRadiant', 'players_7_isRadiant', 'players_8_isRadiant', 'players_9_isRadiant']

    def __init__(self):
        # Set up the strategy for multi-GPU support
        self.strategy = tf.distribute.MirroredStrategy()
        print('Number of devices: {}'.format(self.strategy.num_replicas_in_sync))

    def preprocess_data(self, data_df):
        X = data_df[self.input_variables]
        y = data_df['radiant_win']

        # Convert boolean variables to int for use with tensorflow
        X = X.astype({col: 'int' for col in X.select_dtypes(include='bool').columns})
        y = y.astype('int')
        
        # Convert to numpy arrays
        X = X.values
        y = y.values
        
        return X, y

    def build_model(self, input_size):
        # Build the model within the strategy scope
        with self.strategy.scope():
            model = Sequential([
                Dense(64, input_dim=input_size, activation='relu'),
                Dense(32, activation='relu'),
                Dense(32, activation='relu'),
                Dense(1, activation='sigmoid')
            ])
            
            model.compile(optimizer=Adam(),
                          loss='binary_crossentropy',
                          metrics=['accuracy'])
        return model

    def train(self, data_df):
        X, y = self.preprocess_data(data_df)
        
        model = self.build_model(X.shape[1])
        
        # Train the model
        model.fit(X, y, epochs=10, batch_size=32, validation_split=0.2)

        # Save the model
        #model.save('dota_nn_model.h5')
        model.save('dota_nn_model.keras')
        print("Model training complete and saved as 'dota_nn_model.keras'")

# Example usage
def main():
    # Load data (assuming the JSON file is already created and valid)
    data_df = pd.read_json('C:/Users/Reece Lazarus/DOTA2_Match_Prediction/DotA_data.json')
    
    nn = DotA_NN()
    nn.train(data_df)

if __name__ == "__main__":
    main()
