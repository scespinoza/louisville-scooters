import os
import time
import pickle
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import statsmodels.api as sm

from shapely.geometry import Point
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras import models, layers, optimizers, callbacks, metrics

def fix_duplicated_ids(edges):
    print('Fixing duplicates')
    edges['osmid'] = edges[['u', 'v', 'osmid']].apply(lambda x: str(x[0]) + '-' + str(x[1]) + '-' + str(x[2]), axis=1)
    return edges

class ConditionedKDE:
    def __init__(self, endog, exog):
        self.endog_unique = pd.DataFrame(endog, columns=['endog1', 'endog2']).drop_duplicates().values
        self.dens_c = sm.nonparametric.KDEMultivariateConditional(endog=[endog],
                        exog=[exog], dep_type='cc', indep_type='cc', bw='normal_reference')
        self.bw = self.dens_c.bw
    def compute_weights(self, obs):
        exog_predict = np.array([obs] * self.endog_unique.shape[0])
        return self.dens_c.pdf(exog_predict=exog_predict, endog_predict=self.endog_unique)

    def sample(self, exog):
        d = self.endog_unique.shape[1]
        samples = []
        for i, obs in enumerate(exog):
            print('Sampling {}/{} obs.'.format(i, len(exog)))
            p = self.compute_weights(obs)
            p = p / p.sum()
            xi = np.random.choice(range(self.endog_unique.shape[0]), size=1, p=p)[0]
            x = self.endog_unique[xi]
            cov = np.diag(self.dens_c.bw[-d:])**2
            norm = np.random.multivariate_normal(np.zeros(d), cov)
            samples.append(list(norm + x))
        return np.array(samples)

class KDEPredictor:
    def __init__(self, data, endog_cols, exog_cols, time_col, time_window=4, period='week', name='DestPredictor'):
        self.name = name
        self.data = data
        self.time_window = time_window
        self.endog_cols = endog_cols
        self.exog_cols = exog_cols
        self.time_col = time_col
        self.period = period
        self.kdes = {}
        self.fit()

    def fit(self):
        if self.period == 'week':
            self.period_hours = 24 * 7
        if self.period == 'days':
            self.period_hours = 24

        self.data['timestep'] = ((self.data[self.time_col].dt.dayofweek * 24 + self.data[self.time_col].dt.hour) % self.period_hours) // self.time_window
        print(self.data['timestep'].unique())

        for timestep in range(self.period_hours//self.time_window):
            print('Timestep {}'.format(timestep))
            filter_data = data[data['timestep'] == timestep].copy()
            print('{} obs.'.format(filter_data.shape[0]))
            endog = filter_data[self.endog_cols].values
            exog = filter_data[self.exog_cols].values
            self.kdes[timestep] = ConditionedKDE(endog, exog)

    def predict(self, data):
        data = data.copy()
        destinations = pd.DataFrame(np.zeros((data.shape[0], 2)), index=data.index)
        data['timestep'] = ((data[self.time_col].dt.dayofweek * 24 + data[self.time_col].dt.hour * 24) % self.period_hours) // self.time_window
        for timestep in range(self.period_hours//self.time_window):
            filter_data = data[data['timestep'] == timestep].copy()
            exog_predict = filter_data[self.exog_cols].values
            destinations.loc[filter_data.index, :] = self.kdes[timestep].sample(exog_predict)
        return destinations.values

            



class NeuralNetwork(models.Model):
    def __init__(self, name='NeuralNetwork'):
        super(NeuralNetwork, self).__init__(name=name)
        self.dense1 = layers.Dense(256, activation='relu')
        self.dense2 = layers.Dense(128, activation='relu')
        self.dense3 = layers.Dense(64, activation='relu')
        self.dense4 = layers.Dense(2, activation='linear')
    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return self.dense4(x)

            
class DestinationPredictor:
    def __init__(self, name='DestPredictor', scaler=StandardScaler, model=NeuralNetwork):
        self.name = name
        self.x_scaler = scaler()
        self.y_scaler = scaler()
        self.model = model(name=name)

    def fit(self, X, y, optimizer=optimizers.Adam, lr=9e-4, callbacks_list=None, epochs=500, batch_size=32):
        X_train, X_test, y_train, y_test = train_test_split(X, y)
        X_train_scaled = self.x_scaler.fit_transform(X_train)
        y_train_scaled = self.y_scaler.fit_transform(y_train)
        X_test_scaled = self.x_scaler.transform(X_test)
        y_test_scaled = self.y_scaler.transform(y_test)
        self.model.compile(optimizer=optimizer(lr), loss='mse')
        return self.model.fit(X_train_scaled, y_train_scaled, validation_data=(X_test_scaled, y_test_scaled),
                              epochs=epochs, batch_size=batch_size, callbacks=callbacks_list)
    def predict(self, X):
        X_scaled = self.x_scaler.transform(X)
        y_pred = self.model.predict(X_scaled)
        return self.y_scaler.inverse_transform(y_pred)

    def save(self):
        if not 'models' in os.listdir():
            os.mkdir('models')
        try:
            os.mkdir('models/' + self.name)
        except:
            pass
        with open('models/{}/scalers.pickle'.format(self.name), 'wb') as file:
            pickle.dump({'xscaler': self.x_scaler, 'yscaler': self.y_scaler}, file)
        self.model.save_weights('models/{}/weights.h5'.format(self.name))

    @classmethod
    def load_trained_model(cls, name, dim=4):
        predictor = cls(name=name)
        with open('models/{}/scalers.pickle'.format(name), 'rb') as file:
            scalers_dict = pickle.load(file)
            predictor.x_scaler = scalers_dict['xscaler']
            predictor.y_scaler = scalers_dict['yscaler']
        predictor.model.build(input_shape=(None, dim))
        predictor.model.load_weights('models/{}/weights.h5'.format(name))
        return predictor



class TripGenerator:

    def __init__(self, model, network, arrivals):
        self.network = network
        self.model = model
        self.arrivals = arrivals

    def predict_destinations(self):
        self.arrivals['day'] = self.arrivals['date'].dt.dayofweek
        self.arrivals['hour'] = self.arrivals['date'].dt.hour
        self.arrivals['StartLongitude'] = self.arrivals['geometry'].apply(lambda p: p.x)
        self.arrivals['StartLatitude'] = self.arrivals['geometry'].apply(lambda p: p.y)
        X = self.arrivals[['StartLongitude', 'StartLatitude', 'date']]
        destinations_coords = self.model.predict(X)
        self.destinations = gpd.GeoSeries([Point(x, y) for x, y in destinations_coords])
        return self.destinations

    def create_trips_df(self, origin_layer='walk', dest_layer='bike', origin_transfer=False, dest_transfer=True):
        start = time.time()
        print('Origin OSMIDs')
        origin_osmid = self.network.get_nearest_osmids(self.arrivals['geometry'], layer=origin_layer, transfer=origin_transfer)
        print('Destination OSMIDs')
        print(self.destinations)
        destination_osmid = self.network.get_nearest_osmids(self.destinations, layer=dest_layer, transfer=dest_transfer)
        
        trips_dict = {
            'origin': origin_osmid,
            'destination': destination_osmid,
            'date': list(self.arrivals['date']),
            'interarrival': list(self.arrivals['interarriv']),
            'arrival': list(self.arrivals['arrival'])
        }
        print('Elapsed time: {:.2f}s'.format(time.time() - start))
        return pd.DataFrame(trips_dict)
        

    
if __name__ == '__main__':
    from multimodal_network import MultiModalNetwork
    replicas_dir = 'shapes/replicas'
    model_name = 'louisville-trips'
    study_area_filename = 'shapes/study_area/study_area.shp'
    study_area = gpd.read_file(study_area_filename).to_crs('EPSG:4326')
    study_area_polygon = study_area.iloc[0]['geometry']
    network = MultiModalNetwork.from_polygon(study_area_polygon, speeds={'walk': 1.4, 'bike':2.16})

    data = pd.read_csv('data/data_2019.csv')
    data['date'] = pd.to_datetime(data['date'])
    model = KDEPredictor(data, endog_cols=['EndLongitude', 'EndLatitude'], exog_cols=['StartLongitude', 'StartLatitude'], time_col='date')
    for rep in os.listdir(replicas_dir):
        if '.shp' in rep:
            print('Converting file: {}'.format(rep))
            arrivals = gpd.read_file(replicas_dir + '/' + rep)
            arrivals['date'] = pd.to_datetime(arrivals['date'])
            trip_generator = TripGenerator(model=model,
                                        network=network, arrivals=arrivals)
            trip_generator.predict_destinations()
            trips_df = trip_generator.create_trips_df()
            trips_df.to_csv('data/replicas/{}'.format(rep).replace('.shp', '.csv'))

            