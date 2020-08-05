import statsmodels.api as sm
from statsmodels.tsa.stattools import acf
import numpy as np
import geopandas as gpd
import pandas as pd
from datetime import timedelta
from itertools import product
from math import pi, exp, sin, ceil
from shapely.geometry import Point
import matplotlib.pyplot as plt
import os

import sys
sys.path.append('./modules/')
from helper_funcs import load_pickle_from_py2, save_pickle, load_pickle
from markov_meco import NSNR, Streams
from inverse_method import Inverse_Method

class KDE:

    @classmethod
    def optimal_bw(cls, data):
        print('optimal bw')
        return sm.nonparametric.KDEMultivariate(data, ['c','c']).bw 

    @classmethod
    def variance_matrix(cls, bandwidths, type='diagonal'):
        if type=='diagonal':
            H = np.diag(bandwidths) ** 2
        L = np.linalg.cholesky(H)
        det = np.linalg.det(H)
        inv = np.linalg.inv(H)
        return H, L, det, inv

    def __init__(self, data, bw=None):
        self.bw_dict = {}
        self.adjust_kde(data, bw)

    def adjust_kde(self, data, bw, seed=0):
        self.X_i = {key: np.array(item) for key, item in data.items()}
        for key, item in data.items():
            if bw is None:
                kde = sm.nonparametric.KDEMultivariate(item, ['c', 'c'])
                bw = kde.bw
            H, L, det, inv = KDE.variance_matrix(bw)
            self.bw_dict[key] = {'H':H, 'L':L, 'det':det, 'inv':inv}

    def sample(self, n_samples, kde_list, seed=0):
        self.rand_gauss = np.random.RandomState(seed*2)
        self.rand_position = np.random.RandomState(seed*2+1)
        sampling = []
        for j, key in enumerate(kde_list):
            if n_samples[j]>0:
                # changed this line
                L = self.bw_dict[key]['H']
                z = self.rand_gauss.standard_normal((n_samples[j], 2))
                pos = self.rand_position.choice(list(range(len(self.X_i[key]))), n_samples[j], replace=True)
                x_i = [self.X_i[key][i] for i in pos]
                sampling.append([np.dot(L, z[i]) + x_i[i] for i in range(n_samples[j])])
        return np.concatenate(sampling)

    def plot(self, kde_key, bounds):
        #kde = sm.nonparametric.KDEMultivariate(self.X_i[kde_key], ['c', 'c'], bw)
        x_min, y_min, x_max, y_max = bounds[0], bounds[1], bounds[2], bounds[3]
        X, Y = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
        positions = np.vstack([X.ravel(), Y.ravel()])
        Z = np.reshape(kde.pdf(positions).T, X.shape)
        fig, ax = plt.subplots()
        ax.contourf(X, Y, Z)
        plt.show()

class stKDE(KDE):

    def __init__(self, data, bw=None):
        all_data = np.concatenate([item for key, item in data.items()])
        bw = KDE.optimal_bw(all_data)
        self.bw_dict = {}
        self.adjust_kde(data, bw)
        self.w = Weight_function()
        self.T_obs = len(data.keys())

    def sample(self, n_samples, dummy=None, seed=0):
        self.rand_gauss = np.random.RandomState(seed*3)
        self.rand_position = np.random.RandomState(seed*3+1)
        self.rand_kde = np.random.RandomState(seed*3+2)
        sampling = []
        U = len(n_samples)
        for j, u in enumerate(range(self.T_obs, U+self.T_obs)):
            if n_samples[j]>0:
                L, x_i, z = self.kde_selection(n_samples[j], u)
                sampling_j = [np.dot(L[i], z[i]) + x_i[i] for i in range(n_samples[j])]
                sampling.append([sample[0] for sample in sampling_j])
        return np.concatenate(sampling)
    
    def kde_selection(self, n_sample, u):
        weights = self.w.kde_weights(u, self.T_obs)
        kdes = self.rand_kde.choice(list(range(self.T_obs)), n_sample, p=weights, replace=True)
        L = [self.bw_dict[key]['L'] for key in kdes]
        x_i = [self.X_i[key][self.rand_position.choice(list(range(len(self.X_i[key]))), 1)] for key in kdes]
        z = self.rand_gauss.standard_normal((n_sample, 2))
        return L, x_i, z

class Grid:

    def __init__(self, gdf):
        self.grid = gdf
        self.grid_dict = {}

    def adjust_grid(self, data):
        for key, item in data.items():
            temp_gdf = gpd.GeoDataFrame({'geometry':item}, geometry='geometry', crs=self.grid.crs)
            merged_data = gpd.sjoin(temp_gdf, self.grid[['geometry']], how='left')
            table = merged_data.pivot_table(index='index_right',
                        aggfunc=len, fill_value=0)
            table.name = 'count'
            agg_grid = pd.concat([self.grid, table], axis=1)
            agg_grid['id'] = self.grid.index
            agg_grid['count'] = agg_grid['count'].apply(lambda x: 0 if np.isnan(x) else x)
            agg_grid['prop'] = agg_grid['count'].div(agg_grid['count'].sum())
            self.grid_dict[key] = gpd.GeoDataFrame(agg_grid, geometry='geometry', crs=self.grid.crs)

    def tune(self, data):
        for key, weights in data.items():
            temp_grid = self.grid.copy()
            temp_grid['prop'] = weights
            self.grid_dict[key] = temp_grid
    
    def sample(self, n_samples, grid_list, seed=0):
        self.rand_grid_selection = np.random.RandomState(seed*2)
        self.rand_inside_grid = np.random.RandomState(seed*2 + 1)
        sampling = []
        for i, key in enumerate(grid_list):
            if n_samples[i]>0:
                sample = self.grid_dict[key].sample(n_samples[i], weights='prop', random_state=self.rand_grid_selection, replace=True)['geometry'].bounds.values
                sampling.append([(self.rand_inside_grid.uniform(bounds[0], bounds[2]), self.rand_inside_grid.uniform(bounds[1], bounds[3])) for bounds in sample])
        return np.concatenate(sampling)

class Weight_function:
    ## Falta revisar si al sumar los w_c,
    ## tienen que sumar uno cada uno.
    cells_rhos = None

    def __init__(self):
        self.cells_rhos = Weight_function.cells_rhos

    def kde_weights(self, u, T_obs):
        w_kde = [self.eval(u-t) for t in range(T_obs)]
        return w_kde/np.sum(w_kde)
    
    def eval(self, x):
        return np.sum([self.w_c(x, rhos) for cell, rhos in self.cells_rhos.items()])
        
    def w_c(self, x, rhos, T_1=24, T_2=168):
        rho_1 = rhos[0]**(x)
        rho_2 = rhos[1]**(x)
        rho_3 = rhos[2]**(sin(pi*x/T_1)*sin(pi*x/T_1))
        rho_4 = rhos[3]**(sin(pi*x/T_2)*sin(pi*x/T_2))
        return rho_1 + rho_2*rho_3*rho_4

class ArrivalProcess:

    def __init__(self, rates):
        self.rates = rates
        self.build_rate_dict()
        self.inv = Inverse_Method()
    
    def sample(self, keys, seed=0):
        lambda_t =  self.build_rate_func([self.lambda_t[key] for key in keys])
        return self.inv.sample(lambda_t, seed)

    def build_rate_dict(self):
        pass

    def build_rate_func(self, lambda_t_list):
        return np.concatenate([[{'lb':interval['lb']+24*i, 'ub':interval['ub'] + 24*i, 'rate':interval['rate']}
                for interval in lambda_] for i, lambda_ in enumerate(lambda_t_list)])

class NonStationaryNonRenewal(ArrivalProcess):

    def __init__(self, rates, scv, ac):
        self.rates = rates
        self.scv = scv
        self.ac = ac

    def sample(self, keys, seed=0):
        S = Streams(3*seed, 3*seed+1, 3*seed+2)
        arrivals = []
        for key in keys:
            NSNR_generator = NSNR(24, 1, self.scv[key], self.ac[key])
            NSNR_generator.set_intervals_from_data(self.rates[key])
            arrivals.append(NSNR_generator.generate_interarrival_times(S)['rep0'])     
        return np.concatenate(arrivals)

class NonHomogeneousPoissonProcess(ArrivalProcess):

    def build_rate_dict(self):
        self.lambda_t = {key: [{'lb':hour, 'ub':rates_[i+1][0], 'rate':rate} \
            if i<len(rates_)-1 else {'lb':hour, 'ub':hour+1, 'rate':rate} 
            for i, (hour, rate) in enumerate(rates_)] for key, rates_ in self.rates.items()}

class PoissonProcess(ArrivalProcess):

    def build_rate_dict(self):
        self.lambda_t = {key: [{'lb':0, 'ub':24, 'rate':np.mean([rate for hour, rate in rates_])}] for key, rates_ in self.rates.items()}

class DataProcessor:
    #seasons_per_month = {11: 'Fall', 12: 'Winter', 1: 'Winter', 2: 'Winter', 3: 'Spring', 4: 'Spring', 5:'Spring',
    #    6: 'Summer', 7:'Summer', 8: 'Summer', 9: 'Fall', 10:'Fall'}
    seasons_per_month = {key: 0 for key in range(1,13)}

    def __init__(self):
        pass

    @classmethod
    def process(cls, gdf, save_dir=None, temporal=True, spatial=True):
        if save_dir:
            if os.path.exists(save_dir):
                print('File already exists, preparing to load it..')
                results = load_pickle(save_dir)
                print('Load successful')
                return results['temporal'], results['spatial']

        if 'season' not in gdf.columns:
            gdf = gdf[['date', 'geometry']].copy()
            gdf['season'] = gdf['date'].apply(lambda x: cls.seasons_per_month[x.month])
        else:
            gdf = gdf[['date', 'season', 'geometry']].copy()

        temporal_dict, spatial_dict = {}, {}
        #Temporal processing
        if temporal:
            print('Temporal processing...')
            rates_dict, scv_dict, ac_dict = {}, {}, {}
            print('--- Computing parameters...')
            for season, weekday in product(gdf.season.unique(), gdf.date.dt.weekday.unique()):
                temp_gdf = gdf[(gdf['season']==season) & (gdf.date.dt.weekday==weekday)].copy()
                rate, scv, ac = cls.temporal_processing(temp_gdf)
                print('{} {}: SCV {} AC {} '.format(season, weekday, scv, ac))
                rates_dict[season, weekday] = rate
                scv_dict[season, weekday] = scv
                ac_dict[season, weekday] = ac
            print('--- Building NHNP...')
            nhpp = NonHomogeneousPoissonProcess(rates_dict)
            print('--- Building PP...')
            pp = PoissonProcess(rates_dict)
            print('--- Building NSNR...')
            nsnr = NonStationaryNonRenewal(rates_dict, scv_dict, ac_dict)
            temporal_dict = {'pp':pp, 'nhpp':nhpp, 'nsnr':nsnr}

        #Spatial processing
        if spatial:
            print('Spatial processing...')
            print('--- Computing parameters...')
            hourly_data, agg_hourly_data, opt_bw = cls.spatial_processing(gdf)
            print('--- Building stKDE...')
            stkde = stKDE(hourly_data, bw=opt_bw)
            print('--- Building KDE...')
            kde = KDE(agg_hourly_data)
            spatial_dict =  {'kde':kde, 'stkde':stkde}

        if save_dir:
            save_pickle({'temporal':temporal_dict, 'spatial':spatial_dict}, save_dir)

        return temporal_dict, spatial_dict
    
    @classmethod
    def spatial_processing(cls, gdf):
        gdf = gdf.copy()
        gdf['cum_hour'] = (gdf['date'] - gdf['date'].iloc[0]).dt.seconds.div(3600).astype(int)
        hourly_data = {hour: [(point.x, point.y) for point in gdf[gdf['cum_hour']==hour].geometry.values] for hour in range(gdf['cum_hour'].max()+1)}
        latlon = [(point.x, point.y) for point in gdf.geometry.values]
        print('--- Computing optimal bw...')
        opt_bw = KDE.optimal_bw(latlon)
        #For a KDE adjusted for every agg hour.
        '''
        agg_hourly_data = {(season, weekday, hour): [(point.x, point.y) \
            for point in gdf[(gdf['season']==season) & (gdf['date'].dt.weekday == weekday) & (gdf['date'].dt.hour == hour)].geometry.values] 
            for season, weekday, hour in product(gdf.season.unique(), gdf.date.dt.weekday.unique(), gdf.date.dt.hour.unique())}
        '''
        #The same KDE for each hour.
        points = [(point.x, point.y) \
            for point in gdf.geometry.values] 
        agg_hourly_data = {(season, weekday, hour): points for season, weekday, hour in product(gdf.season.unique(), gdf.date.dt.weekday.unique(), gdf.date.dt.hour.unique())}
        return hourly_data, agg_hourly_data, opt_bw

    @classmethod
    def temporal_processing(cls, gdf):

        def calculate_V(row, cols):
            c = (1/(len(row[cols])-1))*(np.array([(row[i]-row['lambda'])**2 for i in cols]).sum())
            return c

        #Build observations table
        gdf = gdf.copy()
        samples = gdf.groupby([gdf['date'].dt.date, gdf['date'].dt.hour]).size()
        index_0 = samples.index.get_level_values(0).unique()
        new_index = pd.MultiIndex.from_product([index_0, np.array(range(24))], names=('date', 'hour'))
        samples = samples.reindex(new_index, fill_value=0)
        samples = pd.concat([samples[idx] for idx in index_0], axis=1, sort=False) # Idk if this is necessary
        samples.columns = ['obs_'+str(i) for i in range(len(samples.columns))]
        cum_cols = pd.DataFrame({'cum_'+str(i):samples[col].cumsum(axis = 0) for i,col in enumerate(samples.columns)})
        
        #Compute parameters
        lambda_col = pd.DataFrame(cum_cols.mean(axis=1), columns=['lambda'])
        table = pd.concat([samples, cum_cols, lambda_col], axis=1, sort=False)  
        table['V'] = table.apply(lambda x: calculate_V(x, cum_cols.columns), axis=1)
        table['V/lambda'] = table['V']/table['lambda']
        table['mean_rate'] = table[samples.columns].mean(axis=1)

        # From the table, we can now estimate scv and autocorrelation
        cv = table['V/lambda'].mean()
        scv = np.sqrt(cv)
        #ac = acf(table.mean_rate.values, nlags = 1)[1]
        ac = gdf['date'].diff().dt.seconds.autocorr(lag=1)
        print(table.mean_rate.values)
        rate = np.array(list(enumerate(0.6*table.mean_rate.values)))
        return rate, scv, ac

class Sampler:

    def __init__(self, temporal_methods, spatial_methods):
        self.temporal_methods = temporal_methods
        self.spatial_methods = spatial_methods
    
    def sample(self, start, sim_duration, n_reps, temporal_method='nsnr', spatial_method='kde', seeds=None, crs={'init':'epsg:2263'}, save_dir=None):
        print('Sampling {} replicas using {} and {}'.format(n_reps, temporal_method.upper(), spatial_method.upper()))
        self.temporal_method = temporal_method
        self.spatial_method = spatial_method
        self.start_date = pd.to_datetime(start)
        self.sim_duration = ceil(sim_duration/24) #from hours to days
        self.generate_timeline()
        if not seeds:
            self.seeds = [seed for seed in range(n_reps)] 
        else:
            self.seeds = seeds
        samples = []
        for n in range(n_reps):
            print('--- Replica {}'.format(n))
            interarrivals = self.temporal_methods[self.temporal_method].sample(self.timeline, self.seeds[n])
            arrivals = np.cumsum(interarrivals)
            arrivals_dt = [(self.start_date + timedelta(hours=round(x,6))).strftime("%m/%d/%Y, %H:%M:%S") for x in arrivals]
            df = pd.DataFrame({'date': arrivals_dt, 'arrival':arrivals, 'interarrival':interarrivals})
            df['date'] = pd.to_datetime(df['date'])
            n_arrivals_per_hour, spatial_keys = self.get_spatial_counts(df)
            locations = self.spatial_methods[self.spatial_method].sample(n_arrivals_per_hour, spatial_keys)
            geometries = [Point(x,y) for x,y in locations]
            samples.append(gpd.GeoDataFrame(df, geometry=geometries, crs=crs))
        if save_dir:
            self.save_samples(samples, save_dir)
        return samples

    def generate_timeline(self):
        self.time_series = self.start_date + pd.to_timedelta(np.arange(self.sim_duration), 'D')
        self.timeline = [(DataProcessor.seasons_per_month[date.month], date.weekday()) for date in self.time_series]
    
    def get_spatial_counts(self, df):
        gb = df.groupby([df['date'].dt.date, df['date'].dt.hour]).size()
        categories = [(DataProcessor.seasons_per_month[date.month], date.date(), date.weekday(), hour) for date in self.time_series for hour in range(24)]
        keys, counts = [], []
        for season, date, weekday, hour in categories:
            if (date, hour) in gb.index:
                count = gb.loc[(date, hour)]
                counts.append(count)
            else:
                counts.append(0)
            keys.append((season, weekday, hour))
        return counts, keys

    def save_samples(self, samples, save_dir):
        for i, sample in enumerate(samples):
            sample = sample.copy()
            sample['date'] = sample['date'].dt.strftime("%m/%d/%Y %H:%M:%S")
            fname = save_dir + self.spatial_method + '_'+ self.temporal_method +'_'+str(i)+'.shp'
            sample.to_file(fname)

if __name__ == '__main__':
    rhosname = '../data/rhos_dict_louisville.pickle'
    Weight_function.cells_rhos = load_pickle_from_py2(rhosname)
    
    dataname = '../shapes/arrivals/arrivals.shp'
    data = gpd.read_file(dataname)
    data['date'] = pd.to_datetime(data['date'])
    
    
    dicts_save_dir = '../data/sampling_methods_louisville.pickle'
    temporal_methods, spatial_methods = DataProcessor.process(data, save_dir=dicts_save_dir)
    sampler = Sampler(temporal_methods, spatial_methods)
    start = '25 of May, 2020'
    sim_duration = 24 * 7 #hours
    n_reps = 20
    temporal_method = ['nhpp']
    spatial_method = ['stkde']
    save_dir = '../shapes/replicas/'
    for t_meth, s_meth in product(temporal_method, spatial_method):
        sampler.sample(start, sim_duration, n_reps, temporal_method=t_meth, spatial_method=s_meth, save_dir=save_dir, crs=data.crs)
    
    '''
    #Load data for tox example
    region = gpd.read_file('grid/grid_1km.shp')
    ponds_dict = load_pickle('data/grid_spatial.pickle')
    temporal_dict = load_pickle('data/grid_temporal.pickle')
    nsnr = NonStationaryNonRenewal(temporal_dict['rates'], temporal_dict['scv'], temporal_dict['ac'])
    grid = Grid(region)
    grid.tune(ponds_dict)
    sampler = Sampler({'nsnr':nsnr}, {'grid':grid})
    n_reps = 1
    sim_duration = 24*28 #hours
    save_dir = 'data/replicas/'
    start = '1 of June, 2020'
    sampler.sample(start, sim_duration, n_reps, temporal_method='nsnr', spatial_method='grid', save_dir=save_dir)
    '''