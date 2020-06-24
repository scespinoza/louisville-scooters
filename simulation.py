import bisect
import datetime
import json
import math
import pickle
import time
import argparse

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns

from abc import ABC, abstractmethod
from collections import defaultdict, deque
from functools import reduce
from itertools import product
from random import choices, sample
from scipy.spatial.distance import euclidean
from shapely.geometry import Point, Polygon

from multimodal_network import MultiModalNetwork
from models import HRP, Agent

class Random:
    """
    Helper class to manage random functions.
    """
    def __init__(self):
        pass

    @staticmethod
    def exponential(mean):
        return np.random.exponential(mean)

    @staticmethod
    def randint(xmin=0, xmax=1000):
        return np.random.randint(xmin, xmax)


        
class TripReader:
    def __init__(self, filename):
        self.data = pd.read_csv(filename)
        self.data['date'] = pd.to_datetime(self.data['date'])

    def construct_events(self, simulator):
        events = [UserRequest(User((origin, dest)), time=time*3600) 
                  for _, origin,dest,time in self.data[['origin', 'destination', 'arrival']].itertuples()]
        return events

class Grid:
    
    def __init__(self, boxes, shape, nodes_in_boxes=None):
        self.boxes = boxes
        self.shape = shape
        self.nodes_in_boxes = nodes_in_boxes
        self.demand_history = deque(maxlen=8)
        self.satisfied_requests_history = deque(maxlen=8)
        self.prices = np.zeros_like(boxes)
        self.stats =  {
            'demand': np.zeros_like(boxes),
            'satisfied_requests': np.zeros_like(boxes),
            'supply': np.zeros_like(boxes),
            'arrival': np.zeros_like(boxes),
            'expense': np.zeros_like(boxes)
           }
           

    def update_stat(self, osmid, event, value=None):
        idx = self.get_area(osmid)
        if event == 'request':            
            self.stats['demand'][idx] += 1
        if event == 'pickup':
            self.stats['satisfied_requests'][idx] += 1
        if event == 'arrival':
            self.stats['arrival'][idx] += 1
        if event == 'pricing':
            assert value != None, "Must specify an expense value"
            self.stats['expense'][idx] += value

    def get_area(self, osmid):
        node_in_area = {k: osmid in v for k, v in self.nodes_in_boxes.items()}
        return max(node_in_area, key=node_in_area.get)

    def get_area_polygon(self, osmid):
        return self.boxes.iloc[self.get_area(osmid)]

    def nodes_within_area(self, osmid):
        area = self.get_area(osmid)
        return self.nodes_in_boxes[area]
    
    def nodes_within_neighbors(self, osmid):
        area = self.get_area(osmid)
        neighbors = self.compute_neighboring_index(area)
        neighbor_nodes = []
        for neighbor_area in neighbors:
            neighbor_nodes += self.nodes_in_boxes[neighbor_area]
        return neighbor_nodes

    def compute_neighboring_index(self, area):
        n, m = self.shape
        i = area // m
        j = area % m
        neighbors = [(i - 1, j), (i, j + 1), (i + 1, j), (i, j - 1)]
        return [k * 10 + l for k, l in neighbors if (k >= 0 and k < n) and (l >= 0 and l < m)]

    def compute_supply(self):
        self.stats['supply'] = np.zeros_like(self.boxes)
        locations = [scooter.location for scooter in Scooter.scooters]
        for loc in locations:
            idx = self.get_area(loc)
            self.stats['supply'][idx] += 1

    def reset_stats(self):
        self.stats =  {
            'demand': np.zeros_like(self.boxes),
            'satisfied_requests': np.zeros_like(self.boxes),
            'supply': np.zeros_like(self.boxes),
            'arrival': np.zeros_like(self.boxes),
            'expense': np.zeros_like(self.boxes)
           }
    def get_stats(self):
        self.compute_supply()
        return self.stats

    def set_price(self, action):
        self.prices = action.reshape(self.boxes.shape)

    def get_scooter_price(self, scooter):
        area = self.get_area(scooter.location)
        return self.prices[area]

    def get_state(self, simulator):

        stats = gpd.GeoDataFrame(self.get_stats(), geometry=self.boxes)
        stats['remaining_budget'] = simulator.service_provider.budget
        self.demand_history.append(self.stats['demand'])
        self.satisfied_requests_history.append(self.stats['satisfied_requests'])
        demands = np.array(self.demand_history).sum(0)
        demands[demands == 0] = 1
        sr = np.array(self.satisfied_requests_history).sum(0)
        sr[demands == 0] = 1
        stats['unsatisfied_ratio'] = 1 - sr/demands
        state_array = stats.loc[:, ['supply', 'demand', 'arrival', 'expense', 'remaining_budget', 'unsatisfied_ratio']].values.reshape(1, 10, 10, 1, stats.shape[1] - 2).astype(np.float32)
        return state_array

    def get_last_satisfied_requests(self):
        return self.stats['satisfied_requests'].sum()
        
            
    
    def create_nodes_dict(self, nodes_gdf):
        self.nodes_in_boxes = {i: list(nodes_gdf[nodes_gdf.within(box)]['osmid'].astype(str)) for i, box in self.boxes.iteritems()}
                
    @classmethod
    def from_graph(cls, g, n=10):
        bounds = g.nodes.total_bounds
        boxes, shape = cls.generate_grid(bounds, n=n)
        nodes_in_boxes = {i: list(g.nodes[g.nodes.within(box)]['osmid']) for i, box in boxes.iteritems()}            
        return cls(boxes, shape, nodes_in_boxes)
    @classmethod
    def generate_grid(cls, bounds, n):
        xmin, ymin, xmax, ymax = bounds
        xlims = np.linspace(xmin, xmax, n + 1)
        ylims = np.linspace(ymin, ymax, n + 1)

        boxes = [
            Polygon([(xlims[i], ylims[n - j - 1]), (xlims[i + 1], ylims[n - j - 1]), 
                    (xlims[i + 1], ylims[n - j]), (xlims[i], ylims[n - j])])
            for j, i in product(range(n), range(0, n))
        ]
        return gpd.GeoSeries(boxes), (n, n)

    @classmethod
    def from_gdf(cls, gdf, shape):
        return cls(gdf['geometry'], shape)

class ServiceProvider(Agent):
    
    def __init__(self, budget=20000, model=HRP(), buffer_length=100, name='HRP'):
        super(ServiceProvider, self).__init__(name=name, model=model, buffer_length=buffer_length)
        self.total_budget = budget
        self.budget = budget

    def expend(self, value):
        self.budget -= value

    def restore_budget(self):
        self.budget = self.total_budget

    def plot_history(self):
        fig, ax = plt.subplots(1, 2, figsize=(14, 6))
        ax[0].plot(self.history['rewards'], color='green', label='Observed Rewards')
        ax[0].set_xlabel('Episode')
        ax[0].set_xticks(range(0, len(self.history['rewards']) + 1, 10))
        ax[0].set_ylabel('Reward (N° Satisfied Requests)')
        ax[0].set_title('Rewards')
        ax[0].axvline(2, linestyle='--', label='Finish Warmup Stage')
        ax[0].legend()

        ax[1].plot(self.history['dqn_loss'], color='orange')
        ax[1].set_xlabel('Episode')
        ax[1].set_xticks(range(0, len(self.history['dqn_loss']) + 1, 10))
        ax[1].set_ylabel('Loss')
        ax[1].set_title('DQN Training Loss')
        return fig


        
class User:
    count = 0
    users = list()
    
    def __init__(self, trip):
        self.__class__.count += 1
        self.__class__.users.append(self)

        self.user_id = self.__class__.count
        self.origin = str(trip[0])
        self.destination = str(trip[1])
        self.reachable_method = 'distance'
        self.max_walking_distance = 500 # km

        self.velocity = 1.5 # m/s
        self.alpha = 9.0083e-7
        self.trip = {
                'id': self.user_id,
                'origin': self.origin,
                'destination': self.destination,
                'arrival_time': None,
                'walk': [],
                'walk_duration': None,
                'ride': [],
                'ride_duration': None,
                'pickup_time': None,
                'pickup_node': None,
                'scooter': {
                    'id': None,
                    'battery_level_pickup': None,
                    'battery_level_dropoff': None
                }
            }   
    def __str__(self):
        return "User {}, at {}, wanting to go to {}.".format(self.user_id,
                                                            self.origin,
                                                            self.destination)

    def reachable_nodes(self, simulator):
        if self.reachable_method == 'grid':
            return list(simulator.grid.nodes_within_area(self.origin))
        elif self.reachable_method == 'distance':
            sp = simulator.graph.shortest_paths['walk'].loc[self.origin]
            return list(sp[sp < self.max_walking_distance / self.velocity].index)
    def cost_function(self, scooter, simulator):        
        return self.alpha * simulator.graph.shortest_path_distance(self.origin, scooter.location, layer='walk') ** 2

class Scooter:
    count = 0
    scooters = list()
    battery_range = 20 # km
    speed = 2.16  # m/s
    def __init__(self, location):
        self.__class__.count += 1
        self.__class__.scooters.append(self)
        self.available = True
        self.scooter_id = self.__class__.count
        self.location = location
        self.battery_level = 100
        self.price_incentive = 10
        

    @classmethod
    def available_scooters(cls):
        return [scooter for scooter in cls.scooters if scooter.available]

    @classmethod
    def init_supply(cls, network, n=200, strategy='demand'):
        """with open('data/poisson_params.pickle', 'br') as file:
            tract_counts = {tract: sum(list(probs.values())) 
                            for tract, probs in pickle.load(file).items()}"""
        if strategy == 'demand':
            demand = pd.read_csv('data/starting_locations.csv')
            bounds = gpd.read_file('shapes/study_area/study_area.shp').to_crs('EPSG:4326').loc[0, 'geometry']
            demand['geometry'] = [Point(x, y) for _, x, y in demand[['StartLongitude', 'StartLatitude']].itertuples()]
            demand = gpd.GeoDataFrame(demand, crs='EPSG:4326')
            demand = demand[demand.within(bounds)]
            Scooter.demand = demand
            points = demand['geometry'].sample(n=n, weights=demand['Count'])
            locations = list(network.get_nearest_osmids(points, transfer=True).astype(str))
        else:
            locations = choices(list(network.transfer_nodes), k=n)
                
        Scooter.scooters = [cls(loc) for loc in locations]
        with open('visualization/data/scooter_locations.json', 'w') as file:
            json.dump(locations, file)

    def get_price_incentive(self, grid):
        self.price_incentive = grid.get_scooter_price(self)
        return self.price_incentive

class Event(ABC):

    def __init__(self, time):
        self.time = time

    def __lt__(self, other):
        return self.time < other.time

    def __le__(self, other):
        return self.time <= other.time

    @abstractmethod
    def execute(self):
        pass

class UserRequest(Event):

    requests_by_hour = 60 * 2
    total_requests = 0

    def __init__(self, user, time):
        super(UserRequest, self).__init__(time=time)
        self.user = user
        self.user.trip['arrival_time'] = time
        self.__class__.total_requests += 1

        # update grid stats
        

    def execute(self, simulator):
        time = simulator.time
        origin = simulator.graph.find_node(osmid=self.user.origin, layer='walk')
        simulator.grid.update_stat(self.user.origin, 'request')

        reachable_nodes = self.user.reachable_nodes(simulator)
        available_scooters = [scooter for scooter in Scooter.available_scooters() 
                                if scooter.location in reachable_nodes and scooter.battery_level > 0]
        available_locations =  [s.location for s in available_scooters]
        if len(available_scooters) == 0 and simulator.pricing:
            # No available scooters and pricing.
            if simulator.verbose == 2:
                print('No available Scooters within area.')
            nodes_within_neighbor = simulator.grid.nodes_within_neighbors(origin['osmid'])
            available_scooters = [scooter for scooter in Scooter.available_scooters()
                            if scooter.location in nodes_within_neighbor and scooter.battery_level > 0]

            user_utility = [scooter.get_price_incentive(simulator.grid) - self.user.cost_function(scooter, simulator)
                            for scooter in available_scooters]
            if np.any(np.array(user_utility) > 0):
                max_utility = np.argmax(user_utility)
                nearest_scooter = available_scooters[max_utility]
                incentive = nearest_scooter.get_price_incentive(simulator.grid)
                if simulator.verbose:
                    print('User recives an incentive of {:.2f}$.'.format(incentive))
                
                
                distance = simulator.graph.shortest_path_distance(self.user.origin,nearest_scooter.location)
                walking_time = distance / self.user.velocity
                pickup = PickUp(simulator.time + walking_time, self.user, nearest_scooter)
                simulator.grid.update_stat(nearest_scooter.location, 'pricing', value=incentive)
                simulator.service_provider.expend(incentive)
                self.user.trip['pickup_node'] = nearest_scooter.location
                self.user.trip['walk'] = simulator.graph.shortest_path_edges(self.user.origin, nearest_scooter.location)
                simulator.insert(pickup)
            
        elif not len(available_scooters) == 0:
            # Available scooter within area
            nearest_location, time = simulator.graph.get_nearest_node(self.user.origin, available_locations)
            nearest_index = available_locations.index(nearest_location)
            nearest_scooter = available_scooters[nearest_index]            
            pickup = PickUp(simulator.time + time, self.user, nearest_scooter)
            self.user.trip['walk_duration'] = time
            self.user.trip['pickup_node'] = nearest_location
            self.user.trip['walk'] = simulator.graph.shortest_path_edges(self.user.origin, nearest_location)

            simulator.insert(pickup)

        else:
            simulator.insert(UserLeavesSystem(simulator.time, self.user)) 
            return None

    def __str__(self):
        return "User {} arriving at node {}".format(self.user.user_id, self.user.origin)
        
    @classmethod
    def inter_requests_time(cls, area, day, hour):
        return Random.exponential(36000 / cls.requests_by_day_hour[area][day, hour])

    @classmethod
    def init_user_requests(cls, simulator):
        with open('data/poisson_params.pickle', 'rb') as file:
            cls.requests_by_day_hour = pickle.load(file)
        for area in cls.requests_by_day_hour.keys():
            # for each area take the first occurence of (day, hour)
            # to initialize user requests
            day, hour = list(cls.requests_by_day_hour[area].keys())[0]
            time = (24 * 3600) * day + 3600 * hour
            time += simulator.time
            simulator.insert(cls(time, day, hour, area, simulator))

class PickUp(Event):

    satisfied_requests = 0

    def __init__(self, time, user, scooter, incentive=False):
        super(PickUp, self).__init__(time=time)
        self.user = user
        self.scooter = scooter
        self.satisfied = False
        self.incentive = incentive
    
    def execute(self, simulator):
        trip_time = simulator.graph.shortest_path_time(self.scooter.location, self.user.destination, layer='bike')
        battery_consumption = (100 * ((trip_time * Scooter.speed / 1000) / Scooter.battery_range))
        origin = simulator.graph.g['bike'].vs.find(osmid=self.scooter.location)

        if self.scooter.available and battery_consumption <= self.scooter.battery_level:
            if self.incentive:
                simulator.service_provider.budget -= self.scooter.price_incentive
                simulator.grid.update_stat(origin['osmid'], 'pricing', expense=self.scooter.price_incentive)
            self.satisfied = True
            self.__class__.satisfied_requests += 1
            self.scooter.available = False

            
            simulator.grid.update_stat(origin['osmid'], 'pickup')

            self.user.trip['pickup_time'] = simulator.time
            self.user.trip['scooter']['id'] = self.scooter.scooter_id
            self.user.trip['scooter']['battery_level_pickup'] = self.scooter.battery_level          
            self.user.trip['ride'] += simulator.graph.shortest_path_edges(self.scooter.location, self.user.destination, layer='bike')
            self.user.trip['ride_duration'] = trip_time
            trip_duration = simulator.graph.shortest_path_time(self.scooter.location, self.user.destination, layer='bike')
            self.scooter.battery_level -= battery_consumption
            self.user.trip['scooter']['battery_level_dropoff'] = self.scooter.battery_level
            dropoff = Dropoff(self.user, self.scooter, simulator.time + trip_duration)
            simulator.insert(dropoff)
        
        else:
            simulator.insert(UserLeavesSystem(simulator.time, self.user))

    def __str__(self):
        if self.satisfied:
            return "User {} picking scooter {} at location {}".format(self.user.user_id, self.scooter.scooter_id, self.scooter.location)
        else:
            return "Scooter not available, user {} leaves system.".format(self.user.user_id)

class Dropoff(Event):

    def __init__(self, user, scooter, time):
        super(Dropoff, self).__init__(time=time)
        self.user = user
        self.scooter = scooter

    def execute(self, simulator):
        self.scooter.location = self.user.destination
        self.scooter.available = True
        destination = simulator.graph.g['bike'].vs.find(osmid=self.user.destination)
        simulator.grid.update_stat(destination['osmid'], 'arrival')
        simulator.insert(UserLeavesSystem(simulator.time, self.user))
        if self.scooter.battery_range <= 15:
            current_day = self.time // (24 * 3600)
            day_seconds = self.time % (24 * 3600)
            day_hour = day_seconds // 3600
            recharge_start_hours = current_day * (24 * 3600) + 21 * 3600  # starts at 21:00hrs of the current day
            if day_hour >= 21:
                recharge_seconds = (24*3600 - day_seconds * np.random.uniform())
                recharge_time = self.time + recharge_seconds
            else:
                recharge_time = recharge_start_hours + (3 * 3600) * np.random.uniform()


            recharge_event = ChargeScooter(self.scooter, time=recharge_time)
            simulator.insert(recharge_event)

    def __str__(self):
        return "User {} arriving at destination {}. Scooter Level: {:.2f}%".format(self.user.user_id, self.user.destination, self.scooter.battery_level)

class ChargeScooter(Event):
    def __init__(self, scooter, time):
        super(ChargeScooter, self).__init__(time=time)
        self.scooter = scooter

    def execute(self, simulator):
        self.scooter.available = False
        current_day = self.time // (24 * 3600)
        release_time = current_day + ((24 + 7) * 3600) # 7am of next day
        release_event = ReleaseScooter(self.scooter, release_time)
        simulator.insert(release_event)

    def __str__(self):
        return "Recharging scooter {}. Current battery level {:.2f}%.".format(self.scooter.scooter_id, self.scooter.battery_level)

class ReleaseScooter(Event):
    def __init__(self, scooter, time):
        super(ReleaseScooter, self).__init__(time=time)
        self.scooter = scooter
        
    def execute(self, simulator):
        self.point = Scooter.demand['geometry'].sample(n=1, weights=Scooter.demand['Count'])
        self.new_location = list(simulator.graph.get_nearest_osmids(self.point, transfer=True).astype(str))[0]
        self.scooter.location = self.new_location
        self.scooter.battery_level = np.random.uniform(95, 100)
        self.scooter.available = True
    
    def __str__(self):
        return "Releasing scooter {} at {}".format(self.scooter.scooter_id, self.new_location)

class RunPricing(Event):

    inter_status_time = 3600 # 1 hour
    memory = deque(maxlen=8)

    def __init__(self, time, plot=False, save=False, train=False, batch_size=16, warmup_iterations=100):
        super(RunPricing, self).__init__(time=time)
        self.plot = plot
        self.save = save
        self.train = train
        self.batch_size = batch_size
        self.warmup_iterations = warmup_iterations

    def execute(self, simulator):
        print('State!')
        state = self.get_state_from_grid(simulator)
        if self.plot:
            fig, ax = plt.subplots(figsize=(10, 10))
            sns.heatmap(self.region_service_level.values.reshape(10, 10),
                        square=True, annot=True, fmt='.2f', cmap='Greens', ax=ax)
            fig.savefig('figures/service_level_{}_{}.png'.format(simulator.day, simulator.hour))
        
        RunPricing.memory.append(state)
        if self.save:
            np.save('output/stats_{}_{}.npy'.format(simulator.day, simulator.hour),
                                             state)
        if len(RunPricing.memory) >= 3:
            if self.train and self.warmup_iterations > 0:
                simulator.service_provider.train_minibatch(self.batch_size)
        new_pricing = RunPricing(self.time + RunPricing.inter_status_time, self.warmup_iterations - 1)
        simulator.insert(new_pricing)
        return True

    def __str__(self):
        return 'Getting stats:\n Service level: {:.2f} %'.format(self.service_level * 100)

    def get_state_from_grid(self, simulator):
        # get status per grid region
        return simulator.grid.get_state(simulator)        

    @classmethod
    def init_stats(cls, simulator):
        simulator.insert(cls(3600))

class ListQueue:

    def __init__(self):
        self.elements = list()

    def insert(self, x):
        bisect.insort(self.elements, x)

    def remove_first(self):
        try:
            x = self.elements.pop(0)
            return x
        except IndexError:
            print('No more events in queue.')
            return None

    def remove(self, x):
        try:
            i = self.elements.index(x)
            return self.elements.pop(i)
        except ValueError:
            print('Event not in queue.')
            return None

    def __len__(self):
        return len(self.elements)

    def reset(self):
        self.elements = list()

    @property
    def next_time(self):
        return self.elements[0].time

class UserLeavesSystem(Event):

    def __init__(self, time, user):
        super(UserLeavesSystem, self).__init__(time=time)
        self.user = user

    def execute(self, simulator):
        if simulator.trip_saver:
            simulator.trip_saver.store_trip(self.user.trip)
    def __str__(self):
        return 'User {} leaving the system.'.format(self.user.user_id)

class TripsSaver:
    def __init__(self, name='austin'):
        self.name = name
        self.trips = {
            'name': name,
            'trips': []
        }
    def store_trip(self, trip):
        self.trips['trips'].append(trip)

    def save(self):
        print('Saving Trips to JSON')
        with open('visualization/data/' + self.name + '.json', 'w') as file:
            json.dump(self.trips, file)

class ScooterSharingSimulator:

    def __init__(self, graph, grid, initial_supply=200, days=7, month='march', year=2019, trip_saver=None, pricing=False):
        self.pricing = pricing
        self.replicas = replicas
        self.events = ListQueue()
        self.time = 0
        self.simulation_time = days * 24 * 3600
        self.graph = graph
        self.grid = grid
        self.trip_saver = trip_saver
        self.time_window = 3600
        self.timesteps = self.simulation_time // self.time_window
        if self.pricing:
            try:
                self.service_provider = ServiceProvider().load_trained()
            except:
                self.service_provider = ServiceProvider()
        self.initial_supply = initial_supply
        Scooter.init_supply(self.graph, n=initial_supply)
        #UserRequest.init_user_requests(self)
        RunPricing.init_stats(self)

    @property
    def day(self):
        return (self.time // (3600 * 24)) % 7

    @property
    def hour(self):
        return (self.time % (3600 * 24)) // 3600

    def clock(self):
        return str(datetime.timedelta(seconds=self.time)).split('.')[0]

    def insert(self, event):
        self.events.insert(event)

    def insert_events(self, events):
        for event in events:
            self.insert(event)

    def cancel(self, event):
        return self.events.remove(event)

    def perform_action(self, action, verbose=0):
        self.verbose=verbose
        self.grid.set_price(action)
        self.run_timestep(training=True, verbose=verbose)
        state = self.grid.get_state(self)
        reward = self.grid.get_last_satisfied_requests()
        self.grid.reset_stats()
        return state, reward
        
    def do_all_events(self, verbose=0, training=False):
        print('doing all events')
        print(verbose)
        total_time = str(datetime.timedelta(seconds=self.simulation_time)).split('.')[0]
        if not training:
            print('Start Simulator. {} hours'.format(total_time))
        self.verbose = verbose
        while len(self.events) > 0 and self.time < self.simulation_time:
            
            event = self.events.remove_first()
            self.time = event.time         
            
            result = event.execute(self)
            if (verbose == 2) or (verbose == 1 and isinstance(event, RunPricing)):
                print("\nTime: {}".format(self.clock()))
                print(event)
            if result and training:
                print('Break Loop')
                return self.grid.get_last_satisfied_requests()

    def run_timestep(self, verbose=0, training=True):
        starting_time = self.time
        ending_time = starting_time + self.time_window
        print('Timestep: {}h - {}h'.format((starting_time//3600%24), (ending_time//3600)%24))
        while len(self.events) > 0 and self.events.next_time <= ending_time:
            event = self.events.remove_first()
            self.time = event.time
            result = event.execute(self)
            if (verbose == 2) or (verbose == 1 and isinstance(event, RunPricing)):
                print("\nTime: {}".format(self.clock()))
                print(event)
        self.time = ending_time

    def simulate(self, replicas, verbose=0):
        for replica in replicas:
            print('Replica: ', replica)
            self.trip_reader = TripReader(replica)
            arrivals = self.trip_reader.construct_events(self)
            self.events.reset()
            self.insert_events(arrivals)
            self.do_all_events(verbose=2)

    def set_replicas_for_training(self, replicas):
        self.replicas = replicas
        self.n_replicas = len(replicas)
        self.current_replica = 0
        """for replica in replicas:
            print('Replica: ', replica)
            self.trip_reader = TripReader(replica)
            arrivals = self.trip_reader.construct_events(self)
            self.events.reset()
            self.insert_events(arrivals)"""
    def reset(self):
        Scooter.init_supply(self.graph, n=self.initial_supply)
        #UserRequest.init_user_requests(self)
        RunPricing.init_stats(self)
        self.service_provider.restore_budget()
        replica = self.replicas[self.current_replica]
        self.current_replica = (self.current_replica + 1) % self.n_replicas
        print('Replica: ', replica)
        self.trip_reader = TripReader(replica)
        arrivals = self.trip_reader.construct_events(self)
        self.time = 0
        self.events.reset()
        self.insert_events(arrivals)


    def save_trips(self):
        self.trip_saver.save()

    def get_state(self):
        return self.grid.get_state(self)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--simulate', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test_grid', action='store_true')
    parser.add_argument('--pricing', action='store_true')

    args = parser.parse_args()
    if args.simulate:
        replicas = ['data/replicas/stkde_nhpp_{}.csv'.format(i) for i in range(1)]
        trip_saver = TripsSaver(name='test')
        study_area_filename = 'shapes/study_area/study_area.shp'
        study_area = gpd.read_file(study_area_filename).to_crs('epsg:4326')
        
        study_area_polygon = study_area.iloc[0]['geometry']
        graph = MultiModalNetwork.from_polygon(study_area_polygon, speeds={'walk': 1.4, 'bike':2.16})
        grid_gdf = gpd.read_file('shapes/grid/grid_500m.shp').to_crs('epsg:4326')
        
        grid = Grid.from_gdf(grid_gdf, (10,10))
        grid.create_nodes_dict(graph.layers['walk']['nodes'])
        simulator = ScooterSharingSimulator(graph, grid, trip_saver=trip_saver, initial_supply=100, pricing=args.pricing)
        simulator.simulate(replicas, verbose=True)
        simulator.save_trips()
    if args.train:
        replicas = ['data/replicas/stkde_nhpp_{}.csv'.format(i) for i in range(2)]
        trip_saver = TripsSaver(name='test')
        study_area_filename = 'shapes/study_area/study_area.shp'
        study_area = gpd.read_file(study_area_filename).to_crs('epsg:4326')
        
        study_area_polygon = study_area.iloc[0]['geometry']
        graph = MultiModalNetwork.from_polygon(study_area_polygon, speeds={'walk': 1.4, 'bike':2.16})
        grid_gdf = gpd.read_file('shapes/grid/grid_500m.shp').to_crs('epsg:4326')
        
        grid = Grid.from_gdf(grid_gdf, (10,10))
        grid.create_nodes_dict(graph.layers['walk']['nodes'])

        model = HRP()
        agent = ServiceProvider()
        environment = ScooterSharingSimulator(graph, grid, days=1, initial_supply=100, pricing=True)
        environment.set_replicas_for_training(replicas)
        agent.train(environment, warmup_iterations=5, episodes=20)
        agent.save_target_model()
        fig = agent.plot_history()
        fig.savefig('training-history.png')
        
    if args.test_grid:
        replicas = ['data/replicas/stkde_nhpp_{}.csv'.format(i) for i in range(1)]
        trip_saver = TripsSaver(name='test')
        study_area_filename = 'shapes/study_area/study_area.shp'
        study_area = gpd.read_file(study_area_filename).to_crs('epsg:4326')
        print(study_area.crs)
        study_area_polygon = study_area.iloc[0]['geometry']
        graph = MultiModalNetwork.from_polygon(study_area_polygon, speeds={'walk': 1.4, 'bike':2.16})
        grid_gdf = gpd.read_file('shapes/grid/grid_500m.shp').to_crs('epsg:4326')
        print(grid_gdf.crs)
        grid = Grid.from_gdf(grid_gdf, (10,10))
        grid.create_nodes_dict(graph.layers['walk']['nodes'])
        print(grid.get_area('5159078118'))
        print(grid.nodes_in_boxes)
        print(grid.nodes_in_boxes[0])