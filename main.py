import time
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd

from multimodal_network import MultiModalNetwork
from simulation import ScooterSharingSimulator, HistorySaver, ServiceProvider, ServiceProviderWeek, Grid
from models_torch import HRP, RandomPricing



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--simulate', action='store_true')
    parser.add_argument('--days', type=int, default=1)
    parser.add_argument('--supply', type=int, default=80)
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test_grid', action='store_true')
    parser.add_argument('--pricing', type=str, default=None)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--episodes', type=int, default=15)
    parser.add_argument('--noise', type=float, default=2)
    parser.add_argument('--replicas', type=int, default=20)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--budget', type=int, default=100)
    parser.add_argument('--model', type=str, default='100ep_250bg_sigm')
    parser.add_argument('--verbose', type=int, help='verbosity value')
    parser.add_argument('--name', type=str, help='name of agent')
    parser.add_argument('--actor_lr', type=float, default=1e-4)
    parser.add_argument('--critic_lr', type=float, default=1e-4)
    parser.add_argument('--max_action', type=float, default=5)

    args = parser.parse_args()
    if args.simulate:
        replicas = ['data/replicas/stkde_nhpp_{}.csv'.format(i) for i in range(args.replicas)]
        
        study_area_filename = 'shapes/study_area/study_area.shp'
        study_area = gpd.read_file(study_area_filename).to_crs('epsg:4326')
        
        study_area_polygon = study_area.iloc[0]['geometry']
        graph = MultiModalNetwork.from_polygon(study_area_polygon, speeds={'walk': 1.4, 'bike':2.16})
        grid_gdf = gpd.read_file('shapes/grid/grid_500m.shp').to_crs('epsg:4326').sort_values('id')
        
        grid = Grid.from_gdf(grid_gdf, (10,10))
        grid.create_nodes_dict(graph.layers['walk']['nodes'].to_crs('epsg:4326'))
        if args.pricing == 'HRP':
            agent = ServiceProviderWeek.load(name=args.model)
            agent.method = args.pricing
            agent.eval()
            simulator = ScooterSharingSimulator(graph, grid, days=args.days, initial_supply=args.supply, pricing=True, service_provider=agent)
        elif args.pricing == 'random':
            model = RandomPricing()
            agent = ServiceProvider(model=model, budget=args.budget)
            agent.method = args.pricing
            simulator = ScooterSharingSimulator(graph, grid, days=args.days, initial_supply=args.supply, pricing=True, service_provider=agent)
        else:
            simulator = ScooterSharingSimulator(graph, grid, days=args.days, initial_supply=args.supply, pricing=False)
        
        simulator.simulate(replicas, verbose=1)
    if args.train:
        replicas = ['data/replicas/stkde_nhpp_{}.csv'.format(i) for i in range(args.replicas)]
        history_saver = HistorySaver(name='test')
        study_area_filename = 'shapes/study_area/study_area.shp'
        study_area = gpd.read_file(study_area_filename).to_crs('epsg:4326').sort_values('id')
        
        study_area_polygon = study_area.iloc[0]['geometry']
        graph = MultiModalNetwork.from_polygon(study_area_polygon, speeds={'walk': 1.4, 'bike':2.16})
        grid_gdf = gpd.read_file('shapes/grid/grid_500m.shp').to_crs('epsg:4326')
        
        grid = Grid.from_gdf(grid_gdf, (10,10))
        grid.create_nodes_dict(graph.layers['walk']['nodes'])
        agent = ServiceProvider(model=HRP, noise_scale=args.noise, budget=args.budget, buffer_length=1000, batch_size=args.batch,
        actor_lr=args.actor_lr, critic_lr=args.critic_lr, max_action=args.max_action)
        environment = ScooterSharingSimulator(graph, grid, days=args.days, initial_supply=args.supply, pricing=True, service_provider=agent)
        environment.set_replicas_for_training(replicas)
        agent.train(environment, warmup_iterations=args.warmup, episodes=args.episodes)
        agent.save_agent(name=args.name)