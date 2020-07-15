import time
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd

from multimodal_network import MultiModalNetwork
from simulation import ScooterSharingSimulator, HistorySaver, ServiceProvider, Grid
from models_torch import HRP



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--simulate', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test_grid', action='store_true')
    parser.add_argument('--pricing', action='store_true')
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--episodes', type=int, default=15)
    parser.add_argument('--noise', type=float, default=2)
    parser.add_argument('--replicas', type=int, default=20)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--budget', type=int, default=100)
    parser.add_argument('--model', type=str, default='100ep_250bg_sigm')
    parser.add_argument('--verbose', type=int, help='verbosity value')
    parser.add_argument('--name', type=str, help='name of agent')
    parser.add_argument('--lr', type=float, default=1e-4)

    args = parser.parse_args()
    if args.simulate:
        replicas = ['data/replicas/stkde_nhpp_{}.csv'.format(i) for i in range(args.replicas)]
        
        study_area_filename = 'shapes/study_area/study_area.shp'
        study_area = gpd.read_file(study_area_filename).to_crs('epsg:4326')
        
        study_area_polygon = study_area.iloc[0]['geometry']
        graph = MultiModalNetwork.from_polygon(study_area_polygon, speeds={'walk': 1.4, 'bike':2.16})
        grid_gdf = gpd.read_file('shapes/grid/grid_500m.shp').to_crs('epsg:4326').sort_values('id')
        
        grid = Grid.from_gdf(grid_gdf, (10,10))
        grid.create_nodes_dict(graph.layers['walk']['nodes'])
        if args.pricing:
            agent = ServiceProvider.load_agent(name=args.model)
            agent.model.eval()
            simulator = ScooterSharingSimulator(graph, grid, initial_supply=60, pricing=args.pricing)
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
        model = HRP(learning_rate=args.lr)
        agent = ServiceProvider(model=model, noise_scale=args.noise, budget=args.budget, buffer_length=1000, batch_size=args.batch)
        environment = ScooterSharingSimulator(graph, grid, days=1, initial_supply=100, pricing=True, service_provider=agent)
        environment.set_replicas_for_training(replicas)
        agent.train(environment, warmup_iterations=args.warmup, episodes=args.episodes)
        agent.save_agent(name=args.name)