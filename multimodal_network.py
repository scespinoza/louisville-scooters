import time
import pickle
import igraph
import osmnx as ox
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt


from sklearn.neighbors import BallTree
from shapely.geometry import Point

def fix_duplicated_ids(edges):
    print('Fixing duplicates')
    edges['osmid'] = edges[['u', 'v', 'osmid']].apply(lambda x: str(x[0]) + '-' + str(x[1]) + '-' + str(x[2]), axis=1)
    return edges
    
def fix_isolated(edges, nodes):
    outputs = edges[['u', 'v']].groupby('u').size()
    inputs = edges[['u', 'v']].groupby('v').size()
    in_outs = inputs + outputs
    isolated_nodes = in_outs[in_outs <= 1].index.values
    isolated_edges = edges['u'].isin(isolated_nodes) | edges['v'].isin(isolated_nodes)
    if len(isolated_nodes) == 0:
        return edges, nodes
    else:
        return fix_isolated(edges[~isolated_edges], nodes[~nodes['osmid'].isin(isolated_nodes)])
    
    


def get_nearest(src_points, candidates, k_neighbors=1):
    """Find nearest neighbors for all source points from a set of candidate points"""
    # Create tree from the candidate points
    tree = BallTree(candidates, leaf_size=15, metric='haversine')

    # Find closest points and distances
    distances, indices = tree.query(src_points, k=k_neighbors)

    # Transpose to get distances and indices into arrays
    distances = distances.transpose()
    indices = indices.transpose()

    # Get closest indices and distances (i.e. array at index 0)
    # note: for the second closest points, you would take index 1, etc.
    closest = indices[0]
    closest_dist = distances[0]

    # Return indices and distances
    return (closest, closest_dist)
class MultiModalNetwork:

    def __init__(self, layers, speeds, directed):
        
        assert set(layers.keys()) == set(speeds.keys())
        self.layers = layers
        self.transfer_nodes = set.intersection(*(set(layers[layer]['nodes'].osmid.astype(str)) for layer in layers))
        for layer in layers.keys():
            self.layers[layer]['nodes']['transfer'] = self.layers[layer]['nodes']['osmid'].apply(lambda osmid: str(osmid) in self.transfer_nodes)
        self.speeds = speeds
        self.directed = directed
        self.shortest_paths = {}
        self.assign_velocities()
        self.create_graph()
        try:
            with open('data/shortest_paths_dict.pickle', 'rb') as file:
                self.shortest_paths = pickle.load(file)
        except:
            print('Shortest Paths dict not found')
            self.save_shortest_paths_all_layers()

    def assign_velocities(self, save=True):
        for layer_name, layer_gdfs in self.layers.items():
            layer_gdfs['edges']['time'] = layer_gdfs['edges']['length'] / self.speeds[layer_name]
            if save:
                layer_gdfs['edges'].to_file('shapes/{}/edges.shp'.format(layer_name))
    
    def get_nearest_osmids(self, points, layer='walk', transfer=False):
        if transfer:
            nodes = self.layers[layer]['nodes'][self.layers[layer]['nodes']['transfer']].copy().reset_index(drop=True)
        else:
            nodes = self.layers[layer]['nodes'].copy().reset_index(drop=True)
        left_radians = np.array(points.apply(lambda geom: (geom.x * np.pi / 180, geom.y * np.pi / 180)).to_list())
        right_radians = np.array(nodes['geometry'].apply(lambda geom: (geom.x * np.pi / 180, geom.y * np.pi / 180)).to_list())
        closest, dist = get_nearest(left_radians, right_radians)
        closest_points = nodes.loc[closest].reset_index(drop=True)
        return closest_points['osmid']
        
    def shortest_path_edges(self, u, v, layer='walk', output='epath'):
        origin = self.g[layer].vs.find(osmid=u)
        dest = self.g[layer].vs.find(osmid=v)
        sp = self.g[layer].get_shortest_paths(origin, to=dest, weights='length', mode=igraph.OUT, output=output)[0]
        return self.g[layer].es[sp]['osmid']
    def shortest_path_time(self, u, v, layer='walk'):
        return self.shortest_paths[layer].loc[str(u), str(v)]

    def get_nearest_node(self, source, dest, layer='walk'):
        this_sp = self.shortest_paths[layer].loc[str(source), dest]
        return this_sp.idxmin(), this_sp.min()

    def shortest_path_time(self, source, dest, layer='walk'):
        return self.shortest_paths[layer].loc[str(source), str(dest)]
    def shortest_path_distance(self, source, dest, layer='walk'):
        return self.shortest_path_time(source, dest, layer=layer) * self.speeds[layer]
    def create_graph(self):
        self.g = {}
        for layer_name, layer_gdfs in self.layers.items():
            print('Creating {} graph...'.format(layer_name))
            layer_graph = igraph.Graph(directed=self.directed[layer_name])
            layer_graph.add_vertices(list(layer_gdfs['nodes'].osmid.astype(str)))
            layer_graph.vs['coords'] = list(layer_gdfs['nodes']['geometry'])
            layer_graph.vs['osmid'] = list(layer_gdfs['nodes']['osmid'].astype(str))
            layer_graph.add_edges(list(zip(layer_gdfs['edges']['u'].astype(str), layer_gdfs['edges']['v'].astype(str))))
            layer_graph.es['osmid'] = list(layer_gdfs['edges']['osmid'].astype(str))
            layer_graph.es['length'] = list(layer_gdfs['edges']['length'].astype(float))
            layer_graph.es['oneway'] = list(layer_gdfs['edges']['oneway'])
            layer_graph.es['highway'] = list(layer_gdfs['edges']['highway'])
            layer_graph.es['time'] = list(layer_gdfs['edges']['time'])
            self.g[layer_name] = layer_graph

    def compute_shortest_paths(self, layer, weights='time'):
        print('Generating shortest path dict for {} layer.'.format(layer))
        start = time.time()
        shortest_paths_layer = self.g[layer].shortest_paths(weights=weights)
        print('done igraph')
        shortest_paths_df = pd.DataFrame(shortest_paths_layer, 
                                        index=self.layers[layer]['nodes'].osmid.astype(str), 
                                        columns=self.layers[layer]['nodes'].osmid.astype(str))
        self.shortest_paths[layer] = shortest_paths_df
        print('Elapsed time: {:.2f} s'.format(time.time() - start))
        return self.shortest_paths

    def find_node(self, osmid, layer):
        return self.g[layer].vs.find(osmid=osmid)

    def save_shortest_paths_all_layers(self, weights='time', filename='shortest_paths_dict'):
        for layer in self.layers.keys():
            self.compute_shortest_paths(layer=layer, weights=weights)
        with open('data/{}.pickle'.format(filename), 'wb') as file:
            pickle.dump(self.shortest_paths, file)

    def plot(self, ax=None, c={'bike': 'green', 'walk': 'blue'}):
        ax = ax or plt.gca()
        for layer_name, layer_gdfs in self.layers.items():
            layer_gdfs['edges'].plot(ax=ax, color=c[layer_name], alpha=0.5)
            layer_gdfs['nodes'].plot(ax=ax, color=c[layer_name], alpha=0.4, markersize=2)

        transfer_nodes = self.layers['bike']['nodes'][self.layers['bike']['nodes']['osmid'].apply(lambda osmid: osmid in self.transfer_nodes)]
        transfer_nodes.plot(ax=ax, c='yellow', alpha=0.3, markersize=5)
    @classmethod
    def from_polygon(cls, polygon, layers=['walk', 'bike'], save=True, speeds={'walk': 1., 'bike': 1.}, directed={'walk':True, 'bike':True}):
        layers_graph = {}
        for layer in layers:
            
            try:
            
                nodes = gpd.read_file('shapes/{}/nodes.shp'.format(layer))
                edges = gpd.read_file('shapes/{}/edges.shp'.format(layer))
            
            except:
            
                print('Files not found. Downloading...')
                nodes, edges = ox.graph_to_gdfs(ox.graph_from_polygon(polygon, network_type=layer, simplify=False))
                print(nodes.shape)
                edges = fix_duplicated_ids(edges)
                edges, nodes = fix_isolated(edges, nodes)
                print(nodes.shape)
                if save:
                    nodes.to_file('shapes/{}/nodes.shp'.format(layer))
                    edges.to_file('shapes/{}/edges.shp'.format(layer))
            
            layers_graph[layer] = {'nodes': nodes, 'edges': edges}

        return cls(layers_graph, speeds=speeds, directed=directed)


if __name__ == '__main__':
    study_area_filename = 'shapes/utils/Dockless Vehicle Service Area/Dockless_Vehicle_Service_Area.shp'
    study_area = gpd.read_file(study_area_filename).to_crs('EPSG:4326')
    study_area_polygon = study_area.iloc[0]['geometry']
    network = MultiModalNetwork.from_polygon(study_area_polygon, speeds={'walk': 1.4, 'bike':2.16}, directed={'walk': False, 'bike': True})
    fig, ax = plt.subplots(figsize=(13, 13))
    network.plot(ax=ax)
    plt.show()
    