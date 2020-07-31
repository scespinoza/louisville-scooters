import pandas as pd
import geopandas as gpd
from shapely.geometry import Point


convert_to_point = lambda p: Point(p[0], p[1])

def get_starting_locations(df, crs='EPSG:4326'):
    return gpd.GeoSeries(df[['StartLongitude', 'StartLatitude']].apply(convert_to_point, axis=1),
                                    crs=crs)
def get_ending_locations(df, crs='EPSG:4326'):
    return gpd.GeoSeries(df[['EndLongitude', 'EndLatitude']].apply(convert_to_point, axis=1),
                                crs=crs)
def filter_dataset(dataset, study_area, year=2019):
    print('Spatial Filter')
    temporal_filter = (dataset['date'].dt.year == year) & (dataset['date'].dt.month >= 4) & (dataset['date'].dt.month <= 10)
    dataset = dataset[temporal_filter]
    starting_locations = get_starting_locations(dataset)
    ending_locations = get_ending_locations(dataset)
    starting_filter = starting_locations.within(study_area)
    ending_filter = ending_locations.within(study_area)
    spatial_filter = starting_filter & ending_filter
    return dataset[spatial_filter]

def create_arrivals_gdf(data, crs='EPSG:4326'):
    data['geometry'] = get_starting_locations(data).copy()
    data = data[['date', 'geometry']]
    data = gpd.GeoDataFrame(data.sort_values(by='date').values, index=list(range(len(data))), columns=['date', 'geometry'])
    data['date'] = data['date'].astype(str)
    return gpd.GeoDataFrame(data[['date', 'geometry']], crs=crs)

if __name__ == '__main__':
    data = pd.read_csv('../data/DocklessTripOpenData_10.csv')
    print('Creating Timeseries...')
    date_str = data[['StartDate', 'StartTime']].apply(lambda x: str(x[0]) + ' ' + str(x[1]).replace('24:', '00:'), axis=1)
    data['date'] = pd.to_datetime(date_str)
    study_area = gpd.read_file('../shapes/study_area/study_area.shp').to_crs('EPSG:4326')
    study_area_polygon = study_area.loc[0, 'geometry']
    filtered_data = filter_dataset(data, study_area_polygon)
    filtered_data.to_csv('../data/data_2019.csv')
    print('Creating Arrivals Shapefile')
    arrivals_gdf = create_arrivals_gdf(filtered_data)
    arrivals_gdf.to_file('../shapes/arrivals/arrivals.shp')



