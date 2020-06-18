import time
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
from simulation import ScooterSharingSimulator, TripsSaver


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--initial_supply', type=int, default=200, 
                        help='initial supply of scooters')
    parser.add_argument('--days', type=int, default=7,
                        help='simulation time in hours')
    parser.add_argument('--month', type=str, default='march',
                        help='month to simulate')
    parser.add_argument('--year', type=int, default=2019,
                        help='year to simulate')
    parser.add_argument('--name', type=str, default='test', help='name of run')
    parser.add_argument('--verbose', type=int, default=0)
    args = parser.parse_args()

    print('Loading Simulator')

    trip_saver = TripsSaver(name=args.name)
    simulator = ScooterSharingSimulator(initial_supply=args.initial_supply,
                                        days=args.days,
                                        month=args.month,
                                        year=args.year,
                                        trip_saver=trip_saver)
    start = time.time()
    simulator.do_all_events(verbose=args.verbose)
    end = time.time()

    print('Elapsed Time: {} s'.format(end - start))
    simulator.save_trips()