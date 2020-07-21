
import time

def warmup(sp, env, iterations):
    returns = []
    for i in range(iterations):
        env.reset()
        episode_rewards = []
        print('Warmup Episode {}/{}'.format(i+1, iterations))
        for t in env.timesteps:
            day = t // 24
            day_sp = sp.sp_days[day]
            reward = day_sp.act(env)
            episode_rewards.append((self.model.discount_rate**t)*reward)
        returns.append(sum(episode_rewards))

def train(sp, env, iterations):
    for episode in range(iterations):
        env.reset()
        episode_rewards = []
        for t in range(env.timesteps):
            day = t // 24
            day_sp = sp.sp_days[day]
            reward = day_sp.act(environment, episode=e)
            episode_rewards.append((self.model.discount_rate**t)*reward)         
            batch_loss, distance = day_sp.train_minibatch()
            print('Episode {}/{}. Loss = {:.2f}. Reward = {:.2f}'.format(e, episodes, batch_loss.detach().cpu().numpy(), reward))
            day_sp.history['distance'].append(distance)
            day_sp.history['batch_loss'].append(batch_loss.detach().cpu().numpy())
        sp.history['rewards'].append(np.sum(episode_rewards))
    

def weekly_trainer(sp, env, warmup, episodes):
    warmup_start = time.time()
    print('Warmup Stage')
    returns = warmup(sp, env, warmup)
    sp.history = {
        'rewards': returns
    }
    print('Finishing Training: {:.2f}s'.format(time.time() - training_start))
    training_start = time.time()
    train(sp, env, episodes)
    sp.save('weights/weekly_model')
    print('Finishing Training: {:.2f}s'.format(time.time() - training_start))

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
        parser.add_argument('--critic_lr', type=float, default=1e-6)
        replicas = ['data/replicas/stkde_nhpp_{}.csv'.format(i) for i in range(args.replicas)]
        
        study_area_filename = 'shapes/study_area/study_area.shp'
        study_area = gpd.read_file(study_area_filename).to_crs('epsg:4326').sort_values('id')
        
        study_area_polygon = study_area.iloc[0]['geometry']
        graph = MultiModalNetwork.from_polygon(study_area_polygon, speeds={'walk': 1.4, 'bike':2.16})
        grid_gdf = gpd.read_file('shapes/grid/grid_500m.shp').to_crs('epsg:4326')
        
        grid = Grid.from_gdf(grid_gdf, (10,10))
        grid.create_nodes_dict(graph.layers['walk']['nodes'])
        agent = ServiceProviderWeek(model=model, noise_scale=args.noise, budget=args.budget, buffer_length=1000, batch_size=args.batch)
        environment = ScooterSharingSimulator(graph, grid, days=args.days, initial_supply=args.supply, pricing=True, service_provider=agent)
        environment.set_replicas_for_training(replicas)
        
        weekly_trainer(agent, environment, args.warmup, args.episodes)