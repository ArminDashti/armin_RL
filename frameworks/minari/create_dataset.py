import minari
    
    
class CreateDataset:
    def __init__(self, algoritm='PPO'):
        self.algoritm = algoritm
    
    
    def create(env_id, num_steps, dataset_id=None):
        env = minari.DataCollector(gym.make(env_id))
        env.reset()

        for _ in range(num_steps):
            action = env.action_space.sample()
            obs, rew, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                env.reset()
    
        if dataset_id None:
            dataset = env.create_dataset(dataset_id=env_id)
        else:
            dataset = env.create_dataset(dataset_id=dataset_id)
        
        return dataset
    

    

