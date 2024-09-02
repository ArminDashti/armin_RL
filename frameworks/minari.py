import minari



def dataset(env):
    return minari.load_dataset(env, download=True)
    
    
    