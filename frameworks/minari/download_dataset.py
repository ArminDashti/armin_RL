import minari

from base import Base


class DownloadDataset(Base):
    def __init__(self, env_id):
        self.env_id = env_id
    
    
    def download(self):
        self.dataset = minari.load_dataset(self.env_id, download=True)
        return self.dataset.iterate_episodes()
    
    
