import os

import numpy as np

class BlstatsReader():
    def __init__(self, gameid: int, dataset_name: str):
        self.gameid = gameid
        self.dataset_name = dataset_name
        self.blstats_root = "/efs/tuylsjen"
        self.blstats_path = os.path.join(self.blstats_root, self.dataset_name, f"blstats_{gameid}.npy")
        self.load(self.gameid)
        self.cursor = 0

    def read(self, blstats: np.array):
        input_len = blstats.shape[0]
        end_cursor = self.data.shape[0]
        to_read = min(input_len, end_cursor - self.cursor)
        blstats.copy_(self.data[self.cursor:self.cursor + to_read])
        self.cursor += to_read

    def load(self, gameid: int):
        self.gameid = gameid
        self.blstats_path = os.path.join(self.blstats_root, self.dataset_name, f"blstats_{gameid}.npy")
        self.data = np.load(self.blstats_path)
        self.cursor = 0


    
    