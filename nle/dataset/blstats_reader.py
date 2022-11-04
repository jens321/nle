import os

import numpy as np

class BlstatsReader():
    def __init__(self, gameid: int, blstats_root: str):
        self.gameid = gameid
        self.blstats_root = blstats_root
        self.blstats_path = os.path.join(blstats_root, f"blstats_{gameid}.npy")
        self.load(self.gameid)
        self.cursor = 0

    def read(self, blstats: np.array):
        input_len = blstats.shape[0]
        end_cursor = self.data.shape[0]
        to_read = min(input_len, end_cursor - self.cursor)
        np.copyto(blstats[:to_read], self.data[self.cursor:self.cursor + to_read])
        self.cursor += to_read

    def load(self, gameid: int):
        self.gameid = gameid
        self.blstats_path = os.path.join(self.blstats_root, f"blstats_{gameid}.npy")
        self.data = np.load(self.blstats_path)
        self.cursor = 0


    
    