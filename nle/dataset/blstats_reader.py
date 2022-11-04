import os

import numpy as np

class BlstatsReader():
    def __init__(self, gameid: int, blstats_root: str, game_path: str):
        self.gameid = gameid
        self.game_path = game_path
        self.blstats_root = blstats_root
        self.blstats_path = self._get_blstats_path()
        self.load(self.gameid, game_path)
        self.cursor = 0

    def read(self, blstats: np.array):
        input_len = blstats.shape[0]
        end_cursor = self.data.shape[0]
        to_read = min(input_len, end_cursor - self.cursor)
        np.copyto(blstats[:to_read], self.data[self.cursor:self.cursor + to_read])
        self.cursor += to_read

    def load(self, gameid: int, game_path: str):
        self.gameid = gameid
        self.game_path = game_path
        self.blstats_path = self._get_blstats_path()
        self.data = np.load(self.blstats_path)
        self.cursor = 0

    def _get_blstats_path(self):
        dataset_name = self.blstats_root.split('/')[-1]
        if dataset_name == 'nld-nao' or dataset_name == 'nld-aa':
            return os.path.join(self.blstats_root, str(self.gameid), f'blstats_{self.gameid}.npy')
        else:
            # example: /efs/tuylsjen/rl_planning/logs/torchbeast-20221104-202816/bootstrap-round-0/0/nle.61354.0.ttyrec3.bz2
            first_split = self.game_path.split('/')
            pid = first_split[-1].split('.')[1]
            last_folder = first_split[-2]
            return os.path.join(self.blstats_root, last_folder, f'blstats_{pid}.npy')



    
    