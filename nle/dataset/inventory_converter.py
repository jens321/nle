import time

import pyarrow.parquet as pq
import pyarrow as pa
from nle.nethack import INV_STRS_SHAPE, INV_SIZE
import numpy as np

class InventoryConverter:
    def __init__(self):
        self.gameid = 0
        self.part = 0
        self.cursor = 0

        obs_schema_keys = [
            ('inv_strs', pa.list_(pa.uint8())),
            ('inv_letters', pa.list_(pa.uint8())),
            ('inv_oclasses', pa.list_(pa.uint8())),
        ]
        self.obs_schema = pa.schema(obs_schema_keys)

    def load_parquet(self, path: str):
        # example of path: /n/fs/nlp-il-scale/autoascend/nle_data/20240205-092247_d7q9uqm1/nle.438938.33.ttyrec3.bz2
        # break up an dextract the pid and episode number
        full_path_parts = path.split("/")
        filename = full_path_parts[-1]
        filename_parts = filename.split(".")
        pid = filename_parts[1]
        epi = filename_parts[2]
        inv_path = "/".join(full_path_parts[:-1]) + f"/inventory_pid_{pid}_epi_{epi}.parquet"

        # Load stuff
        try:
            self.table = pq.read_table(
                inv_path, 
                columns=['inv_strs', 'inv_letters', 'inv_oclasses'], 
                memory_map=True,
                schema=self.obs_schema
            )
            num_rows = self.table.num_rows

            self.inv_strs = pa.compute.list_flatten(self.table['inv_strs']).to_numpy()
            self.inv_strs = self.inv_strs.reshape(num_rows, *INV_STRS_SHAPE)

            self.inv_letters = pa.compute.list_flatten(self.table['inv_letters']).to_numpy()
            self.inv_letters = self.inv_letters.reshape(num_rows, *INV_SIZE)

            self.inv_oclasses = pa.compute.list_flatten(self.table['inv_oclasses']).to_numpy()
            self.inv_oclasses = self.inv_oclasses.reshape(num_rows, *INV_SIZE)
        except:
            # NOTE: sometimes we have corrupted parquet files. If so,
            # just fill with zeros. 
            print(f"Error loading parquet file: {inv_path}")
            self.inv_strs = np.zeros((int(2e5), *INV_STRS_SHAPE), dtype=np.uint8)
            self.inv_letters = np.zeros((int(2e5), *INV_SIZE), dtype=np.uint8)
            self.inv_oclasses = np.zeros((int(2e5), *INV_SIZE), dtype=np.uint8)

        # Reset cursor
        self.cursor = 0

    def convert(self, inv_strs, inv_letters, inv_oclasses):
        input_len = inv_strs.shape[0]
        end_cursor = self.inv_strs.shape[0]
        to_read = min(input_len, end_cursor - self.cursor)

        np.copyto(inv_strs[:to_read], self.inv_strs[self.cursor: self.cursor + to_read])
        np.copyto(inv_letters[:to_read], self.inv_letters[self.cursor: self.cursor + to_read])
        np.copyto(inv_oclasses[:to_read], self.inv_oclasses[self.cursor: self.cursor + to_read])

        self.cursor += to_read

        return input_len - to_read