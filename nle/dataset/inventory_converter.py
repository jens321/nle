import time

import pyarrow.parquet as pq
import pyarrow as pa
from nle.nethack import INV_SIZE
import numpy as np

class InventoryConverter:
    def __init__(self):
        self.gameid = 0
        self.part = 0
        self.cursor = 0

        obs_schema_keys = [
            ('inv_glyphs', pa.list_(pa.int16())),
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
                columns=['inv_glyphs'], 
                memory_map=True,
                schema=self.obs_schema
            )
            num_rows = self.table.num_rows

            self.inv_glyphs = pa.compute.list_flatten(self.table['inv_glyphs']).to_numpy()
            self.inv_glyphs = self.inv_glyphs.reshape(num_rows, *INV_SIZE)

        except:
            # NOTE: sometimes we have corrupted parquet files. If so,
            # just fill with zeros. 
            print(f"Error loading parquet file: {inv_path}")
            self.inv_glyphs = np.zeros((int(2e5), *INV_SIZE), dtype=np.uint8)

        # Reset cursor
        self.cursor = 0

    def convert(self, inv_glyphs):
        input_len = inv_glyphs.shape[0]
        end_cursor = self.inv_glyphs.shape[0]
        to_read = min(input_len, end_cursor - self.cursor)

        np.copyto(inv_glyphs[:to_read], self.inv_glyphs[self.cursor: self.cursor + to_read])

        self.cursor += to_read

        return input_len - to_read