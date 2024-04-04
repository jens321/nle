import time

import pyarrow.parquet as pq
import pyarrow as pa
from nle.nethack import TERMINAL_SHAPE, INV_SIZE, DUNGEON_SHAPE, BLSTATS_SHAPE, MESSAGE_SHAPE
import numpy as np

class ParquetConverter():
    def __init__(self):
        self.gameid = 0
        self.part = 0
        self.cursor = 0

        obs_schema_keys = [
            ('glyphs', pa.list_(pa.int16())),
            ('inv_glyphs', pa.list_(pa.int16())),
            ('message', pa.list_(pa.uint8())),
            ('blstats', pa.list_(pa.int64())),
            ('tty_chars', pa.list_(pa.uint8())),
            ('tty_colors', pa.list_(pa.int8())),
            ('tty_cursor', pa.list_(pa.uint8())),
            ('rewards', pa.int32()),
            ('actions', pa.uint8()),
            # ('specials', pa.list_(pa.uint8())),
            # ('program_state', pa.list_(pa.int32())),
            # ('internal', pa.list_(pa.int32())),
            # ('inv_strs', pa.list_(pa.uint8())),
            # ('inv_letters', pa.list_(pa.uint8())),
            # ('inv_oclasses', pa.list_(pa.uint8())),
            # ('screen_descriptions', pa.list_(pa.uint8())),
            # ('misc', pa.list_(pa.int32())),
            # # potential additional keys to store
            # ('attributes', pa.list_(pa.uint8())),
            # ('enhance', pa.list_(pa.uint8())),
            # ('terrain', pa.list_(pa.uint8())),
            # ('known', pa.list_(pa.uint8()))
        ]
        self.obs_schema = pa.schema(obs_schema_keys)

    def load_parquet(self, path: str):
        full_path_parts = path.split("/")
        filename = full_path_parts[-1]
        filename_parts = filename.split(".")
        pid = filename_parts[1]
        epi = filename_parts[2]
        path = "/".join(full_path_parts[:-1]) + f"/glyphs_pid_{pid}_epi_{epi}.parquet"

        # Load stuff 
        self.table = pq.read_table(
            path, 
            columns=['tty_chars', 'tty_colors', 'tty_cursor', 'actions', 'blstats', 'inv_glyphs', 'glyphs', 'message'], 
            memory_map=True,
            schema=self.obs_schema
        )
        num_rows = self.table.num_rows - 1

        self.chars = pa.compute.list_flatten(self.table['tty_chars'][:-1]).to_numpy()
        self.chars = self.chars.reshape(num_rows, TERMINAL_SHAPE[0], TERMINAL_SHAPE[1])

        self.colors = pa.compute.list_flatten(self.table['tty_colors'][:-1]).to_numpy()
        self.colors = self.colors.reshape(num_rows, TERMINAL_SHAPE[0], TERMINAL_SHAPE[1])
    
        self.cursors = pa.compute.list_flatten(self.table['tty_cursor'][:-1]).to_numpy()
        self.cursors = self.cursors.reshape(num_rows, 2)

        self.actions = self.table['actions'][:-1].to_numpy()

        self.blstats = pa.compute.list_flatten(self.table['blstats'][:-1]).to_numpy()
        self.blstats = self.blstats.reshape(num_rows, *BLSTATS_SHAPE)

        self.message = pa.compute.list_flatten(self.table['message'][:-1]).to_numpy()
        self.message = self.message.reshape(num_rows, *MESSAGE_SHAPE)

        self.inv_glyphs = pa.compute.list_flatten(self.table['inv_glyphs'][:-1]).to_numpy()
        self.inv_glyphs = self.inv_glyphs.reshape(num_rows, *INV_SIZE)

        self.glyphs = pa.compute.list_flatten(self.table['glyphs'][:-1]).to_numpy()
        self.glyphs = self.glyphs.reshape(num_rows, *DUNGEON_SHAPE)

        # Reset cursor
        self.cursor = 0

    def convert(self, chars, colors, curs, actions, blstats, inv_glyphs, glyphs, message):
        input_len = chars.shape[0]
        end_cursor = self.chars.shape[0]
        to_read = min(input_len, end_cursor - self.cursor)

        np.copyto(chars[:to_read], self.chars[self.cursor: self.cursor + to_read])
        np.copyto(colors[:to_read], self.colors[self.cursor: self.cursor + to_read])
        np.copyto(curs[:to_read], self.cursors[self.cursor: self.cursor + to_read])
        np.copyto(actions[:to_read], self.actions[self.cursor: self.cursor + to_read])
        np.copyto(blstats[:to_read], self.blstats[self.cursor: self.cursor + to_read])
        np.copyto(message[:to_read], self.message[self.cursor: self.cursor + to_read])
        np.copyto(inv_glyphs[:to_read], self.inv_glyphs[self.cursor: self.cursor + to_read])
        np.copyto(glyphs[:to_read], self.glyphs[self.cursor: self.cursor + to_read])

        self.cursor += to_read

        return input_len - to_read