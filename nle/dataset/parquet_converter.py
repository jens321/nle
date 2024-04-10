import time
import copy

import pyarrow.parquet as pq
import pyarrow as pa
from pyarrow import compute
from nle.nethack import TERMINAL_SHAPE, INV_SIZE, DUNGEON_SHAPE, BLSTATS_SHAPE, MESSAGE_SHAPE
import numpy as np

class ParquetConverter():
    def __init__(self, unroll_length: int):
        self.gameid = 0
        self.part = 0
        self.cursor = 0

        self.unroll_length = unroll_length

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

        # read parquet file but not full table
        try:
            self.pq_file = pq.ParquetFile(path)
            self.num_rows = self.pq_file.metadata.num_rows
            self.table = self.pq_file.iter_batches(batch_size=self.unroll_length)
            self.batch = next(self.table)
            self.process_new_batch(self.batch)
        except:
            print(f"Error reading parquet file!")
            self.num_rows = 0


        # Reset cursor
        self.cursor = 0
        self.batch_cursor = 0

    def convert(self, chars, colors, curs, actions, blstats, inv_glyphs, glyphs, message):
        # old_chars = copy.deepcopy(chars)
        # old_colors = copy.deepcopy(colors)
        # old_curs = copy.deepcopy(curs)
        # old_actions = copy.deepcopy(actions)

        input_len = chars.shape[0]
        end_cursor = self.num_rows
        total_to_read = min(input_len, end_cursor - self.cursor)
        total_read = 0

        if total_to_read == 0:
            return input_len

        batch_to_read = min(input_len, len(self.batch) - self.batch_cursor)

        np.copyto(chars[:batch_to_read], self.chars[self.batch_cursor:self.batch_cursor + batch_to_read])
        np.copyto(colors[:batch_to_read], self.colors[self.batch_cursor:self.batch_cursor + batch_to_read])
        np.copyto(curs[:batch_to_read], self.cursors[self.batch_cursor:self.batch_cursor + batch_to_read])
        np.copyto(actions[:batch_to_read], self.actions[self.batch_cursor:self.batch_cursor + batch_to_read])
        np.copyto(blstats[:batch_to_read], self.blstats[self.batch_cursor:self.batch_cursor + batch_to_read])
        np.copyto(message[:batch_to_read], self.message[self.batch_cursor:self.batch_cursor + batch_to_read])
        np.copyto(inv_glyphs[:batch_to_read], self.inv_glyphs[self.batch_cursor:self.batch_cursor + batch_to_read])
        np.copyto(glyphs[:batch_to_read], self.glyphs[self.batch_cursor:self.batch_cursor + batch_to_read])

        total_read += batch_to_read
        self.batch_cursor += batch_to_read
        self.cursor += batch_to_read

        if batch_to_read < total_to_read:
            self.batch = next(self.table)
            self.process_new_batch(self.batch)
            self.batch_cursor = 0

            batch_to_read = min(input_len - total_read, len(self.batch) - self.batch_cursor)

            np.copyto(chars[total_read:total_read + batch_to_read], self.chars[self.batch_cursor:self.batch_cursor + batch_to_read])
            np.copyto(colors[total_read:total_read + batch_to_read], self.colors[self.batch_cursor:self.batch_cursor + batch_to_read])
            np.copyto(curs[total_read:total_read + batch_to_read], self.cursors[self.batch_cursor:self.batch_cursor + batch_to_read])
            np.copyto(actions[total_read:total_read + batch_to_read], self.actions[self.batch_cursor:self.batch_cursor + batch_to_read])
            np.copyto(blstats[total_read:total_read + batch_to_read], self.blstats[self.batch_cursor:self.batch_cursor + batch_to_read])
            np.copyto(message[total_read:total_read + batch_to_read], self.message[self.batch_cursor:self.batch_cursor + batch_to_read])
            np.copyto(inv_glyphs[total_read:total_read + batch_to_read], self.inv_glyphs[self.batch_cursor:self.batch_cursor + batch_to_read])
            np.copyto(glyphs[total_read:total_read + batch_to_read], self.glyphs[self.batch_cursor:self.batch_cursor + batch_to_read])

            total_read += batch_to_read
            self.batch_cursor += batch_to_read
            self.cursor += batch_to_read
        
        if self.batch_cursor == len(self.batch) and self.cursor < end_cursor:
            self.batch = next(self.table)
            self.process_new_batch(self.batch)
            self.batch_cursor = 0

        # try:
        #     # assert np.all(chars[:total_read] == old_chars[:total_read])
        #     # assert np.all(colors[:total_read] == old_colors[:total_read])
        #     # assert np.all(curs[:total_read] == old_curs[:total_read])
        #     breakpoint()
        #     assert np.all(actions[:total_read] == old_actions[:total_read])
        # except:
        #     breakpoint()

        return input_len - total_to_read
    
    def process_new_batch(self, batch):
        chars_flattened = compute.list_flatten(batch['tty_chars']).to_numpy()
        self.chars = chars_flattened.reshape(-1, TERMINAL_SHAPE[0], TERMINAL_SHAPE[1])

        colors_flattened = compute.list_flatten(batch['tty_colors']).to_numpy()
        self.colors = colors_flattened.reshape(-1, TERMINAL_SHAPE[0], TERMINAL_SHAPE[1])

        curs_flattened = compute.list_flatten(batch['tty_cursor']).to_numpy()
        self.cursors = curs_flattened.reshape(-1, 2)

        self.actions = batch['actions'].to_numpy()

        blstats_flattened = compute.list_flatten(batch['blstats']).to_numpy()
        self.blstats = blstats_flattened.reshape(-1, *BLSTATS_SHAPE)

        message_flattened = compute.list_flatten(batch['message']).to_numpy()
        self.message = message_flattened.reshape(-1, *MESSAGE_SHAPE)

        inv_glyphs_flattened = compute.list_flatten(batch['inv_glyphs']).to_numpy()
        self.inv_glyphs = inv_glyphs_flattened.reshape(-1, *INV_SIZE)

        glyphs_flattened = compute.list_flatten(batch['glyphs']).to_numpy()
        self.glyphs = glyphs_flattened.reshape(-1, *DUNGEON_SHAPE)