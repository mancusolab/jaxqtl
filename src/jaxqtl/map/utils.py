import equinox as eqx
import pandas as pd

class Aggregator(eqx.Module):
    frames: list = []

    def add_row(self, row: dict):
        # cheap 1-row DataFrame, but *only* created when needed
        self._frames.append(pd.DataFrame([row]))

    def add_df(self, df: pd.DataFrame):
        self._frames.append(df)

    def to_dataframe(self):
        return pd.concat(self._frames, ignore_index=True)
