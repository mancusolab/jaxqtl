import pandas as pd


class GeneMetaData:
    data: pd.DataFrame

    def __init__(self):
        pass

    def __iter__(self):
        pass


class ExpressionData:
    def __init__(self):
        pass

    def __iter__(self):
        pass

    def __getitem__(self, name):
        return self.X[name, :]
