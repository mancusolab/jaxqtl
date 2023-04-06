import pandas as pd


class GeneMetaData:
    """Store gene meta data
    Gene name, chrom, start, rend
    """

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
