import pandas as pd


class GeneMetaData:
    """Store gene meta data
    Gene name, chrom, start, rend
    """

    data: pd.DataFrame

    # def __init__(self, gene_path: str = './example/data/ensembl_allgenes.txt', window: int):
    #     gene_map = pd.read_csv(gene_path, delimiter='\t')
    #     gene_map.columns = [
    #         "chr",
    #         "gene_start",
    #         "gene_end",
    #         "symbol",
    #         "tss_start",
    #         "strand",
    #         "gene_type",
    #         "ensemble_id",
    #         "refseq_id",
    #     ]
    #     gene_map["tss_left_end"] = gene_map["tss_start"] - window  # it's ok to be negative
    #     gene_map["tss_right_end"] = gene_map["tss_start"] + window
    #
    #
    # def __iter__(self, window: int):
    #     return self.gene_map
    #
    # def __getitem__(self, item: str):
    #
    #     return name, chrom, start_min, end_max


class ExpressionData:
    def __init__(self):
        pass

    def __iter__(self):
        pass

    def __getitem__(self, name):
        return self.X[name, :]
