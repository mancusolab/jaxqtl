# import genomicranges
import pandas as pd


# TODO: find strand of imputed variant
# TODO: use which gene database, eg. emsembl? (onek1k has this identifier)
def cis_window_cutter(W, gene_start, gene_end, var_list):
    """
    return variant list in cis for given gene
    """
    gene_info = "./example/data/list_genes_qc.all.txt"
    df = pd.read_csv(gene_info, delimiter="\t")
    df = df[
        [
            "chr",
            "ensembl.start",
            "ensembl.end",
            "ensembl.strand",
            "name",
            "ensembl.ENSG",
        ]
    ]
    # format it as: seqnames, starts, end, strand
    df.columns = ["seqnames", "starts", "ends", "old_strand", "symbol", "ensembl_id"]
    gr_strand = ["+", "+", "-", "-", "*", "*", "*"]
    df_strand = ["1", "1|1", "-1", "-1|-1", "-1|1", "1|-1", "1|1|-1"]

    df["strand"] = df["old_strand"].replace(df_strand, gr_strand)

    # gr = genomicranges.fromPandas(df)  # convert to gr object

    return
