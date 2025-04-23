### count data
# library(tidyverse)
library(dplyr); library(readr)
# library(tidyverse) # or load tidyverse
library(glue)

# TODO: put this script in the /code directory
# assume ./data/pheno is created
pheno_dir <- "../data/pheno"

# TODO: create a file with one column for all cell type names (NO header)
# note: no space is allowed in the cell type names, please use "_" to space
celltype_path <- "../data/pheno/celltype_list.tsv"

# create result dir if not exist
system("mkdir -p ../result")
system("mkdir -p ../result/cis")
res_dir <- "../result/cis"

# TODO: specify the number of genes per chunk
chunk <- 50

# read cell type file
allcelltypes <- read_tsv(celltype_path, F) %>%
  rename(celltype = X1) %>%
  pull(celltype)

system("mkdir -p ../data/genelist")
for (celltype in allcelltypes){
  # create cell type gene list
  system(glue("mkdir -p ../data/genelist/{celltype}"))

  # TODO: change to tsv
  pheno <- read_tsv(glue("{pheno_dir}/{celltype}.bed.gz"))
  colnames(pheno)[1] <- "chr"
  colnames(pheno)[4] <- "gene_id"

  for (chr_idx in unique(pheno$chr)){
    gene_pheno_chr <- pheno %>% filter(chr == chr_idx)
    n <- nrow(gene_pheno_chr)
    if (n < chunk){
      r <- rep(1, n)
    }else{
      r  <- rep(1:ceiling(n/chunk),each=chunk)[1:n]
    }
    d <- split(gene_pheno_chr,r)

    if (grepl("chr", tolower(chr_idx))){
      chr_idx <- tolower(chr_idx)
    }else{
      chr_idx <- paste0("chr", chr_idx)
    }
    system(glue("mkdir -p ../data/genelist/{celltype}/{chr_idx}"))

    for (i in 1:length(d)){
      d[[i]] %>%
        select(gene_id) %>%
        write_tsv(glue("../data/genelist/{celltype}/{chr_idx}/chunk_{i}"), col_names = F)
    }
  }
}

# prepare params file and result out directory
# setwd("../data/genelist")

params <- data.frame()
for (celltype in allcelltypes){
  allchr <- list.files(glue("../data/genelist/{celltype}"))

  # make result dir for cell type
  system(glue("mkdir -p {res_dir}/{celltype}"))

  for (chr in allchr){
    chr_idx <- gsub("chr", "", tolower(chr))
    allfiles <- list.files(glue("../data/genelist/{celltype}/chr{chr_idx}"))
    tmp <- tibble(cell_type = celltype, chr_col = chr_idx, files = allfiles)
    params <- bind_rows(params, tmp)

    # make result directory for chr in the cell type
    system(glue("mkdir -p {res_dir}/{celltype}/chr{chr_idx}"))
  }
}

params %>% write_tsv("../data/genelist/params_all_celltype", col_names = F)
