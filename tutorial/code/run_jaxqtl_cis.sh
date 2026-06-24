#!/bin/bash

data_path="../input"
out_path="../output"

celltype="CD4_NC"

# genelist to perform cis-eQTL mapping
chr=22
chunk_file="genelist_10"

# choose test method: score test (recommended) or wald
test_method="score"

window=500000 # default extend 500kb on either side, i.e., [start-window, end+window]
N=982

# jaxQTL by default compute expression PCs using the entire data provided in *.bed.gz
# to disable this, set this to 0
num_expression_pc=2

pheno="${data_path}/${celltype}.N100.bed.gz"
geno="${data_path}/chr${chr}_N100" # prefix for plink triplet files
covar="${data_path}/donor_features.tsv"

# choose gene list for eQTL mapping
genelist="${data_path}/${chunk_file}"

# choose eQTL model: NB for negative binomial, poisson, gaussian
model="nb"

# if using permutation method to calibrate gene-level p value, set number of permutation
nperm=1000

# prefix for output file
out="${out_path}/${celltype}_N${N}_chr${chr}_${chunk_file}_jaxqtl_${model}_cis_acat"

# viztracer \
# --tracer_entries 10000000 \
# -o results.json \
# -- jaxqtl \
jaxqtl \
 cis \
 --bfile ${geno} \
 --covar ${covar} \
 --pheno ${pheno} \
 --model ${model} \
 --gene-list ${genelist} \
 --test ${test_method} \
 --nperm ${nperm} \
 --acat \
 --set-offset-from-libsize \
 --normalize-covar \
 --out ${out}
