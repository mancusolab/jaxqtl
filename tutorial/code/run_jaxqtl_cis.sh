cd ../..

data_path="./tutorial/input"
out_path="./tutorial/output"

celltype="CD4_NC"

# # genelist to perform cis-eQTL mapping
chr=22
chunk_file="genelist_10"

# choose test method: score test (recommended) or wald
test_method="score"

# choose cis or nominal scan
mode="cis" # nominal, cis
window=500000 # default extend 500kb on either side, i.e., [start-window, end+window]

# jaxQTL by default compute expression PCs using the entire data provided in *.bed.gz
# to disable this, set this to 0
num_expression_pc=2

pheno="${data_path}/${celltype}.N100.bed.gz"
geno="${data_path}/chr${chr}" # prefix for plink triplet files
covar="${data_path}/donor_features.all.6PC.tsv"

# choose gene list for eQTL mapping
genelist="${data_path}/${chunk_file}"

# choose eQTL model: NB for negative binomial, poisson, gaussian
model="NB"

# if using permutation method to calibrate gene-level p value, set number of permutation
nperm=1000

# prefix for output file
out="${out_path}/${celltype}_chr${chr}_${chunk_file}_jaxqtl_${model}"

jaxqtl \
 --geno ${geno} \
 --covar ${covar} \
 --pheno ${pheno} \
 --model ${model} \
 --mode ${mode} \
 --genelist ${genelist} \
 --test-method ${test_method} \
 --nperm ${nperm} \
 --addpc ${num_expression_pc} \
 --standardize \
 --out ${out}
