indir=../example/local/NK
geno=${indir}/chr3
covar=${indir}/donor_features.all.6PC.tsv
pheno=${indir}/NK.bed.gz
genelist=${indir}/chr3_ENSG00000268279
out=${indir}/chr3_ENSG00000268279

model=NB
mode=cis
test_method=score
nperm=1

jaxqtl \
 -geno ${geno} \
 -covar ${covar} \
 -pheno ${pheno} \
 -model ${model} \
 -mode ${mode} \
 -genelist ${genelist} \
 -test-method ${test_method} \
 -nperm ${nperm} \
 -out ${out}
