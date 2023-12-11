indir=../example/local/NK
geno=${indir}/chr3
covar=${indir}/donor_features.all.6PC.tsv
pheno=${indir}/NK.bed.gz
genelist=${indir}/chr3_ENSG00000228242_good
out=${indir}/chr3_ENSG00000228242_good_new

model=NB
mode=cis
test_method=score
nperm=1000

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
