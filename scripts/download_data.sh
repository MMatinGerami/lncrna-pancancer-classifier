#!/usr/bin/env bash
# Download public TCGA pan-cancer data (UCSC Xena) and the GENCODE v23 annotation.
set -euo pipefail
cd "$(dirname "$0")/../data/raw"
fetch() { [ -s "$2" ] || curl -fL --retry 3 -o "$2" "$1"; }
XT=https://toil-xena-hub.s3.us-east-1.amazonaws.com/download
XP=https://tcga-pancan-atlas-hub.s3.us-east-1.amazonaws.com/download
fetch "$XT/tcga_RSEM_gene_tpm.gz"                                 tcga_RSEM_gene_tpm.gz
fetch "$XT/probeMap%2Fgencode.v23.annotation.gene.probemap"      gencode.v23.annotation.gene.probemap
fetch "$XP/TCGA_phenotype_denseDataOnlyDownload.tsv.gz"           TCGA_phenotype_denseDataOnlyDownload.tsv.gz
fetch "$XP/Survival_SupplementalTable_S1_20171025_xena_sp"        Survival_SupplementalTable_S1_20171025_xena_sp
fetch "https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_23/gencode.v23.annotation.gtf.gz" gencode.v23.annotation.gtf.gz
# MET500 metastatic cohort (Robinson et al. 2017), UCSC Xena public hub: external validation
XM=https://ucscpublic.xenahubs.net/download/MET500/geneExpression
fetch "$XM/M.mx.log2.txt.gz"                                    MET500_M.mx.log2.txt.gz
fetch "$XM/M.meta.plus.txt"                                     MET500_M.meta.plus.txt
shasum -a 256 * > SHA256SUMS
