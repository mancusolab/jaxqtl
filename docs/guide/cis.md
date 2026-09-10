# Cis mapping

Cis mapping tests variants within a window around each molecular phenotype and reports one selected association per
phenotype. The output includes the lead variant, its nominal association, and a gene-level adjusted p-value.

The `--window` value defaults to 500,000 bases. By default, the interval extends from `TSS - window` through
`TES + window`. Add `--tss-centered` to instead use `TSS - window` through `TSS + window`.

<span id="permutation-calibration"></span>
<span id="spa-and-acat"></span>

## Choose a calibration method

Use `--spa --acat` for fast score-test aggregation without permutations; SPA is strongly recommended because
ACAT is sensitive to variant p-value calibration. Omit `--acat` to use Beta permutation, with `--nperm` controlling
the number of shuffles. The methods need not produce the same p-values or discoveries. See
[Tests and gene-level calibration](tests.md) for the statistical tradeoffs.

The [Quickstart](quickstart.md#run-a-cis-scan) contains complete commands for both methods.

## Select regions and phenotypes

Use `--chr LABEL` to restrict the scan to an exact chromosome label shared by the inputs, and `--gene-list PATH`
to select phenotype IDs from a file. [Identifier lists](covariates.md#identifier-lists) contain one ID per line.
Using a gene list with the full phenotype matrix preserves library-size offsets computed before gene selection;
see [Offsets](offsets.md) when the input file itself has already been restricted.

## Execution and fitting

Score and SPA scans reuse one null fit per phenotype; permutation scans fit a null model for each shuffle.
Compiled kernels are reused across cis-window sizes. See [Run large scans](hpc.md#compilation-and-memory)
for compilation and memory behavior.

GLM fitting is controlled by `--tol`, `--gtol`, `--max-iter`, and `--step-size`. See
[Troubleshooting](troubleshooting.md#stopping-rules) for defaults and stopping rules.

## Inspect results

Follow [Post-process cis results](postprocessing.md) to filter failures and apply study-level FDR correction.
The gene-level `pvalue_adj` is not an across-gene FDR value.

!!! note "Cis mode retains some failed tests"

    If every SNP-level p-value for a tested gene is non-finite, jaxQTL writes one row with `result_valid = false` and
    `failure_reason = "no_finite_pvalues"`. Association and lead-variant fields are null because no lead exists.

Genes with no variants in the requested window or no phenotype variance are skipped. See
[Cis output](../reference/outputs.md#cis-output) for the complete result contract.
