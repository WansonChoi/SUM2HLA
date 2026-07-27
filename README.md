# SUM2HLA

## (1) Introduction

**SUM2HLA** performs __Human Leukocyte Antigen (HLA) fine-mapping__ using only GWAS summary statistics of a target disease, eliminating the need for individual-level genotype data.

SUM2HLA enhances analytical resolution by calculating the **Approximated Posterior Probability (APP)** of causality for each candidate marker—a 2-field classical HLA allele or an amino acid position—by leveraging joint-association information, which offers higher resolution than marginal p-values.

Using the GWAS summary statistics of the target disease and a reference correlation matrix, SUM2HLA identifies putative causal HLA loci with the highest APP and performs **Stepwise Conditional Analysis (SWCA)** to detect independent HLA loci.



## (2) Requirements and Installation

### (2-1) Operating System (OS)

SUM2HLA supports Linux and macOS. Windows is supported only via the Windows Subsystem for Linux (WSL).

We have specifically tested SUM2HLA on the following environments:
- Linux: CentOS 7
- macOS: Sequoia (v15.7.2; Intel) and Tahoe (v26.1; M1 Pro)
- Windows: Windows 11 WSL-Ubuntu 22.04.5 LTS and 24.04.3 LTS


### (2-2) Prerequisites: Anaconda or Miniconda

We assume that the latest version of **Anaconda** (or **Miniconda**) is installed on your system to utilize `conda`.


> Tip: We recommend **Miniconda**, the lightweight alternative to Anaconda, as it allows you to use `conda` while keeping the initial installation of unnecessary Python packages to a minimum. (https://www.anaconda.com/docs/getting-started/miniconda/install)



### (2-3) Clone the Repository

Ensure that `git` is installed on your system. Clone this repository and move to the directory using the following commands:

```bash
git clone https://github.com/WansonChoi/SUM2HLA.git
cd SUM2HLA/
```


### (2-4) Create a Conda Environment

Create a virtual environment named "SUM2HLA" and install the necessary dependencies using the command below:

```bash
conda create -y -n SUM2HLA -c conda-forge jax=0.4.14 "jaxlib=0.4.14=cpu*" git-lfs pandas scipy numpy threadpoolctl bioconda::plink bioconda::ucsc-liftover
```

#### For Users with NVIDIA GPUs
If you are using a Linux or WSL with an NVIDIA GPU, you can install the GPU-enabled version of jaxlib to accelerate SUM2HLA. Use the following command instead to create the environment:

```bash
conda create -y -n SUM2HLA -c conda-forge jax "jaxlib=*=cuda*" git-lfs pandas scipy numpy threadpoolctl bioconda::plink bioconda::ucsc-liftover
```

<!-- ```bash
conda create -y -n SUM2HLA -c conda-forge jax=0.4.14 "jaxlib=0.4.14=cuda112py310*" git-lfs pandas scipy numpy threadpoolctl bioconda::plink bioconda::ucsc-liftover 
``` -->

> Note: The "jaxlib=\*=cuda*" pattern ensures that Conda selects a GPU-accelerated build compatible with your specific driver version, whether it is CUDA 11 or 12.

> Note: You only need to create the environment once. For future usage, you can skip this step and proceed directly to activation.


### (2-5) Activate and Fetch Example Data

After creating the environment, you must activate it and retrieve the reference correlation matrix file (382MB). This step is crucial for running the example successfully.


First, activate the SUM2HLA environment:

```bash
conda activate SUM2HLA
```

Next, use the git-lfs tool installed within the environment to initialize and fetch the actual data files:

```bash
git lfs install --local
git lfs pull
```

Why are these commands necessary?

- `git lfs install --local`: We use the --local option to ensure the configuration is applied only to this repository using the version installed in our Conda environment, without modifying or conflicting with your global system settings.

- `git lfs pull`: Even if git clone completed successfully, the large correlation matrix file (example/REF_1kG.EUR.hg19.SNP+HLA.NoNA.PSD.ld.gz) may have been downloaded as a small "pointer file" rather than the actual binary data. This command ensures the real file is downloaded.



> Note: These two git-lfs commands also need to be performed only once during the initial setup.



## (3) Running an Example

With the SUM2HLA environment activated, run SUM2HLA using the provided example data:

```bash
python SUM2HLA.py \
	--sumstats example/WTCCC.RA.GWASsummary.N4798.assoc.logistic \
	--ref example/REF_1kG.EUR.hg19.SNP+HLA \
	--out OUT.WTCCC_RA.REF_1kG.EUR
```

This example uses GWAS summary statistics for Rheumatoid arthritis (RA) and a **1000 Genomes (1kG) Project European** reference dataset, both provided in this repository.

### Note on the reference panel provided in this repository

The reference panel in `example/` is provided **for the example run only**.

All analyses in our paper were performed with the Type 1 Diabetes Genetics Consortium (T1DGC) reference panel. That panel is available only upon request for research purposes, so we cannot redistribute it here. Instead, we provide a reference panel that we built from the publicly available 1000 Genomes Project EUR reference panel distributed with CookHLA, using the same steps we applied to the T1DGC reference panel.

We verified this panel only to the extent of confirming that the example run works correctly. We did not benchmark it as extensively as the T1DGC reference panel, and we therefore recommend against using it for actual analyses. To build a reference panel for your own analyses, follow the procedure in the Wiki: [Constructing the T1DGC Reference Correlation Matrix](https://github.com/WansonChoi/SUM2HLA/wiki/Constructing-the-T1DGC-Reference-Correlation-Matrix).

Expected Runtime: Approximately 3 minutes on a GPU or 10 minutes on a CPU (based on our system specifications).

Once finished, you can deactivate the environment:

```bash
conda deactivate
```



## (4) Output Files

For each run, SUM2HLA generates **one `*.APP` file per candidate set** plus the result of the Stepwise Conditional Analysis (SWCA).

### (4-1) The `*.APP` Files

SUM2HLA reports the approximated posterior probability (APP) of causality **separately for three candidate sets**, so that you can choose the set you want to analyze:

| File | Candidate set |
| :--- | :--- |
| `*.AA+HLA.APP` | 2-field classical HLA alleles **and** amino acid positions (the set used in our paper) |
| `*.HLA.APP` | 2-field classical HLA alleles only |
| `*.AA.APP` | amino acid positions only |

Each file is self-contained: the `APP` values are normalized **within that candidate set**, so they sum to 1.0 over the markers of that set. The `rank`, `rank_p`, and `CredibleSet(99%)` columns are likewise computed within the set. A marker therefore receives a different APP in `*.AA+HLA.APP` than in `*.AA.APP`, because the two sets have different denominators.

> Note: One-field (allele group) markers such as `HLA_A_01` are not used in any candidate set; only 2-field classical HLA alleles such as `HLA_A_0101` are. Amino acid markers are restricted to positions of the mature HLA protein, so signal-peptide positions (negative position numbers) are excluded.

Each file contains one row per marker in the corresponding candidate set and 11 columns, sorted in descending order by `APP`. The example below is from `*.AA+HLA.APP`.

| rank | rank_p | SNP | A1 | A2 | APP | CredibleSet(99%) | LL+Lprior | LL+Lprior_diff | LL+Lprior_diff_acc | logAPP |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | 0.0 | HLA_DRB1_0401 | P | A | 0.9986325167877164 | True | 267.54632568359375 | 0.0 | 0.0 | -0.0013684190707294874 |
| 2 | 0.0006357... | AA_DRB1_120_32657518_S | P | A | 0.0013124103068841014 | False | 260.91180419921875 | 6.634521484375 | 6.634521484375 | -6.6358899034457295 |
| 3 | 0.0012714... | AA_DRB1_120_32657518_N | P | A | 1.990844656179909e-05 | False | 256.72332763671875 | 4.1884765625 | 10.822998046875 | -10.82436646594573 |
| 4 | 0.0019071... | AA_DRB1_11_32660115_V | P | A | 1.990844656179909e-05 | False | 256.72332763671875 | 0.0 | 10.822998046875 | -10.82436646594573 |
| 5 | 0.0025429... | AA_DRB1_96_32657590_Y | P | A | 2.6564321773915476e-06 | False | 254.70916748046875 | 2.01416015625 | 12.837158203125 | -12.83852662219573 |

Column Descriptions:
1. SNP: The marker label of the classical HLA allele or amino acid position.
2. A1: The effect allele, as defined in the reference panel (`.bim` file, 5th column).
3. A2: The non-effect allele, as defined in the reference panel (`.bim` file, 6th column).
4. APP: The causal posterior probability.
5. CredibleSet(99%): Indicates whether the variant is included in the 99% credible set (accumulated top APPs reaching 0.99).
6. rank: The rank of the variant (The variant with the highest APP has a rank of 1).
7. rank_p: The percentile rank among all $h$ markers of the candidate set.
	- Note: The highest APP variant has a rank_p of 0.0 (calculated as $0 / h$). The 2nd highest is $1 / h$.
8. LL+Lprior: The sum of the log-likelihood (calculated using observed association z-scores and the reference correlation matrix via multivariate normal distribution) and the log-prior probability. This value is used to calculate logAPP.
9. logAPP: The natural logarithm of the posterior probability, derived from the LL+Lprior column before conversion to the final APP.
10. LL+Lprior_diff: The difference in LL+Lprior values between two adjacent variants in the sorted list.
11. LL+Lprior_diff_acc: The difference in LL+Lprior values between the top-ranked variant (rank 1) and the current variant (rank N).



### (4-2) The `*.r2pred0.6.ma.SWCA.dict` File

This file contains the results of the Stepwise Conditional Analysis (SWCA) in JSON format. The SWCA is performed on the `AA+HLA` candidate set.

```bash
{
    "ROUND_1": [
        "HLA_DRB1_0401"
    ],
    "ROUND_2": {
        "AA_DRB1_11_32660115_SGP": {
            "AA_DRB1_96_32657590_HQ": {
                "r": 0.946231,
                "r2": 0.895353
            },
            "AA_DRB1_96_32657590_YE": {
                "r": -0.948716,
                "r2": 0.900063
            },
```

- ROUND_1: Represents the top candidate identified with the highest APP.
- ROUND_2 (and subsequent rounds): Represents the results of SWCA.
	- The key (e.g., "AA_DRB1_11_32660115_SGP") represents the **independent HLA locus** identified in that round.
	- The dictionary nested under this key lists the variants that are **clumped** with this independent variant.
	- The innermost dictionary provides the correlation values ($r$ and $r^2$) between the clumped marker and the identified independent HLA locus.

### (4-3) Other Output Files

For details on additional output files, please refer to the Wiki section.



## (5) How to create a reference dataset for SUM2HLA?

Detailed instructions are available in the Wiki section (https://github.com/WansonChoi/SUM2HLA/wiki/Constructing-the-T1DGC-Reference-Correlation-Matrix).



## (6) Summary Statistics from the Paper

We provide the output summary statistics corresponding to the main analyses presented in our manuscript. These files are located in the [`results/`](results/) directory.

### Directory Structure


| Directory | Dataset / Analysis | Description | Related Figures/Tables |
| :--- | :--- | :--- | :--- |
| **`MVP`** | Million Veteran Program | HLA fine-mapping results for 131 traits. | Figure 1, Table 2 |
| **`Consortium`** | Consortium-scale Summaries | HLA fine-mapping results for 9 autoimmune diseases. | Table 2 |
| **`WTCCC`** | Wellcome Trust Case Control Consortium | HLA fine-mapping results and genotype-based gold-standard summary statistics for RA and T1D. | Figure 3 |
| **`UKB_FG`** | UK Biobank & FinnGen | HLA fine-mapping results for 33 traits used in community detection (Figure 4), which include the 12 traits used for genotype-based validation (Table 1). | Table 1, Figure 4 |

### File Naming Convention

Files in these directories follow the naming patterns below:
- `{Dataset}.{Trait}.AA+HLA.APP`
- `{Dataset}.{Trait}.Z_imputed`


## (7) Citation

Under review.
