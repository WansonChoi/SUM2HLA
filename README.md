# SUM2HLA

## (1) Introduction

**SUM2HLA** performs __Human Leukocyte Antigen (HLA) fine-mapping__ using only GWAS summary statistics of a target disease, eliminating the need for individual-level genotype data.

SUM2HLA enhances analytical resolution by calculating the **posterior probability (PP)** of causality for each HLA variant—including both HLA alleles and amino acid residues—by leveraging joint-association information, which offers higher resolution than marginal p-values.

Using the GWAS summary statistics of the target disease and a reference LD matrix, SUM2HLA identifies putative causal HLA loci with the highest PP and performs **Stepwise Conditional Analysis (SWCA)** to detect independent HLA loci.



## (2) Requirements and Installation

### (2-1) Operating System (OS)

SUM2HLA supports Linux and macOS. Windows is supported only via the Windows Subsystem for Linux (WSL).

We have specifically tested SUM2HLA on the following environments:
- Linux: CentOS 7
- macOS: Sequoia (v15.7.2)
- Windows: Windows 11 WSL-Ubuntu 22.04.5 LTS


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

**For Users with NVIDIA GPUs (Linux only)** If you are using a Linux system with an NVIDIA GPU, you can install the GPU-enabled version of jaxlib to accelerate SUM2HLA. Use the following command instead to create the environment:

```bash
conda create -y -n SUM2HLA -c conda-forge jax=0.4.14 "jaxlib=0.4.14=cuda112py310*" git-lfs pandas scipy numpy threadpoolctl bioconda::plink bioconda::ucsc-liftover 
```

> Note: You only need to create the environment once. For future usage, you can skip this step and proceed directly to activation.


### (2-5) Activate and Fetch Example Data

After creating the environment, you must activate it and retrieve the reference LD matrix file (382MB). This step is crucial for running the example successfully.


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

- `git lfs pull`: Even if git clone completed successfully, the large LD matrix file (example/REF_1kG.EUR.hg19.SNP+HLA.NoNA.PSD.ld.gz) may have been downloaded as a small "pointer file" rather than the actual binary data. This command ensures the real file is downloaded.

- `git lfs install --local`: We use the --local option to ensure the configuration is applied only to this repository using the version installed in our Conda environment, without modifying or conflicting with your global system settings.


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

> Note: While the original paper utilizes the Type 1 Diabetes Genetic Consortium (T1DGC) reference dataset, this example uses the 1kG dataset because the T1DGC data is not publicly open.

Expected Runtime: Approximately 3 minutes on a GPU or 10 minutes on a CPU (based on our system specifications).

Once finished, you can deactivate the environment:

```bash
conda deactivate
```


## (4) Output Files

SUM2HLA generates two main output files:

### (4-1) The `*.AA+HLA.PP` File

This file provides the causal Posterior Probabilities (PP) for each HLA variant in the target disease. It contains $h$ rows (corresponding to the number of HLA variants in the reference dataset; e.g., 1,573 for the 1kG EUR reference) and 9 columns. The file is sorted in descending order by PP.

| rank | rank_p | SNP | PP | CredibleSet(99%) | LL+Lprior | LL+Lprior_diff | LL+Lprior_diff_acc | logPP |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | 0.0 | HLA_DRB1_0401 | 0.9986325167877164 | True | 267.54632568359375 | 0.0 | 0.0 | -0.0013684190707294874 |
| 2 | 0.0006357... | AA_DRB1_120_32657518_S | 0.0013124103068841014 | False | 260.91180419921875 | 6.634521484375 | 6.634521484375 | -6.6358899034457295 |
| 3 | 0.0012714... | AA_DRB1_120_32657518_N | 1.990844656179909e-05 | False | 256.72332763671875 | 4.1884765625 | 10.822998046875 | -10.82436646594573 |
| 4 | 0.0019071... | AA_DRB1_11_32660115_V | 1.990844656179909e-05 | False | 256.72332763671875 | 0.0 | 10.822998046875 | -10.82436646594573 |
| 5 | 0.0025429... | AA_DRB1_96_32657590_Y | 2.6564321773915476e-06 | False | 254.70916748046875 | 2.01416015625 | 12.837158203125 | -12.83852662219573 |

Column Descriptions:
1. SNP: The marker label for each HLA variant.
2. PP: The causal posterior probability.
3. CredibleSet(99%): Indicates whether the variant is included in the 99% credible set (accumulated top PPs reaching 0.99).
4. rank: The rank of the variant (The variant with the highest PP has a rank of 1).
5. rank_p: The percentile rank among all $h$ HLA variants provided by the reference.
	- Note: The highest PP variant has a rank_p of 0.0 (calculated as $0 / h$). The 2nd highest is $1 / h$.
6. LL+Lprior: The sum of the log-likelihood (calculated using observed association z-scores and the reference LD matrix via multivariate normal distribution) and the log-prior probability. This value is used to calculate logPP.
7. logPP: The natural logarithm of the posterior probability, derived from the LL+Lprior column before conversion to the final PP.
8. LL+Lprior_diff: The difference in LL+Lprior values between two adjacent variants in the sorted list.
9. LL+Lprior_diff_acc: The difference in LL+Lprior values between the top-ranked variant (rank 1) and the current variant (rank N).



### (4-2) The `*.r2pred0.6.ma.SWCA.dict` File

This file contains the results of the Stepwise Conditional Analysis (SWCA) in JSON format.

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

- ROUND_1: Represents the top HLA variant identified with the highest PP.
- ROUND_2 (and subsequent rounds): Represents the results of SWCA.
	- The key (e.g., "AA_DRB1_11_32660115_SGP") represents the **independent HLA locus** identified in that round.
	- The dictionary nested under this key lists the variants that are **clumped** with this independent variant.
	- The innermost dictionary provides the LD values ($r$ and $r^2$) between the clumped variant and the identified independent HLA variant.

### (4-3) Other Output Files

For details on additional output files, please refer to the Wiki section.


## (5) How to create a reference dataset for SUM2HLA?

Detailed instructions are available in the Wiki section.

## (6) Citation

Under review.
