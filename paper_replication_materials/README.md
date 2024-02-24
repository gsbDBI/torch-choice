# Torch-Choice Paper Replication Materials
Updated: 2024-02-01 by Tianyu Du

All replication materials of the paper are located in the `paper_replication_materials` directory, including a copy of this README file.

Readers are expected to have `torch-choice` and dependencies properly installed before running the replication materials.

Please use the `paper` branch after cloning the repository (frozen for paper replications) of `torch-choice` for these replication exercises.

``` bash
# clone repository
git clone git@github.com:gsbDBI/torch-choice.git
git checkout paper
```

You can install `torch-choice` by running the following command:

```bash
# NOTE: you would need to navigate to the directory where the setup.py file is located.
python setup.py develop
```



Please refer to the installation guideline page for more details.

The replication materials consist of two components:
1. The `paper_demo.ipynb` notebook covers the basic usage of `torch-choice` and model demonstration covered in the paper.
2. Scripts and notebooks for running benchmarks and reproducing figures in the "Performance" section of our paper. These scripts generate synthetic datasets of different scales, estimating PyTorch and R models on these datasets, and visualize the performance of different implementations (i.e., reproduce figures in our paper).


## Datasets
We have the replication data ready from (1) the Harvard Dataverse and (2) the author's homepage at Stanford.edu. The replication material is named as `./torch_choice_paper_data.tar.gz` with MD5 checksum `da0ab6c1c1d8ff8a03c6b9bc001b4370`.

**Option 1** Download from Stanford.edu using the following command:
```bash
wget https://stanford.edu/~tianyudu/torch_choice_paper_data.tar.gz
```

**Option 2** Alternatively, download the `torch_choice_paper_data.tar.gz` file from Harvard Dataverse directly at `https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/UZ2Q13`.

After downloading the `torch_choice_paper_data.tar.gz` file, run a MD5 check sum of the file to ensure integrity. The following command should return `da0ab6c1c1d8ff8a03c6b9bc001b4370`:
```bash
# call md5sum on linux machines.
md5sum torch_choice_paper_data.tar.gz
# call md5 if you are using macOS.
md5 torch_choice_paper_data.tar.gz
```

Finally, unzip the `torch_choice_paper_data.tar.gz` file downloaded, it would create a folder called `torch_choice_paper_data` with all the datasets. You can use the following command to unzip it on Linux or MacOS.
```bash
tar -xvf torch_choice_paper_data.tar.gz
```
Alternatively, you can use other GUI tools to unzip the file (e.g., 7-zip).

### List of Datasets Expected
The `get_torch_choice_paper_data.sh` downloads the replication data from the Harvard Dataverse and unzips it. The replication data includes the following datasets:

```
9328b2aa1413fac66f495c8d3a6f3148  ./torch_choice_paper_data/car_choice.csv
c1d73db9b721a5f78d6dc7bcab7109fd  ./torch_choice_paper_data/R_performance_num_items.csv
ca7aedc245a3336975d461e6cfa98487  ./torch_choice_paper_data/R_performance_num_params.csv
13e2ddddb3407c372a6f1eec73aaab82  ./torch_choice_paper_data/R_performance_num_records.csv
0dc7a0408d4bc7e220cec6a334a6fa73  ./torch_choice_paper_data/simulated_choice_data_100_items.pt
03700cf9459d7b8fe8b234a6669efb98  ./torch_choice_paper_data/simulated_choice_data_10_items.pt
ba67f211e8b59e07af7f0b2d6a792f17  ./torch_choice_paper_data/simulated_choice_data_10k_records.csv
18d7ea7bf48f2fa70af4ab8cf6c377de  ./torch_choice_paper_data/simulated_choice_data_150_items.pt
e3bd6b783beecb74c00141d99cdf4a6c  ./torch_choice_paper_data/simulated_choice_data_200_items.pt
36a96daee00e100557cb157076fd6b2d  ./torch_choice_paper_data/simulated_choice_data_20_items.pt
c200267746f872afa33411fc199ac818  ./torch_choice_paper_data/simulated_choice_data_250_items.pt
a21323a973c8cdc99b53a4914dcbb4db  ./torch_choice_paper_data/simulated_choice_data_300_items.pt
58d531f21d91744544e0e7a204b5fcf7  ./torch_choice_paper_data/simulated_choice_data_30_items.pt
13b8aedf2a6b923392157d3d5533afd2  ./torch_choice_paper_data/simulated_choice_data_350_items.pt
1bb34ad31d31fa199c7c4623e0be5739  ./torch_choice_paper_data/simulated_choice_data_400_items.pt
025fa94f10c0a474f127d30e275a4ff8  ./torch_choice_paper_data/simulated_choice_data_450_items.pt
f8332ea05135c5eb515038e8b4e9c02a  ./torch_choice_paper_data/simulated_choice_data_500_items.pt
9fa6531bf4a05e9a3a16b284ede2be78  ./torch_choice_paper_data/simulated_choice_data_50_items.pt
2b3f2fca7bbd158d70c372c8f35f53d8  ./torch_choice_paper_data/simulated_choice_data_dev.csv
f35ebae25ed169087f2868be8ac3faa7  ./torch_choice_paper_data/simulated_choice_data_dev.pt
80ff69248a7e9736561ee7ca46731167  ./torch_choice_paper_data/simulated_choice_data_full.pt
602ad92790eb380ed0cd87011f913aa6  ./torch_choice_paper_data/simulated_choice_data_num_items_experiment_100.csv
4cebce800d0559633f0b708604afc282  ./torch_choice_paper_data/simulated_choice_data_num_items_experiment_100.pt
6099ca9a40407abb06a44ad1ae64f8ca  ./torch_choice_paper_data/simulated_choice_data_num_items_experiment_10.csv
60b93d09597763e18f4152f8a3790161  ./torch_choice_paper_data/simulated_choice_data_num_items_experiment_10.pt
e5ee17460716f3bfca75be7b74cb45b6  ./torch_choice_paper_data/simulated_choice_data_num_items_experiment_150.csv
b5acff58a0aea4a542b635143be68d5d  ./torch_choice_paper_data/simulated_choice_data_num_items_experiment_150.pt
1f0cec0ec619045286e77166b9fe5964  ./torch_choice_paper_data/simulated_choice_data_num_items_experiment_200.csv
0fa19c6e56deef96ee49f3cddb74c719  ./torch_choice_paper_data/simulated_choice_data_num_items_experiment_200.pt
db450469c2fc33e448057be8c0c91145  ./torch_choice_paper_data/simulated_choice_data_num_items_experiment_250.csv
d8bf20422ca9c186167d55ec7230f8b2  ./torch_choice_paper_data/simulated_choice_data_num_items_experiment_250.pt
13adf8c2b8f81b83b742ef92eeef2913  ./torch_choice_paper_data/simulated_choice_data_num_items_experiment_30.csv
52d2e51f67b655a0738ee41741d1f3dd  ./torch_choice_paper_data/simulated_choice_data_num_items_experiment_30.pt
3781cb28788ae2e62593145251ae6fd5  ./torch_choice_paper_data/simulated_choice_data_num_items_experiment_50.csv
178d63fe6d890e6677b03a924c032ba5  ./torch_choice_paper_data/simulated_choice_data_num_items_experiment_50.pt
6c213c991a06658dde938c1ba06d13a0  ./torch_choice_paper_data/simulated_choice_data_num_items_experiment_5.csv
daaadbf257f98af342fe4aea98497af2  ./torch_choice_paper_data/simulated_choice_data_num_items_experiment_5.pt
1974836e7099a7a97145080e87e6f465  ./torch_choice_paper_data/simulated_choice_data_num_params_experiment_small.csv
86150c13c148acb9b34288de26844d4c  ./torch_choice_paper_data/simulated_choice_data_num_params_experiment_small.pt
1030eb9e4239224a0c18a40c50f55d02  ./torch_choice_paper_data/simulated_choice_data_num_records_experiment.csv
174640f86e905d6f05e83558170e966b  ./torch_choice_paper_data/simulated_choice_data_num_records_experiment.pt
```

## Run Python Benchmarks
`run_torch_choice.py` is the script for running six benchmarks using `torch-choice`. Specifically, the following commands will run the benchmarks for the number of records N, the number of model parameters P, and the number of items in the choice set I (corresponding to Figure 4 to 9 in the paper). The following command will generate the CSV files recording package performances in the `results` directory. Note that the `results` directory is prefilled with the results from our experiments, so that you can skip this step and directly generate the figures (see [this section](#generate-figures)).

```bash

These experiments are designed to stress test the performance of `torch-choice` under different scales of the dataset and model. They could take several hours to run, depending on the computing environment. You can modify the `NUM_SEEDS` variable in the script to reduce the number of random seeds (we are using 5) to speed up the process.

We were running these benchmarks on a Linux machine with 16 CPU cores Intel Xeon processor and 128GiB of memory, with a single RTX3090 (24G) GPU.

```bash
# make a directory to store the results, we will use this directory to store the results of the benchmarks, there should already be a directory called `results` in the replication material. If not, please create one.
# mkdir results

# stress test for the number of records N (Figure 4 and 5).
# the bottom panel.
python ./run_torch_choice.py num_records_experiment_large
# the top right panel.
python ./run_torch_choice.py num_records_experiment_small

# stress test for the number of model parameters P (Figure 6 and 7).
# the bottom panel.
python ./run_torch_choice.py num_params_experiment_small
# the top right panel.
python ./run_torch_choice.py num_params_experiment_large

# stress test for the number of items in the choice set I (Figure 8 and 9).
# the bottom panel.
python ./run_torch_choice.py num_items_experiment_small
# the top right panel.
python ./run_torch_choice.py num_items_experiment_large
```

For your convenience, we have also included the CSV files from OUR experiments in the `our_results.tar.gz` zipped file together with the replication material; you can fill the `results` directory with our results and proceed to the next step (See section on reproducing figures below) to visualize the performance of different implementations while you are benchmarking the package on your own machine.

## Run R Benchmarks
The `run_logit_num_{items, params, records}.R` are R scripts to run the benchmarks in `R`.
Certain experiments in the stress test will demand extensive computational resources, and we recommend running these benchmarks on a high-performance computing cluster or a machine with sufficient computational resources. We tested these benchmarks on a Linux machine with 16 CPU cores Intel Xeon processor and 112GiB of memory. No GPU is required for these benchmarks in R.

You need to install the following R packages before running the benchmarks:

```R
# install mlogit
install.packages("mlogit")
# install tictoc
install.packages("tictoc")
```

```

You can reduce the `num.seeds` variable in the R scripts to speed up the process by using fewer random seeds (we are using 5 random seeds).

```bash
Rscript run_mlogit_num_items.R
Rscript run_mlogit_num_records.R
Rscript run_mlogit_num_params.R
```

After running the nine benchmarks above, you should get the following CSV files in the `results` directory:
```
Python_num_items_experiment_large_Adam.csv
Python_num_items_experiment_small_Adam.csv
Python_num_params_experiment_large_Adam.csv
Python_num_params_experiment_small_Adam.csv
Python_num_records_experiment_large_Adam.csv
Python_num_records_experiment_small_Adam.csv
R_performance_num_items.csv
R_performance_num_params.csv
R_performance_num_records.csv
```
These CSV files records the runtime of the benchmarks for the number of records N, the number of model parameters P, and the number of items in the choice set I, depending on the experiment.


# Reproduce Figures in Chapter 5 of the Paper {#generate-figures}
Please run the `visualize_performance_benchmarks.ipynb` to generate figures using the CSV files generated from the benchmarks. The notebook will generate figures in the `figures` directory.

You can set the `REPORT_RATIO` variable in the notebook to change the Y-axis. The Y-axis can either report the relative time (i.e., the time of the algorithm divided by the time of the baseline case) or the absolute time (i.e., the time of the algorithm).
The notebook will generate figures in the `figures` directory.

For your convenience, we have included the performance data from OUR experiments in the `our_results.tar.gz` zipped file with the replication material. You can fill the `results` directory with our results and try to visualize the performance of different implementations while you are running your own benchmarks. Results in `our_results.tar.gz` should generate the same figures as in the paper.

Note: the actual font size and figure size in the paper are adjusted for the publication format. The figures generated by the notebook may not be exactly the same as the figures in the paper in terms of format; but the content should be the same.
