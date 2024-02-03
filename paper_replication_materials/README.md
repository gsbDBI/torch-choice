# Torch-Choice Paper Replication Materials
Updated: 2024-02-01 by Tianyu Du

All replication materials of the paper are located in the `paper_replication_materials` directory, including a copy of this README file.

Readers are expected to have `torch-choice` and dependencies properly installed before running the replication materials.
You can either install `torch-choice` from the source code or from the PyPI repository.
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
The `get_torch_choice_paper_data.sh` would download the following datasets to the `torch_choice_paper_data` directory, we have included the MD5 checksums for each file to ensure the integrity of the datasets.

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

## To Run Python Benchmarks
`run_torch_choice.py` is the script for running six benchmarks using `torch-choice`. Specifically, the following commands will run the benchmarks for the number of records N, the number of model parameters P, and the number of items in the choice set I (corresponding to Figure 4 to 9 in the paper).

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

Commands above will generate CSV files named as `Python_<benchmark name>.csv` (e.g., )

You would need to change the `DATA_PATH` variable in the script to the location where you put the benchmark datasets.
Please use the command in `run_torch_choice.sh` to run the benchmarks.

## Run R Benchmarks**
The `run_logit_num_{items, params, records}.R` are R scripts to run the benchmarks in `R`.
```bash
Rscript run_mlogit_num_items.R
Rscript run_mlogit_num_records.R
Rscript run_mlogit_num_params.R
```

## Reproduce Figures in Chapter 5 of the Paper
Please run the `visualize_performance_benchmarks.ipynb` to generate figures in the paper. You can set the `REPORT_RATIO` variable in the notebook to change the Y-axis. The Y-axis can either report the relative time (i.e., the time of the algorithm divided by the time of the baseline case) or the absolute time (i.e., the time of the algorithm).
The notebook will generate figures in the `figures` directory.
