# Evaluating text embedding models in tabular data retrieval - SIGIR 2024

This code facilitates an examination of the durability and effectiveness of premier text embedding models when applied to datasets in tabular form. The analysis is segmented into two distinct experiments. The first experiment focuses on evaluating the models' resilience in the face of data alterations, which include reorganizing the columns and diminishing the data quantity. The second experiment is dedicated to assessing their capability in retrieving information accurately from a substantial tabular dataset.

## Getting started
The retrieval and integration pipeline is divided in two processes. Each one has a specific folder:

1. [Assess models robustness](https://github.com/aberenguerpas/SIGIR24/tree/main/exp1)
2. [Evaluate data retrieval performance](https://github.com/aberenguerpas/SIGIR24/tree/main/exp2)

### Prerequisites
Create a python virtual env and install the requirements by running the following command:
``
pip install -r requirements.txt
``

Also, download the necesary datasets to perform the experiment by executing the following script:

``
cd data
./download_unzip_data.sh
``

If there is any problem with the former script you can download the data directly:
[Wikitables dataest](https://drive.google.com/file/d/1SD7NXIdWK97aFTKdePuw8qusMeGITrlu/view?usp=sharing)
[Dublin dataset](https://drive.google.com/file/d/1vT5M8kdcGVdqB3Lt0ltvhDJtCE3Z-nnI/view?usp=sharing)
and then extract it in the `data` folder

*Note: _Sensors_ dataset it is not include due to privacy aspects*

## 1. Assess models robustness


## 2. Evaluate data retrieval performance
The first step involves constructing a comprehensive embedding index. This is based on the individual table embeddings computed in a prior experiment. To accomplish this, run the following command:

```
python rebuild_index.py -i ../exp1/embeddings/wikitables/ -m bge-base
```

Here, the `-i` flag specifies the directory containing all table embeddings, while -m denotes the model employed.

Executing the above command will create a general index. This index consolidates all embeddings from a specific dataset and model.

The second step is to carry out a search. You can initiate this by using the command below:
```
python search -m bge-base
```

In this case, -m represents the model. Performing this step will result in the creation of a directory containing the search outcomes.

The final step is to evaluate these search results. To do this, simply execute:

```
python evaluate.py -i ./search_results/bge-base_wikitables.csv
```
Here, `-i` indicates the file path where the results are stored. This process will yield a folder named 'evaluate_results', containing the final results for each metric.