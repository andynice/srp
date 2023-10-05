# Cluster environment for text cleaning

## create an environment
* Create a new environment "clean_text" with python 3.10
```
(base) correa@master:~$ conda create -n model_bert_training python=3.10.11
```
* Confirm it was created
```
conda info --envs
# conda environments:
#
base                  *  /home/correa/miniconda3
clean_text               /home/correa/miniconda3/envs/clean_text
word2vec                 /home/correa/miniconda3/envs/word2vec
word2vec                 /home/correa/miniconda3/envs/model_bert_training
```
* From the base environment, install the following packages in the "model_bert_training" environment
  * huggingface transformers
  * pytorch
  * nltk
  * spacy
```
conda install -n model_bert_training -c huggingface transformers
conda install -n model_bert_training -c huggingface -c conda-forge datasets
conda install -n model_bert_training pytorch torchvision -c pytorch
conda install -n model_bert_training numpy
conda install -n model_bert_training scikit-learn
conda install -n model_bert_training -c conda-forge accelerate
conda install -n model_bert_training pandas
```
* To check the version installed for each library we can activate the environment "model_bert_training" and run the command * to list the installed libraries
```
conda activate model_bert_training
conda list
...
huggingface_hub           0.17.2                     py_0    huggingface
datasets                  2.14.5                     py_0    huggingface
transformers              4.32.1          py310haa95532_0
pytorch                   2.0.1              py3.10_cpu_0    pytorch
pytorch-mutex             1.0                         cpu    pytorch
torchvision               0.15.2                py310_cpu    pytorch
numpy                     1.25.2          py310h055cbcc_0
scikit-learn              1.3.0           py310h4ed8f06_0
accelerate                0.23.0             pyhd8ed1ab_0    conda-forge
```
* In case we need to delete the environment
```
conda remove --name model_bert_training --all
```
# Model BERT Training
## Prepare directory
* Create a directory "model_bert_training"
```
mkdir model_bert_training
```
* Create a directory inside "model_bert_training" called "data"
```
cd ./model_bert_training
mkdir data
```
* Create a directory inside "model_bert_training" called "output"
```
mkdir output
```
* Create the python script "model_covid_twitter_bert_training.py"
```
vi model_covid_twitter_bert_training.py
```

## Prepare huggingface model offline
* Nodes are not connected to internet, so it's necessary to download first everything needed
* Otherwise, you will get an error message like this
```
OSError: We couldn't connect to 'https://huggingface.co' to load this file, couldn't find it in the cached files and it looks like digitalepidemiologylab/covid-twitter-bert is not the path to a directory containing a file named config.json.
Checkout your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offline-mode'.
```

* Using the documentation from huggingface https://huggingface.co/docs/transformers/installation#offline-mode
* Run the file "download_bert_model.py" in the local environment
* Prepare a zip file of the jsut created folder "covid-twitter-bert"
* Upload the zip file with the model to the cluster
```
scp [path]\covid-twitter-bert.zip correa@master.ismll.de:/home/correa/model_bert_training
```
* Unzip zip file into "/home/correa/model_bert_training" folder
```
unzip ./covid-twitter-bert.zip
```
* Delete zip file
```
rm ./covid-twitter-bert.zip
```
## Run script
* Upload input file, from local host to cluster
```
scp [path]\g_cases_2021.csv correa@master.ismll.de:/home/correa/model_bert_training/data
scp [path]\en_2021-01-01_output.csv correa@master.ismll.de:/home/correa/model_bert_training/data
```
* Create the bash script "test.sh"
```
vi test.sh
```
* Run the bash script "test.sh"
```
sbatch test.sh
Submitted batch job 449092
```
## Results
* Check progress of the job
```
squeue -u correa
JOBID PARTITION     NAME     USER ST       TIME  NODES NODELIST(REASON)
449092      STUD     test   correa  R       1:04      1 stud-000
```
* Once the job is finished the folder will include "word2vec.model" and err and log files
```
(base) correa@master:~/word2vec_training$ ls -ltr
total 61536
-rw-r--r-- 1 correa students 30880165 Jul 11 16:40 biorxiv_medrxiv.pickle
-rw-r--r-- 1 correa students     4240 Jul 11 16:56 word_embedding.py
-rw-r--r-- 1 correa students      344 Jul 11 17:19 test.sh
-rw-r--r-- 1 correa students      368 Jul 14 12:51 test449092.err
-rw-r--r-- 1 correa students 32102994 Jul 14 12:52 word2vec.model
-rw-r--r-- 1 correa students     5286 Jul 14 12:52 test449092.log
```
* The log file will include the result of the generated vector with 300 dimensions
```
...
  8.11976613e-04 -2.09953310e-03  2.98258266e-03 -1.04414101e-03]
(300,)
```



## References
* https://huggingface.co/learn/nlp-course/chapter3/3?fw=pt
* https://lajavaness.medium.com/regression-with-text-input-using-bert-and-transformers-71c155034b13