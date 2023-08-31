# Cluster environment for text cleaning

## create an environment
* Create a new environment "clean_text" with python 3.10
```
(base) correa@master:~$ conda create -n clean_text python=3.10.11
```
* Confirm it was created
```
conda info --envs
# conda environments:
#
base                  *  /home/correa/miniconda3
clean_text               /home/correa/miniconda3/envs/clean_text
word2vec                 /home/correa/miniconda3/envs/word2vec
```
* From the base environment, install the following packages in the "clean_text" environment
  * pandas
  * demoji
  * nltk
  * spacy
```
conda install -n clean_text pandas
conda install -n clean_text nltk
conda install -n clean_text spacy

conda activate clean_text

pip install demoji
python -m spacy download de
```
* To check the version installed for each library we can activate the environment "clean_text" and run the command to list
* the installed libraries
```
conda activate clean_text
conda list
...
pandas                    2.0.3           py310h1128e8f_0
de-core-news-sm           3.5.0                    pypi_0    pypi
demoji                    1.1.0                    pypi_0    pypi
nltk                      3.8.1           py310h06a4308_0
spacy                     3.5.3           py310h3c18c91_0
```
* In case we need to delete the environment
```
conda remove --name clean_text --all
```
# Text cleaning
## Prepare directory
* Create a directory "text_cleaning"
```
mkdir text_cleaning
```
* Create a directory inside "text_cleaning" called "data"
```
cd ./text_cleaning
mkdir data
```
* Create a directory inside "text_cleaning" called "output"
```
mkdir output
```
* Create the python script "text_cleaner.py"
```
vi text_cleaner.py
```

## Prepare wordnet corpora from nltk, for WordNetLemmatizer
* Nodes are not connected to internet, so it's necessary to download first everything needed
* Using the documentation from nltk https://www.nltk.org/data.html
* Download the wordnet corpora (zip file) from http://www.nltk.org/nltk_data/ 
* Link for downloading directly (https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/wordnet.zip)
* Create the folder "/nltk_data/corpora" in the environment folder "/home/correa/miniconda3/envs/clean_text"
```
cd /home/correa/miniconda3/envs/clean_text

mkdir nltk_data
cd nltk_data
mkdir corpora
```
* Upload the zip from local host to cluster
```
scp [path]\wordnet.zip correa@master.ismll.de:/home/correa/miniconda3/envs/clean_text/nltk_data
```
* Unzip zip file into "nltk_data/tokenizers" folder
```
unzip ./wordnet.zip
```
* Delete zip file
```
rm ./punkt.zip
```
## Run script
* Upload input file, from local host to cluster
```
scp [path]\2021-01-01.csv correa@master.ismll.de:/home/correa/text_cleaning/data
scp [path]\2021-01-02.csv correa@master.ismll.de:/home/correa/text_cleaning/data
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