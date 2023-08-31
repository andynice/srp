# Cluster environment preparation
## connect to cluster
using the credentials provided by email
ssh correa@master.ismll.de
/***

## install miniconda
* Check OS in cluster
```
cat /etc/os-release
NAME="Ubuntu"
```
* Guide used: https://waylonwalker.com/install-miniconda/, "Installing miniconda on Linux"
* At the end it was necessary to use only
```
~/miniconda3/bin/conda init bash
```
* As a reference it was used too https://docs.conda.io/projects/conda/en/latest/user-guide/install/linux.html
* After running the init script, this is how ./-bashrc looks like
```
cat ./-bashrc
...
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/home/correa/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/home/correa/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/home/correa/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/home/correa/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<
```
* Disconnect and connect to cluster
* conda command is available now
```
(base) correa@master:~$ conda --help
...
usage: conda [-h] [-V] command ...

conda is a tool for managing and deploying applications, environments and packages.
...
```
```
conda --version
conda 23.3.1
conda info --envs
# conda environments:
#
base                  *  /home/correa/miniconda3
```
* Update of conda, when trying to create a new environment, it recommended to update conda
```
conda update -n base -c defaults conda
...
conda --version
conda 23.5.0
```
## create an environment
* Create a new environment "word2vec" with python 3.10
```
(base) correa@master:~$ conda create -n word2vec python=3.10.11
```
* Confirm it was created
```
conda info --envs
# conda environments:
#
base                  *  /home/correa/miniconda3
word2vec                 /home/correa/miniconda3/envs/word2vec
```
* From the base environment, install the following packages in the "word2vec" environment
  * nltk
  * numpy
  * pandas
  * gensim
```
conda install -n word2vec nltk
conda install -n word2vec numpy
conda install -n word2vec pandas
conda install -n word2vec gensim
```
* To check the version installed for each library we can activate the environment "word2vec" and run the command to list
* the installed libraries
```
conda activate word2vec
conda list
...
nltk                      3.8.1           py310haa95532_0
numpy                     1.25.0          py310h055cbcc_0
pandas                    1.5.3           py310h4ed8f06_0
gensim                    4.3.0           py310h4ed8f06_0
```
* In case we need to delete the environment
```
conda remove --name word2vec --all
```
# Word2Vec training
## Prepare directory
* Create a directory "word2vec_training"
```
mkdir word2vec_training
```
* Create the python script "word_embedding.py"
```
vi word_embedding.py
```
## Prepare punkt model from nltk, for tokenization
* Nodes are not connected to internet, so it's necessary to download first everything needed
* Using the documentation from nltk https://www.nltk.org/data.html
* Download the punkt model (zip file) from http://www.nltk.org/nltk_data/
* Create the folder "/nltk_data/tokenizers" in the environment folder "/home/correa/miniconda3/envs/word2vec"
```
cd /home/correa/miniconda3/envs/word2vec

mkdir nltk_data
cd nltk_data
mkdir tokenizers
```
* Upload the zip from local host to cluster
```
scp [path]\punkt.zip correa@master.ismll.de:/home/correa/miniconda3/envs/word2vec/nltk_data/tokenizers
```
* Unzip zip file into "nltk_data/tokenizers" folder
```
unzip ./punkt.zip
```
* Delete zip file
```
rm ./punkt.zip
```
## Run script
* Upload input file, from local host to cluster
```
scp [path]\biorxiv_medrxiv.pickle correa@master.ismll.de:/home/correa/word2vec_training
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