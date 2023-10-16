# Cluster environment for longformer model training

## create an environment
* Create a new environment "model_longformer_training" with python 3.10
```
(base) correa@master:~$ conda create -n model_longformer_training python=3.10.11
```
* Confirm it was created
```
conda info --envs
# conda environments:
#
base                  *   /home/correa/miniconda3
clean_text                /home/correa/miniconda3/envs/clean_text
word2vec                  /home/correa/miniconda3/envs/word2vec
model_longformer_training /home/correa/miniconda3/envs/model_longformer_training
```
* From the base environment, install the following packages in the "model_longformer_training" environment
  * huggingface transformers
  * pytorch
  * cuda
  * numpy
  * pandas
  * scikit-learn
```
conda install -n model_longformer_training -c huggingface transformers
conda install -n model_longformer_training -c huggingface -c conda-forge datasets
conda install -n model_longformer_training pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install -n model_longformer_training numpy
conda install -n model_longformer_training scikit-learn
conda install -n model_longformer_training pandas
conda install -n model_longformer_training -c conda-forge accelerate
```
* To check the version installed for each library we can activate the environment "model_longformer_training" and run the command * to list the installed libraries
```
conda activate model_longformer_training
conda list
...
huggingface_hub           0.17.2                     py_0    huggingface
datasets                  2.14.5                     py_0    huggingface
transformers              4.32.1          py310haa95532_0
pytorch                   2.0.1              py3.10_cpu_0    pytorch
pytorch-mutex             1.0                         cpu    pytorch
torchvision               0.15.2                py310_cpu    pytorch
pytorch-cuda              11.7                 h16d0643_5    pytorch
numpy                     1.25.2          py310h055cbcc_0
scikit-learn              1.3.0           py310h4ed8f06_0
accelerate                0.23.0             pyhd8ed1ab_0    conda-forge
```
* In case we need to delete the environment
```
conda remove --name model_longformer_training --all
```
# Model LONGFORMER Training
## Prepare directory
* Create a directory "model_longformer_training"
```
mkdir model_longformer_training
```
* Create a directory inside "model_longformer_training" called "data"
```
cd ./model_longformer_training
mkdir data
```
* Create the python script "model_longformer_training.py"
```
vi model_longformer_training.py
```

## Prepare huggingface model offline
* Nodes are not connected to internet, so it's necessary to download first everything needed
* Otherwise, you will get an error message like this
```
OSError: We couldn't connect to 'https://huggingface.co' to load this file, couldn't find it in the cached files and it looks like digitalepidemiologylab/covid-twitter-bert is not the path to a directory containing a file named config.json.
Checkout your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offline-mode'.
```
* Using the documentation from huggingface https://huggingface.co/docs/transformers/installation#offline-mode
* Run the file "download_longformer_model.py" in the local environment
* Prepare a zip file of the just created folder "longformer-base-4096"
* Upload the zip file with the model to the cluster
```
scp [path]\longformer-base-4096.zip correa@master.ismll.de:/home/correa/model_longformer_training
```
* Unzip zip file into "/home/correa/model_longformer_training" folder
```
unzip ./longformer-base-4096.zip
```
* Delete zip file
```
rm ./longformer-base-4096.zip
```

## Arguments
When running the script you can send some command line arguments
### train
* Value: False or True
* name: -t or --train
```
model_longformer_training.py -t False
model_longformer_training.py --train True
```

## Run script
* Upload input file, from local host to cluster
```
scp [path]\g_cases_2021.csv correa@master.ismll.de:/home/correa/model_longformer_training/data
scp [path]\en_2021-01-01_output.csv correa@master.ismll.de:/home/correa/model_longformer_training/data
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
* Once the job is finished the folder will include "models" and "model_longformer.model" folders and err and log files
```
(base) correa@master:~/model_longformer_training/models/longformer-fine-tuned-regression$ ls -ltr
total 80
drwxr-xr-x 2 correa students 4096 Oct  3 14:00 checkpoint-1
drwxr-xr-x 2 correa students 4096 Oct  3 14:01 checkpoint-2
drwxr-xr-x 2 correa students 4096 Oct  3 14:02 checkpoint-3
drwxr-xr-x 2 correa students 4096 Oct  3 14:02 checkpoint-4
drwxr-xr-x 2 correa students 4096 Oct  3 14:03 checkpoint-5
drwxr-xr-x 2 correa students 4096 Oct  3 14:04 checkpoint-6
drwxr-xr-x 2 correa students 4096 Oct  3 14:04 checkpoint-7
drwxr-xr-x 2 correa students 4096 Oct  3 14:05 checkpoint-8
drwxr-xr-x 2 correa students 4096 Oct  3 14:06 checkpoint-9
drwxr-xr-x 2 correa students 4096 Oct  3 14:06 checkpoint-10
drwxr-xr-x 2 correa students 4096 Oct  3 14:07 checkpoint-11
drwxr-xr-x 2 correa students 4096 Oct  3 14:08 checkpoint-12
drwxr-xr-x 2 correa students 4096 Oct  3 14:08 checkpoint-13
drwxr-xr-x 2 correa students 4096 Oct  3 14:09 checkpoint-14
drwxr-xr-x 2 correa students 4096 Oct  3 14:10 checkpoint-15
drwxr-xr-x 2 correa students 4096 Oct  3 14:10 checkpoint-16
drwxr-xr-x 2 correa students 4096 Oct  3 14:11 checkpoint-17
drwxr-xr-x 2 correa students 4096 Oct  3 14:12 checkpoint-18
drwxr-xr-x 2 correa students 4096 Oct  3 14:12 checkpoint-19
drwxr-xr-x 2 correa students 4096 Oct  3 14:13 checkpoint-20

(base) correa@master:~/model_longformer_training/models/longformer-fine-tuned-regression/checkpoint-1$ ls -ltr
total 3927936
-rw-r--r-- 1 correa students        760 Oct  6 14:23 config.json
-rw-r--r-- 1 correa students 1340700593 Oct  6 14:23 pytorch_model.bin
-rw-r--r-- 1 correa students       4091 Oct  6 14:23 training_args.bin
-rw-r--r-- 1 correa students 2681460792 Oct  6 14:24 optimizer.pt
-rw-r--r-- 1 correa students        783 Oct  6 14:24 trainer_state.json
-rw-r--r-- 1 correa students        627 Oct  6 14:24 scheduler.pt
-rw-r--r-- 1 correa students      13553 Oct  6 14:24 rng_state.pth
```
```
(base) correa@master:~/model_longformer_training/model_longformer.model$ ls -ltr
total 1310224
-rw-r--r-- 1 correa students        760 Oct  6 14:25 config.json
-rw-r--r-- 1 correa students 1340700593 Oct  6 14:25 pytorch_model.bin
-rw-r--r-- 1 correa students       4091 Oct  6 14:25 training_args.bin
-rw-r--r-- 1 correa students        394 Oct  6 14:25 tokenizer_config.json
-rw-r--r-- 1 correa students        125 Oct  6 14:25 special_tokens_map.json
-rw-r--r-- 1 correa students     231508 Oct  6 14:25 vocab.txt
-rw-r--r-- 1 correa students     711562 Oct  6 14:25 tokenizer.json
```

## References
* https://huggingface.co/docs/transformers/model_doc/longformer