# Cluster environment for led model training

# Model LED Training
## Prepare directory
* Create a directory "model_led_training"
```
mkdir model_led_training
```
## Prepare tokenization process
* Create a directory inside "model_led_training" called "tokenize_led_output"
```
cd ./model_led_training
mkdir tokenize_led_output
```
* Create a directory inside "model_led_training" called "data"
```
cd ./model_led_training
mkdir data
```
## Run tokenization
* Create the python script to tokenize "tokenize_led.py"
* Create the python script to tokenize "test_tokenize.sh"
* Upload input file, from local host to cluster
```
scp [path]\g_cases_2021.csv correa@master.ismll.de:/home/correa/model_led_training/data
scp [path]\en_2021-01-01_output.csv correa@master.ismll.de:/home/correa/model_led_training/data
```
* Create the bash script "test_tokenize.sh"
```
vi test_test_tokenize.sh
```
* Run the bash script "test_test_tokenize.sh"
```
sbatch test_test_tokenize.sh
Submitted batch job 449092
```
 
## Run training
* Create the python scripts
```
vi model_led_trainer_training.py
vi model_led_no_trainer_training.py
```

## Prepare huggingface model offline
* Nodes are not connected to internet, so it's necessary to download first everything needed
* Otherwise, you will get an error message like this
```
OSError: We couldn't connect to 'https://huggingface.co' to load this file, couldn't find it in the cached files and it looks like digitalepidemiologylab/covid-twitter-bert is not the path to a directory containing a file named config.json.
Checkout your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offline-mode'.
```
* Using the documentation from huggingface https://huggingface.co/docs/transformers/installation#offline-mode
* Run the file "download_led_model.py" in the local environment
* Prepare a zip file of the just created folder "led-base-16384"
* Upload the zip file with the model to the cluster
```
scp [path]\led-base-16384.zip correa@master.ismll.de:/home/correa/model_led_training
```
* Unzip zip file into "/home/correa/model_led_training" folder
```
unzip ./led-base-16384.zip
```
* Delete zip file
```
rm ./led-base-16384.zip
```

## Arguments
When running the script you can send some command line arguments
### train
* Value: False or True
* name: -t or --train
```
model_led_training.py -t False
model_led_training.py --train True
```

## Run script
* Upload input file, from local host to cluster
```
scp [path]\g_cases_2021.csv correa@master.ismll.de:/home/correa/model_led_training/data
scp [path]\en_2021-01-01_output.csv correa@master.ismll.de:/home/correa/model_led_training/data
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
* Once the job is finished the folder will include "models" and "model_led.model" folders and err and log files
```
(base) correa@master:~/model_led_training/models/led-fine-tuned-regression$ ls -ltr
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

(base) correa@master:~/model_led_training/models/led-fine-tuned-regression/checkpoint-1$ ls -ltr
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
(base) correa@master:~/model_led_training/model_led.model$ ls -ltr
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
* https://huggingface.co/docs/transformers/model_doc/led