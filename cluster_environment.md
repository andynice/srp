# Cluster environment preparation
## connect to cluster
* In order to connect to the cluster by SHH you need
    * to be connected to the University's VPN. You can find more info here https://www.uni-hildesheim.de/rz/uni-vpn/
    * you need username and password. Those are not included for security purposes, but you can contact the team to ask for them They were provided by email:

```
ssh correa@master.ismll.de
```

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
cat ./.bashrc
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