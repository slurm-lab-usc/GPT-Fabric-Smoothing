# Installation guide
This is to provide a complete installation guide on GPT-Fabric-Smoothing
## Install Softgym
### 1
After clone this repo, create the `gptfab` environment by running `conda env create -f environment.yml`

### 2

Make sure you have replaced `PATH_TO_GPT_FABRIC` and `PATH_TO_CONDA` with the corresponding paths (make sure to use absolute path!).
Then recompile Softgym via running:
```
docker run \
    -v PATH_TO_GPT_FABRIC:/workspace/softgym \
    -v PATH_TO_CONDA:PATH_TO_CONDA \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -e DISPLAY=$DISPLAY \
    -e QT_X11_NO_MITSHM=1 \
    -it xingyu/softgym:latest bash
```



### 3
Then you should be entering the Docker and inside Docker:
```
root@9ac1efa91ca9:/workspace# cd softgym/
root@9ac1efa91ca9:/workspace/softgym# export PATH="PATH_TO_CONDA/bin:$PATH"
root@9ac1efa91ca9:/workspace/softgym# . ./prepare_1.0.sh 
(softgym) root@9ac1efa91ca9:/workspace/softgym# . ./compile_1.0.sh
```

You are all set when seeing at the end.
```
[100%] Linking CXX shared module pyflex.cpython-38-x86_64-linux-gnu.so
[100%] Built target pyflex
```

### 4
To run from the "normal" command lines, one can input such lines:
```
conda activate gptfab
export PYFLEXROOT=${PWD}/PyFlex
export PYTHONPATH=${PYFLEXROOT}/bindings/build:$PYTHONPATH
export LD_LIBRARY_PATH=${PYFLEXROOT}/external/SDL2-2.0.4/lib/x64:$LD_LIBRARY_PATH
```

A good alternative to this is to put them in a `.sh` file, my version is `prepare_gpt.sh` in the `GPT-FABRIC` folder. I execute by running `. ./prepare_gpt.sh`. The expected command lines should look like this:
```
(base) enyuzhao@blackcoffee:~/code/softgym$ . ./prepare_gpt.sh
(softgym) enyuzhao@blackcoffee:~/code/softgym$
```

### 5 
Then you can verify the installation by executing `python RGBD_manipulation`

