# GPT-Fabric: Folding and Smoothing Fabric by Leveraging Pre-Trained Foundation Models
**Vedant Raval\*, Enyu Zhao\*, Hejia Zhang, Stefanos Nikolaidis, Daniel Seita**

**University of Southern California**

This repository is a python implementation of the paper "GPT-Fabric: Folding and Smoothing Fabric by Leveraging Pre-Trained Foundation Models", submitted to IROS 2024. This repository contains the code used to run the GPT-fabric simulation experiments for fabric smoothing and heavily built upon [SoftGym](https://github.com/Xingyu-Lin/softgym). The code for performing fabric smoothing can be found in the repo [GPT-Fabric-Folding](https://github.com/slurm-lab-usc/GPT-fabric-folding/tree/main)

[Website](https://sites.google.com/usc.edu/gpt-fabrics/home) | [ArXiv: Coming soon]()

**Important Update: According to OpenAI, the "gpt-4-vision-preview" model checkpoint used in GPT-Fabric-Smoothing is going to be deprecated. It's encouraged to use GPT-4o instead**

## Table of Contents
* [Installation](#Installation)
* [GPT-Fabric-smoothing,zero-shot](#evaluating-gpt-fabric-in-zero-shot-setting-with-random-starting-config)
* [GPT-Fabric-smoothing,cached_states](#evaluating-with-cached-starting-configs-of-the-fabric)
* [License](#license)
* [Acknowledgements](#acknowledgements)
* [Contact](#contact)

## Installation



This simulation environment is based on [SoftGym](https://github.com/Xingyu-Lin/softgym). However, in this repo we made some changes to the original [SoftGym](https://github.com/Xingyu-Lin/softgym), so it's recommended to just use the SoftGym provided in this repo. 


1. Clone this repository.

2. [A nice blog](https://danieltakeshi.github.io/2021/02/20/softgym/) written by Daniel Seita may help you get started on SoftGym. You can also refer to the `installation_log.md` for a complete walk-through installation guide on SoftGym.

3. Before you use the code, you should make sure the conda environment activated(`conda activate gptfab`) and set up the paths appropriately (It's highly recommened to run `. ./prepare_gpt.sh` instead of `conda activate` and set up the following paths): 
   ~~~
   export PYFLEXROOT=${PWD}/PyFlex
   export PYTHONPATH=${PYFLEXROOT}/bindings/build:$PYTHONPATH
   export LD_LIBRARY_PATH=${PYFLEXROOT}/external/SDL2-2.0.4/lib/x64:$LD_LIBRARY_PATH
   ~~~
   The provided script `prepare_gpt.sh` includes these commands above.

## Evaluating GPT-Fabric in zero-shot setting with random starting config
You should run the following code with changing the working directory into this directory (i.e. GPT-FABRIC-SMOOTHING)

1. First you should **paste your GPT API-key into the** `GPT-API-Key.txt` **file**.

2. Then run the following code to prepare the environment if you didn't activate the conda environment:

    ```
    . ./prepare_gpt.sh
    ```

3. Run following code
    ```
    python RGBD_manipulation.py
    ```

You will be able to see the result in the `tests` folder.

We also provide several parameters for more control of the manipulation:
- `method_name`:specify the method name and its configuration, we use 'RGBD_simple' in our paper. 
- `headless`: whether to render the scene. If you want to render the scene and visualize it while processing, you can set it to be 0. Otherwise you can set it to be 1. Note that the .gif file recording the manipulation will always be saved so don't worry.
- `trails`: How many pick-and-place steps you want to take. Default to be 5.
- `gif_speed`: The speed multiple the recorded file. Default to be 4 (Meaning the .gif file will be 4x faster) 
- `save_obs_dir`: where to store the observation (and other related files) of this rollout. Defualt to be `./tests`.
- `specifier`: The suffix of the observation folder.

For the observation folder, it should be in this format:
```
./save_obs_dir/method_name[specifier]
```

For example, if you run:
```
python RGBD_manipulation.py --method_name RGBD_simple --headless 0 --trails 10 --gif_speed 4 --save_obs_dir tests --specifier _env_config_test
```

The folder contains all the observation and the recorded .gif file will be at `./tests/RGBD_simple_env_config_test`.

## Evaluating with cached starting configs of the fabric

You should run the following code with changing the working directory into this directory (i.e. GPT-FABRIC-SMOOTHING)

1. First you should **paste your GPT API-key into the** `GPT-API-Key.txt` **file**.

2. Then run the following code to prepare the environment if you didn't activate the conda environment:

    ```
    . ./prepare_gpt.sh
    ```

3. Be sure to have a folder containing the the cached states (We provided the 40 configs from [VCD](https://arxiv.org/abs/2105.10389) paper. The `cloth_flatten_states_40_test` contains the first 20 starting configs and the `cloth_flatten_states_40_test_2` contains the rest.)

4. Run following code
    ```
    python cached_config_test_RGB.py --cache_state_path cloth_flatten_states_40_test --num_variations 20 --save_obs_dir ./1_20_config_test_5 --method_name RGBD_simple --reps 1 --trails 5 --starting_config 0
    ```

You will be able to see the result in the `1_20_config_test_5/RGBD_simple` folder. The folder structure looks like this:

```
save_obs_dir/method_name/state_[i]/rep[j]
```


We also provide several parameters for more control of the manipulation:
- `method_name`:specify the method name and its configuration, we use 'RGBD_simple' in our paper. 
- `headless`: whether to render the scene. If you want to render the scene and visualize it while processing, you can set it to be 0. Otherwise you can set it to be 1. Note that the .gif file recording the manipulation will always be saved so don't worry.
- `cache_state_path`: where the cached states are in
- `use_cached_states`: whether to use the cached states in the `cache_state_path`. If not, you can set it to be 0. Otherwise you can set it to 1. For evaluation purpose, it's better to be 1. 
- `save_cached_states`: whether to save the states to the `cache_state_path`. If not, you can set it to be 0. Otherwise you can set it to 1. If you just want to evaluate, you can set it to 0. But if you want to create new starting configs, you should set it to be 1. 
- `num_variation`: the number of the cached states, it should align with your cached states.
- `save_image_dir`: the folder to save the images corresponding to the cached states if you want to save them. Only valid if `save_cached_states` is 1.

- `starting_config`: which config in the cached states to start with. Default to be 0
- `reps`: How many repetitive experiments you want to carry out on the same starting config.
- `trails`: How many pick-and-place steps you want to take. Default to be 5.
- `gif_speed`: The speed multiple the recorded file. Default to be 4 (Meaning the .gif file will be 4x faster) 
- `save_obs_dir`: where to store the observation (and other related files) of this rollout. Defualt to be `./10_env_tests`.


## Acknowledgements

Coming soon

## Contact

For any additional questions, feel free to email [enyuzhao@usc.edu](enyuzhao@usc.edu) or [zhaoenyu344@gmail.com](zhaoenyu344@gmail.com)
