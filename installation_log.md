# Enyu Steps 
This is to record my installation.
## Install Softgym
### 1:
After cloning the softgym repo, first chaning into `vcd` branch, then make changes to the `environment.yml` file shown below:
```
name: gptfab
channels:
  - defaults
dependencies:
  - python=3.8
  - numpy
  - h5py
  - imageio
  - glob2
  - cmake
  - pybind11
  - click
  - matplotlib
  - joblib
  - Pillow
  - plotly
  - ipython
  - pip
  - pip:
      - gtimer
      - opencv-python
      - Shapely
      - sk-video
      - moviepy
      - gym==0.15.7
      - pyquaternion==0.9.5

```

### 2
After the changes are made, create the `softgym` environment by running `conda env create -f environment.yml`

### 3
Then recompile Softgym on the `vcd` branch via running:
```
docker run \
    -v /home/enyuzhao/code/softgym:/workspace/softgym \
    -v /home/enyuzhao/miniconda3:/home/enyuzhao/miniconda3 \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    -e DISPLAY=$DISPLAY \
    -e QT_X11_NO_MITSHM=1 \
    -it xingyu/softgym:latest bash
```

change the paths if needed

### 4
Then you should be entering the Docker and inside Docker:
```
root@9ac1efa91ca9:/workspace# cd softgym/
root@9ac1efa91ca9:/workspace/softgym# export PATH="/home/enyuzhao/miniconda3/bin:$PATH"
root@9ac1efa91ca9:/workspace/softgym# . ./prepare_1.0.sh 
(softgym) root@9ac1efa91ca9:/workspace/softgym# . ./compile_1.0.sh
```

You are all set when seeing at the end.
```
[100%] Linking CXX shared module pyflex.cpython-38-x86_64-linux-gnu.so
[100%] Built target pyflex
```

### 5
To run from the "normal" command lines, one can input such lines:
```
conda activate softgym
export PYFLEXROOT=${PWD}/PyFlex
export PYTHONPATH=${PYFLEXROOT}/bindings/build:$PYTHONPATH
export LD_LIBRARY_PATH=${PYFLEXROOT}/external/SDL2-2.0.4/lib/x64:$LD_LIBRARY_PATH
```

A good alternative to this is to put them in a `.sh` file, my version is `prepare_gpt.sh` in the `softgym` folder. I execute by running `. ./prepare_gpt.sh`. The expected command lines should look like this:
```
(base) enyuzhao@blackcoffee:~/code/softgym$ . ./prepare_gpt.sh
(softgym) enyuzhao@blackcoffee:~/code/softgym$
```

### 6 
Then you can verify the installation by executing `python examples/random_env.py --env_name ClothFlatten --headless 1`

To deal with potential error listed here:
```
AttributeError: module 'numpy' has no attribute 'float'.
`np.float` was a deprecated alias for the builtin `float`. To avoid this error in existing code, use `float` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.float64` here.
The aliases was originally deprecated in NumPy 1.20; for more details and guidance see the original release note at:
    https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
```

You can try to install `numpy==1.20`, `matplotlib==3.6`

I am encountering this error here:
```
(softgym) enyuzhao@blackcoffee:~/code/softgym$ python examples/random_env.py --env_name PourWater
Waiting to generate environment variations. May take 1 minute for each variation...
Unable to initialize SDLCould not initialize GL extensions
Reshaping
Segmentation fault (core dumped)
```
By changing `python examples/random_env.py --env_name PourWater` into `python examples/random_env.py --env_name PourWater -headless 1`, it works:
```
(softgym) enyuzhao@blackcoffee:~/code/softgym$ python examples/random_env.py --env_name PourWater --headless 1
Waiting to generate environment variations. May take 1 minute for each variation...
Compute Device: NVIDIA GeForce RTX 4090

Pyflex init done!
pour water generate env variations 0
generate env variation: medium volume water
stablize water!
MoviePy - Building file ./data/PourWater.gif with imageio.
Video generated and save to ./data/PourWater.gif
```

## Install VCD
### 1
Set the symlink as following:
```
(softgym) enyuzhao@blackcoffee:~/code/zero-shot-fabric-manipulation$ ln -s ../softgym softgym
(softgym) enyuzhao@blackcoffee:~/code/zero-shot-fabric-manipulation$ ls -lh
total 40K
drwxrwxr-x 7 enyuzhao enyuzhao 4.0K Oct 10 21:30 chester
-rwxrwxr-x 1 enyuzhao enyuzhao  278 Oct 10 21:30 compile_1.0.sh
-rw-rw-r-- 1 enyuzhao enyuzhao  443 Oct 10 21:30 environment.yml
-rw-rw-r-- 1 enyuzhao enyuzhao 4.2K Oct 11 01:21 installation_log.md
-rw-rw-r-- 1 enyuzhao enyuzhao 1.1K Oct 10 21:30 LICENSE
-rwxrwxr-x 1 enyuzhao enyuzhao  276 Oct 10 22:56 prepare_1.0.sh
drwxrwxr-x 2 enyuzhao enyuzhao 4.0K Oct 10 21:30 pretrained
-rw-rw-r-- 1 enyuzhao enyuzhao 3.3K Oct 10 21:30 README.md
lrwxrwxrwx 1 enyuzhao enyuzhao   10 Oct 11 01:18 softgym -> ../softgym
drwxrwxr-x 3 enyuzhao enyuzhao 4.0K Oct 10 21:30 VCD
```

### 2
Modify the `prepare_1.0.sh` file in the `zero-shot-fabric-manipulation` folder as following:
```
cd softgym
. ./prepare_gpt.sh
cd ..
export PYFLEXROOT=${PWD}/softgym/PyFlex
export PYTHONPATH=${PWD}:${PWD}/softgym:${PYFLEXROOT}/bindings/build:$PYTHONPATH
export LD_LIBRARY_PATH=${PYFLEXROOT}/external/SDL2-2.0.4/lib/x64:$LD_LIBRARY_PATH
export EGL_GPU=$CUDA_VISIBLE_DEVICES
```
Then run `. ./prepare_1.0.sh` to set up the path.

If you see errors like below:
```
(softgym) enyuzhao@blackcoffee:~/code/zero-shot-fabric-manipulation$ . ./prepare_1.0.sh
bash: cd: softgym: No such file or directory
bash: ./prepare_gpt.sh: No such file or directory
```

Then that's because you didn't set the symlink first. You should set the symlink at first then run this `. ./prepare_1.0.sh`.
### 3

Next to install the additial packages.
We choose `pytorch==1.12.0` for this case, with following command to install:
```
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.6 -c pytorch -c conda-forge
```

Verify torch installation via:
```
(softgym) enyuzhao@blackcoffee:~/code/zero-shot-fabric-manipulation$ ipython
Python 3.8.18 (default, Sep 11 2023, 13:40:15) 
Type 'copyright', 'credits' or 'license' for more information
IPython 8.12.2 -- An enhanced Interactive Python. Type '?' for help.

In [1]: import torch

In [2]: torch.__version__, torch.cuda.is_available(),torch.cuda.get_device_name(0)
Out[2]: ('1.12.0', True, 'NVIDIA GeForce RTX 4090')

In [3]: exit
```
### 4 

Run the example script to verify the installation:`python VCD/generate_cached_initial_state.py`

Encounter errors here:
```
(softgym) enyuzhao@blackcoffee:~/code/zero-shot-fabric-manipulation$ python VCD/generate_cached_initial_state.py
Traceback (most recent call last):
  File "VCD/generate_cached_initial_state.py", line 3, in <module>
    from VCD.main import get_default_args
  File "/home/enyuzhao/code/zero-shot-fabric-manipulation/VCD/main.py", line 5, in <module>
    from VCD.utils.utils import vv_to_args, set_resource
  File "/home/enyuzhao/code/zero-shot-fabric-manipulation/VCD/utils/utils.py", line 9, in <module>
    import h5py
  File "/home/enyuzhao/miniconda3/envs/softgym/lib/python3.8/site-packages/h5py/__init__.py", line 25, in <module>
    from . import _errors
  File "h5py/_errors.pyx", line 1, in init h5py._errors
ValueError: numpy.ndarray size changed, may indicate binary incompatibility. Expected 88 from C header, got 80 from PyObject
```
This is caused by using `numpy==1.19.5`, suggest use `numpy==1.20`

#### 4.1
If using `numpy==1.20`, you would see the following error:
```
Traceback (most recent call last):
  File "VCD/generate_cached_initial_state.py", line 3, in <module>
    from VCD.main import get_default_args
  File "/home/enyuzhao/code/zero-shot-fabric-manipulation/VCD/main.py", line 5, in <module>
    from VCD.utils.utils import vv_to_args, set_resource
  File "/home/enyuzhao/code/zero-shot-fabric-manipulation/VCD/utils/utils.py", line 36, in <module>
    import pcl
ModuleNotFoundError: No module named 'pcl'
```

This can be solve by following this ![link](https://github.com/Xingyu-Lin/VCD/issues/5)



first you need to go to the `VCD\utils.py` file and change following code:

```
import pcl


# def get_partial_particle(full_particle, observable_idx):
#     return np.array(full_particle[observable_idx], dtype=np.float32)


def voxelize_pointcloud(pointcloud, voxel_size):
    cloud = pcl.PointCloud(pointcloud)
    sor = cloud.make_voxel_grid_filter()
    sor.set_leaf_size(voxel_size, voxel_size, voxel_size)
    pointcloud = sor.filter()
    pointcloud = np.asarray(pointcloud).astype(np.float32)
    return pointcloud
```


into:

```
import open3d as o3d
def voxelize_pointcloud(pc, voxel_size=0.00625*10):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    try:
        voxelized_pcd = pcd.voxel_down_sample(voxel_size)
    except RuntimeError:
        return None
    voxelized_pc = np.asarray(voxelized_pcd.points)
    return voxelized_pc

```

#### 4.2
Then installing those packages: `open3d`,`wandb`,`torch_geometric`,`torch_scatter`.

After all those packages are installed, you might run into :
```
AttributeError: module 'numpy' has no attribute 'float'.
`np.float` was a deprecated alias for the builtin `float`. To avoid this error in existing code, use `float` by itself. Doing this will not modify any behavior and is safe. If you specifically wanted the numpy scalar type, use `np.float64` here.
The aliases was originally deprecated in NumPy 1.20; for more details and guidance see the original release note at:
    https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations
```
The solution to this is to change all the `np.float` in this repo into `np.float64`.



I am encountering this following error here while try to install `torch_scatter`:
```
(softgym) enyuzhao@blackcoffee:~/code/zero-shot-fabric-manipulation$ pip install torch_scatter
Collecting torch_scatter
  Downloading torch_scatter-2.1.2.tar.gz (108 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 108.0/108.0 kB 11.4 MB/s eta 0:00:00
  Preparing metadata (setup.py) ... done
Building wheels for collected packages: torch_scatter
  Building wheel for torch_scatter (setup.py) ... error
  error: subprocess-exited-with-error
  
  × python setup.py bdist_wheel did not run successfully.
  │ exit code: 1
  ╰─> [74 lines of output]
      running bdist_wheel
      running build
      running build_py
      creating build
      creating build/lib.linux-x86_64-cpython-38
      creating build/lib.linux-x86_64-cpython-38/torch_scatter
      copying torch_scatter/utils.py -> build/lib.linux-x86_64-cpython-38/torch_scatter
      copying torch_scatter/segment_coo.py -> build/lib.linux-x86_64-cpython-38/torch_scatter
      copying torch_scatter/placeholder.py -> build/lib.linux-x86_64-cpython-38/torch_scatter
      copying torch_scatter/scatter.py -> build/lib.linux-x86_64-cpython-38/torch_scatter
      copying torch_scatter/segment_csr.py -> build/lib.linux-x86_64-cpython-38/torch_scatter
      copying torch_scatter/testing.py -> build/lib.linux-x86_64-cpython-38/torch_scatter
      copying torch_scatter/__init__.py -> build/lib.linux-x86_64-cpython-38/torch_scatter
      creating build/lib.linux-x86_64-cpython-38/torch_scatter/composite
      copying torch_scatter/composite/softmax.py -> build/lib.linux-x86_64-cpython-38/torch_scatter/composite
      copying torch_scatter/composite/std.py -> build/lib.linux-x86_64-cpython-38/torch_scatter/composite
      copying torch_scatter/composite/logsumexp.py -> build/lib.linux-x86_64-cpython-38/torch_scatter/composite
      copying torch_scatter/composite/__init__.py -> build/lib.linux-x86_64-cpython-38/torch_scatter/composite
      running egg_info
      writing torch_scatter.egg-info/PKG-INFO
      writing dependency_links to torch_scatter.egg-info/dependency_links.txt
      writing requirements to torch_scatter.egg-info/requires.txt
      writing top-level names to torch_scatter.egg-info/top_level.txt
      reading manifest file 'torch_scatter.egg-info/SOURCES.txt'
      reading manifest template 'MANIFEST.in'
      warning: no previously-included files matching '*' found under directory 'test'
      adding license file 'LICENSE'
      writing manifest file 'torch_scatter.egg-info/SOURCES.txt'
      running build_ext
      Traceback (most recent call last):
        File "<string>", line 2, in <module>
        File "<pip-setuptools-caller>", line 34, in <module>
        File "/tmp/pip-install-o0_baquv/torch-scatter_9201ecd5021a4e2689f1b8b3d86cd5fe/setup.py", line 120, in <module>
          setup(
        File "/home/enyuzhao/miniconda3/envs/softgym/lib/python3.8/site-packages/setuptools/__init__.py", line 107, in setup
          return distutils.core.setup(**attrs)
        File "/home/enyuzhao/miniconda3/envs/softgym/lib/python3.8/site-packages/setuptools/_distutils/core.py", line 185, in setup
          return run_commands(dist)
        File "/home/enyuzhao/miniconda3/envs/softgym/lib/python3.8/site-packages/setuptools/_distutils/core.py", line 201, in run_commands
          dist.run_commands()
        File "/home/enyuzhao/miniconda3/envs/softgym/lib/python3.8/site-packages/setuptools/_distutils/dist.py", line 969, in run_commands
          self.run_command(cmd)
        File "/home/enyuzhao/miniconda3/envs/softgym/lib/python3.8/site-packages/setuptools/dist.py", line 1234, in run_command
          super().run_command(command)
        File "/home/enyuzhao/miniconda3/envs/softgym/lib/python3.8/site-packages/setuptools/_distutils/dist.py", line 988, in run_command
          cmd_obj.run()
        File "/home/enyuzhao/miniconda3/envs/softgym/lib/python3.8/site-packages/wheel/bdist_wheel.py", line 364, in run
          self.run_command("build")
        File "/home/enyuzhao/miniconda3/envs/softgym/lib/python3.8/site-packages/setuptools/_distutils/cmd.py", line 318, in run_command
          self.distribution.run_command(command)
        File "/home/enyuzhao/miniconda3/envs/softgym/lib/python3.8/site-packages/setuptools/dist.py", line 1234, in run_command
          super().run_command(command)
        File "/home/enyuzhao/miniconda3/envs/softgym/lib/python3.8/site-packages/setuptools/_distutils/dist.py", line 988, in run_command
          cmd_obj.run()
        File "/home/enyuzhao/miniconda3/envs/softgym/lib/python3.8/site-packages/setuptools/_distutils/command/build.py", line 131, in run
          self.run_command(cmd_name)
        File "/home/enyuzhao/miniconda3/envs/softgym/lib/python3.8/site-packages/setuptools/_distutils/cmd.py", line 318, in run_command
          self.distribution.run_command(command)
        File "/home/enyuzhao/miniconda3/envs/softgym/lib/python3.8/site-packages/setuptools/dist.py", line 1234, in run_command
          super().run_command(command)
        File "/home/enyuzhao/miniconda3/envs/softgym/lib/python3.8/site-packages/setuptools/_distutils/dist.py", line 988, in run_command
          cmd_obj.run()
        File "/home/enyuzhao/miniconda3/envs/softgym/lib/python3.8/site-packages/setuptools/command/build_ext.py", line 84, in run
          _build_ext.run(self)
        File "/home/enyuzhao/miniconda3/envs/softgym/lib/python3.8/site-packages/setuptools/_distutils/command/build_ext.py", line 345, in run
          self.build_extensions()
        File "/home/enyuzhao/miniconda3/envs/softgym/lib/python3.8/site-packages/torch/utils/cpp_extension.py", line 434, in build_extensions
          self._check_cuda_version(compiler_name, compiler_version)
        File "/home/enyuzhao/miniconda3/envs/softgym/lib/python3.8/site-packages/torch/utils/cpp_extension.py", line 812, in _check_cuda_version
          raise RuntimeError(CUDA_MISMATCH_MESSAGE.format(cuda_str_version, torch.version.cuda))
      RuntimeError:
      The detected CUDA version (12.2) mismatches the version that was used to compile
      PyTorch (11.6). Please make sure to use the same CUDA versions.
      
      [end of output]
  
  note: This error originates from a subprocess, and is likely not a problem with pip.
  ERROR: Failed building wheel for torch_scatter
  Running setup.py clean for torch_scatter
Failed to build torch_scatter
ERROR: Could not build wheels for torch_scatter, which is required to install pyproject.toml-based projects


```


If you see this error, try to install the `torch_scatter` via running this command `pip install torch_scatter==2.1.0 -f https://data.pyg.org/whl/torch-1.12.1+cu116.html`, the `torch-1.12.1+cu116` should be your torch and cuda version.


The successful installation can be verified if you see the following command lines after executing `python VCD/generate_cached_initial_state.py --num_variations 3`:
```
(softgym) enyuzhao@blackcoffee:~/code/zero-shot-fabric-manipulation$ python VCD/generate_cached_initial_state.py --num_variations 3
Compute Device: NVIDIA GeForce RTX 4090

Pyflex init done!
config 0: camera params {'default_camera': {'pos': array([-0.  ,  0.82,  0.82]), 'angle': array([ 0.        , -0.78539816,  0.        ]), 'width': 360, 'height': 360}}, flatten area: 0.07558593749999999
config 1: camera params {'default_camera': {'pos': array([-0.  ,  0.82,  0.82]), 'angle': array([ 0.        , -0.78539816,  0.        ]), 'width': 360, 'height': 360}}, flatten area: 0.06566406250000004
config 2: camera params {'default_camera': {'pos': array([-0.  ,  0.82,  0.82]), 'angle': array([ 0.        , -0.78539816,  0.        ]), 'width': 360, 'height': 360}}, flatten area: 0.0671875
This is the cache path! /home/enyuzhao/code/zero-shot-fabric-manipulation/softgym/softgym/envs/../cached_initial_states/1213_release_n1000.pkl
Traceback (most recent call last):
  File "VCD/generate_cached_initial_state.py", line 35, in <module>
    env = create_env(args)
  File "VCD/generate_cached_initial_state.py", line 30, in create_env
    return SOFTGYM_ENVS[args.env_name](**env_args)
  File "/home/enyuzhao/code/zero-shot-fabric-manipulation/softgym/softgym/envs/cloth_flatten.py", line 18, in __init__
    self.get_cached_configs_and_states(cached_states_path, self.num_variations)
  File "/home/enyuzhao/code/zero-shot-fabric-manipulation/softgym/softgym/envs/flex_env.py", line 101, in get_cached_configs_and_states
    with open(cached_states_path, 'wb') as handle:
FileNotFoundError: [Errno 2] No such file or directory: '/home/enyuzhao/code/zero-shot-fabric-manipulation/softgym/softgym/envs/../cached_initial_states/1213_release_n1000.pkl'
```

Due to default `num_varations` is 1000, so 3 is only to let you check whether the dependencies of the packages are sorted or not. If you see the lines rolling, then the dependencies should be all set

The error is caused by not having a folder `cached_initial_states` in the `softgym/softgym` folder, you can create one and re-execute the command to see the following result.

```
python VCD/generate_cached_initial_state.py --num_variations 3
Compute Device: NVIDIA GeForce RTX 4090

Pyflex init done!
config 0: camera params {'default_camera': {'pos': array([-0.  ,  0.82,  0.82]), 'angle': array([ 0.        , -0.78539816,  0.        ]), 'width': 360, 'height': 360}}, flatten area: 0.06886718750000001
config 1: camera params {'default_camera': {'pos': array([-0.  ,  0.82,  0.82]), 'angle': array([ 0.        , -0.78539816,  0.        ]), 'width': 360, 'height': 360}}, flatten area: 0.06406250000000001
config 2: camera params {'default_camera': {'pos': array([-0.  ,  0.82,  0.82]), 'angle': array([ 0.        , -0.78539816,  0.        ]), 'width': 360, 'height': 360}}, flatten area: 0.0671875
This is the cache path! /home/enyuzhao/code/zero-shot-fabric-manipulation/softgym/softgym/envs/../cached_initial_states/1213_release_n1000.pkl
3 config and state pairs generated and saved to /home/enyuzhao/code/zero-shot-fabric-manipulation/softgym/softgym/envs/../cached_initial_states/1213_release_n1000.pkl
```

Then it's all set.