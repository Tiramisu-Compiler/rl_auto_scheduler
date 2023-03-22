# RL for performance optimization
The sections that follow outline the steps required to successfully train the RL agent.


## Generate the dataset
To create the dataset, run the script `tiramisu maker.py` from the submodule `RandomTiramisu` located under the `utils` directory. Below are the script options and their default values:
```
  -h, --help            show this help message and exit
  --output-path OUTPUT_PATH [Default = "Dataset_multi"]
                        The location where to save the dataset.
  --first-seed FIRST_SEED  [Default = 105100]
                        The first seed from which the programs will be generated. Every seed yields a program, and two separate seeds make two different programs.
  --nb-programs NB_PROGRAMS  [Default = 20000]
                        The number of programs to generate.

```


## Configuring the repository
To set up the repository, first copy the template files to the same place but without the `.template` extension, as seen below:
```bash  
cp config.yaml.template config.yaml
cp scripts/env.sh.template scripts/env.sh
cp scripts/run_rllib_slurm.sh.template scripts/run_rllib_slurm.sh
```
Then, depending on the context in which the code is running, modify the fields to the appropriate value.

### RL system configuration

The file `config.yml` contains all configurations for the RL system. The most critical fields in this file, without which the code would not run, are:  
* `ray.base_path`:  The absolute path to where the code is located on disk. 
* `environment.dataset_path`: The path to the training dataset. 
* `tiramisu.tiramisu_path`: The absolute path to the complete tiramisu installation, with the autoscheduler also installed.
* `tiramisu.model_checkpoint`: The path to the surrogate model used for prediction. A model chackpoint is provided under `weights/surrogate_model`.

The other flags can also be set for a specific purpose. You can, for example, modify the policy model's characteristics. You may also choose whether or not to preserve the dataset files after utilizing them for an episode. 

### SLURM configuration

The file `scripts/env.sh` is used in order to configure the job that runs on SLURM. The most important fields in this file, that the code will not run without setting are:
* `CONDA_DIR`: The path to the conda installation.
* `CONDA_ENV`: The path to the specific conda environment.
Other flags can be set in order to control the number of workers to use in training.  

To get the conda environment location, if the desired conda is already activated, you follow these steps:  
```bash  
conda activate base
which python
# example: /home/user/miniconda3/bin/python
```
Then, you only keep the parts before `bin/python` and you set it to the variable `CONDA_DIR` in the `scripts/env.sh` file. You can follow the same steps, by activating the desired working environment and get the path to it using the `which` command without the `bin/python` part. This value is the stored to the variable `CONDA_ENV`.


The file `scripts/run_rllib_slurm.sh` can be configured to your needs in the file heading. The settings that needs to be configure are:  
* The partition name on line 2.
* Number of needed nodes on line 3.
* The number of cpus per node on line 6, and
* The time limit for excution on line 7

When using SLURM, the `outputs` folder needs to be created:  
```bash
mkdir outputs
```

### Compilation commands
The compilation command may change depending on the c++ installation and the machine. To alter the compilation command, modify the variables `compile_tiramisu_cmd` and `compile_wrapper_cmd` in the file `utils/rl_autoscheduler_config.py`. Examples of compilation commands:
```bash

# Command for compiling the Tiramisu autoschedule code when -lz is supported
c++ -I${TIRAMISU_ROOT}/3rdParty/Halide/include -I${TIRAMISU_ROOT}/include -I${TIRAMISU_ROOT}/3rdParty/isl/include  -Wl,--no-as-needed -ldl -g -fno-rtti -lz -lpthread -std=c++11 -O0 -o ${FILE_PATH}.o -c ${FILE_PATH};\
c++ -Wl,--no-as-needed -ldl -g -fno-rtti -lz -lpthread -std=c++11 -O0 ${FILE_PATH}.o -o ./${FILE_PATH}.out   -L${TIRAMISU_ROOT}/build  -L${TIRAMISU_ROOT}/3rdParty/Halide/lib  -L${TIRAMISU_ROOT}/3rdParty/isl/build/lib  -Wl,-rpath,${TIRAMISU_ROOT}/build:${TIRAMISU_ROOT}/3rdParty/Halide/lib:${TIRAMISU_ROOT}/3rdParty/isl/build/lib -ltiramisu -ltiramisu_auto_scheduler -lHalide -lisl 

# Command for compiling the Tiramisu autoschedule code when -lz is not supported
${CXX} -I${TIRAMISU_ROOT}/3rdParty/Halide/include -I${TIRAMISU_ROOT}/include -I${TIRAMISU_ROOT}/3rdParty/isl/include  -Wl,--no-as-needed -ldl -g -fno-rtti   -lpthread -std=c++11 -O0 -o ${FILE_PATH}.o -c ${FILE_PATH};\
${CXX} -Wl,--no-as-needed -ldl -g -fno-rtti   -lpthread -std=c++11 -O0 ${FILE_PATH}.o -o ./${FILE_PATH}.out   -L${TIRAMISU_ROOT}/build  -L${TIRAMISU_ROOT}/3rdParty/Halide/lib  -L${TIRAMISU_ROOT}/3rdParty/isl/build/lib  -Wl,-rpath,${TIRAMISU_ROOT}/build:${TIRAMISU_ROOT}/3rdParty/Halide/lib:${TIRAMISU_ROOT}/3rdParty/isl/build/lib -ltiramisu -ltiramisu_auto_scheduler -lHalide -lisl


# Command for compiling the Tiramisu wrapper code when -lz is supported
g++ -shared -o ${FUNC_NAME}.o.so ${FUNC_NAME}.o;\
g++ -std=c++11 -fno-rtti -I${TIRAMISU_ROOT}/include -I${TIRAMISU_ROOT}/3rdParty/Halide/include -I${TIRAMISU_ROOT}/3rdParty/isl/include/ -I${TIRAMISU_ROOT}/benchmarks -L${TIRAMISU_ROOT}/build -L${TIRAMISU_ROOT}/3rdParty/Halide/lib/ -L${TIRAMISU_ROOT}/3rdParty/isl/build/lib -o ${FUNC_NAME}_wrapper -ltiramisu -lHalide -ldl -lpthread -lz -lm -Wl,-rpath,${TIRAMISU_ROOT}/build ./${FUNC_NAME}_wrapper.cpp ./${FUNC_NAME}.o.so -ltiramisu -lHalide -ldl -lpthread -lz -lm

# Command for compiling the Tiramisu wrapper code when -lz is not supported
${GXX} -shared -o ${FUNC_NAME}.o.so ${FUNC_NAME}.o;\
${CXX} -I${TIRAMISU_ROOT}/3rdParty/Halide/include -I${TIRAMISU_ROOT}/include -I${TIRAMISU_ROOT}/3rdParty/isl/include -Wl,--no-as-needed -ldl -g -fno-rtti -lpthread -std=c++11 -O3 -o ${FUNC_NAME}_wrapper ${FUNC_NAME}_wrapper.cpp ./${FUNC_NAME}.o.so -L${TIRAMISU_ROOT}/build  -L${TIRAMISU_ROOT}/3rdParty/Halide/lib  -L${TIRAMISU_ROOT}/3rdParty/isl/build/lib  -Wl,-rpath,${TIRAMISU_ROOT}/build:${TIRAMISU_ROOT}/3rdParty/Halide/lib:${TIRAMISU_ROOT}/3rdParty/isl/build/lib -ltiramisu -ltiramisu_auto_scheduler -lHalide -lisl


```



## Running the script
To run the script on one worker, you use the following command:
```bash
python train_ppo.py --num-workers 1

```
To run on slurm, you execute the following command:
```bash
sbatch scripts/run_rllib_slurm.sh

``` 

## Visualization
We use tensorboard for visualization, to visualize the experiements, use the following command:  
```bash
tensorboard dev upload --logdir ray_results/
```

## Logs
Logs are available under `outputs`. The filename job.[job_id].out (where [job_id] is the id attributed to the job by SLURM) contains all the execution logs, while the file job.[job_id].err contains all the errors.  