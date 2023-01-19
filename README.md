# RL for performance optimization

## Generate the dataset
To generate the dataset, you use the script `utils/program_generator/tiramisu_maker.py`. Here are the options to pass the script and their default values:
```
  -h, --help            show this help message and exit
  --output-path OUTPUT_PATH [Default = "Dataset_multi"]
                        The location where to save the dataset.
  --first-seed FIRST_SEED  [Default = 105100]
                        The first seed to start generating the programs from. Every seed produces a program,and two different seeds produce two different programs.
  --nb-programs NB_PROGRAMS  [Default = 20000]
                        The number of programs to generate.

```


## Configuring the repository
To configure the repository first copy the template files into the same location without the `.template` extension, like follows:
```bash  
cp config.yaml.template config.yaml
cp scripts/env.sh.template scripts/env.sh
```
 Then, change the fields to their approriate value based on the environment the code is running on. The most important fields in the `config.yml` file, that the code will not run without setting are:
 * `ray.base_path`:  The absolute path to where the code is located on disk. 
* `environment.dataset_path`: The path to the training dataset. 
* `tiramisu.tiramisu_path`: The absolute path to the complete tiramisu installation, with the autoscheduler also installed.
* `tiramisu.model_checkpoint`: The path to the surrogate model used for prediction. A model chackpoint is provided under `weights/surrogate_model`.

The other flags can also be set for a desired purpose. For instance, you can change the policy model properties. You can also control whether or not you want to keep the dataset files after using them for an episode. 

The file `scripts/env.sh` is used in order to train on SLURM. The most important fields in this file, that the code will not run without setting are:
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







## Compilation commands
Depending on the c++ installation and the machine, the compilation command may differ. To change the compilation command, change the variables `compile_tiramisu_cmd` and `compile_wrapper_cmd` in the file `utils/rl_autoscheduler_config.py`. Examples of compilation commands:
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
sbatch run_rllib_slurm.sh

```  