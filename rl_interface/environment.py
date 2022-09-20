# np.set_printoptions(threshold=sys.maxsize)
import json
import random
import sys

import gym
import numpy as np
import ray

# from pyfiglet import Figlet
from rl_interface.action import Action
from tiramisu_programs.optimization import optimization_command
from tiramisu_programs.schedule import Schedule, ScheduleUtils
from tiramisu_programs.tiramisu_program import Tiramisu_Program

np.seterr(invalid="raise")
import copy
import math
import os
import time
import traceback

import torch

from tiramisu_programs.surrogate_model_utils.modeling import Model_Recursive_LSTM_v2
from tiramisu_programs.cpp_file import CPP_File



class SearchSpaceSparseEnhancedMult(gym.Env):
    def __init__(
        self,
        programs_file,
        dataset_path,
        shared_variable_actor,
        pretrained_weights_path,
        **args
    ):

        # f = Figlet(font='banner3-D')
        # # print(f.renderText("Tiramisu"))
        print("Initializing the environment")

        self.placeholders = []
        self.speedup = 0
        self.schedule = []
        self.tiramisu_progs = []
        self.progs_annot = {}
        self.programs_file = programs_file
        self.args = args
        self.measurement_env = None

        print("Récupération des données depuis {} \n".format(dataset_path))
        self.shared_variable_actor = shared_variable_actor
        self.id = ray.get(self.shared_variable_actor.increment.remote())
        self.progs_list = ray.get(
            self.shared_variable_actor.get_progs_list.remote(self.id)
        )
        self.progs_dict = ray.get(self.shared_variable_actor.get_progs_dict.remote())
        print("Loaded the dataset!")

        self.scheds = ScheduleUtils.get_schedules_str(
            list(self.progs_dict.keys()), self.progs_dict
        )  # to use it to get the execution time

        self.action_space = gym.spaces.Discrete(62)
        self.observation_space = gym.spaces.Dict(
            {
                "representation": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(5, 1052)
                ),
                "action_mask": gym.spaces.Box(low=0, high=1, shape=(62,)),
                "loops_representation": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(15, 26)
                ),
                "child_list": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(12, 11)),
                "has_comps": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(12,)),
                "computations_indices": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(12, 5)
                ),
            }
        )

        self.dataset_path = dataset_path
        self.depth = 0
        self.nb_executions = 5
        self.episode_total_time = 0
        # self.lc_total_time = 0
        # self.codegen_total_time = 0
        self.prog_ind = 0
        self.model = Model_Recursive_LSTM_v2()
        self.model.load_state_dict(
            torch.load(pretrained_weights_path, map_location="cpu")
        )
        

    def reset(self, file=None):
        print("\n----------Resetting the environment-----------\n")
        self.episode_total_time = time.time()
        # self.lc_total_time = 0
        # self.codegen_total_time = 0
        while True:
            try:
                init_indc = random.randint(0, len(self.progs_list) - 1)
                file = CPP_File.get_cpp_file(self.dataset_path, self.progs_list[init_indc])
                self.prog = Tiramisu_Program(file)
                self.schedule_object = Schedule(self.prog, self.model, nb_executions = self.nb_executions, scheds=self.scheds,**self.args)
                self.obs = self.schedule_object.get_observation()
                if self.args["env_type"] == "cpu":
                    if self.progs_dict == {} or self.prog.name not in self.progs_dict.keys():
                        print("Getting the intitial exe time by execution")
                        start_time=time.time()
                        self.prog.initial_execution_time=self.schedule_object.measurement_env([],'initial_exec', self.nb_executions)
                        cg_time=time.time()-start_time 
                        #print("After getting initial exec time:",cg_time, "initial exec time is :", self.prog.initial_execution_time)
                        # self.codegen_total_time +=cg_time
                        self.progs_dict[self.prog.name]={}
                        self.progs_dict[self.prog.name]["initial_execution_time"]=self.prog.initial_execution_time

                    else:
                        print("The initial execution time exists")
                        self.prog.initial_execution_time=self.progs_dict[self.prog.name]["initial_execution_time"]
                else:
                    self.prog.initial_execution_time = 1.0
                    self.progs_dict[self.prog.name]={}
                    self.progs_dict[self.prog.name]["initial_execution_time"]=self.prog.initial_execution_time

            except:
                print("RESET_ERROR", traceback.format_exc())
                continue

            self.steps = 0
            self.new_scheds = {}
            self.search_time = time.time()
            return self.obs

    def step(self, raw_action):
        # print("in step function")
        action_name = Action.ACTIONS_ARRAY[raw_action]
        print("\nThe current schedule is: ", len(self.schedule), self.schedule_object.schedule_str)
        print("The action {} has been chosen\n".format(action_name))
        exit = False
        done = False
        info = {}
        applied_exception = False
        skew_params_exception = False
        skew_unroll = False
        reward = 0
        self.steps += 1

        try:
            action = Action(raw_action, self.schedule_object.it_dict, self.schedule_object.common_it)
            # print("after creating the action")
            self.obs = copy.deepcopy(
                self.schedule_object.get_observation()
            )  # get current observation
            self.obs, reward, done, info = self.schedule_object.apply_action(action) # Should return speedup instead of reward
        except Exception as e:
            print(e.__class__.__name__)
            if applied_exception:
                # reward = -1
                print("Already Applied exception")
                info = {"more than one time": True}
                done = False
                return self.obs, reward, done, info

            else:
                print("ERROR_STEP", traceback.format_exc())
                ex_type, ex_value, ex_traceback = sys.exc_info()
                # if (
                #     self.schedule != []
                #     and not skew_params_exception
                #     and not skew_unroll
                # ):
                #     self.schedule.pop()
                # reward = -1
                print("This action yields an error. It won't be applied.")
                print("else exception", ex_type.__name__, ex_value, ex_traceback)
                done = False
                info = {
                    "depth": self.depth,
                    "error": "ended with error in the step function",
                }
        # try:
        #     self.save_sched_to_dataset()
        #     self.write_data()
        #     writing_time=time.time()-start_time
        #     print("Data saved in ",writing_time)
        # except:
        #     print(f"failed to save schedule", traceback.format_exc() , file=sys.stderr, flush=True)
        return self.obs, reward, done, info

   
    def save_sched_to_dataset(self):
        for func in self.new_scheds.keys():
            for schedule_str in self.schedule_object.new_scheds[func].keys():#schedule_str represents the key, for example: 'Interchange Unrolling Tiling', the value is a tuple(schedule,execution_time)

                schedule=self.new_scheds[func][schedule_str][0]#here we get the self.obs["schedule"] containing the omtimizations list
                exec_time=self.new_scheds[func][schedule_str][1]
                search_time=self.new_scheds[func][schedule_str][2]
                for comp in self.comps:

                    #Initialize an empty dict
                    sched_dict={
                    comp: {
                    "schedule_str":schedule_str,
                    "search_time":search_time,
                    "interchange_dims": [],
                    "tiling": {
                        "tiling_depth":None,
                        "tiling_dims":[],
                        "tiling_factors":[]
                    },
                    "unrolling_factor": None,
                    "parallelized_dim": None,
                    "reversed_dim": None,
                    "skewing": {    
                        "skewed_dims": [],
                        "skewing_factors": [],
                        "average_skewed_extents": [],
                        "transformed_accesses": []
                                },
                    "unfuse_iterators": [],
                    "tree_structure": {},
                    "execution_times": []}
                    }

                    for optim in schedule:
                        if optim.type == 'Interchange':
                            sched_dict[comp]["interchange_dims"]=[self.schedule_object.it_dict[comp][optim.params_list[0]]['iterator'], self.schedule_object.it_dict[comp][optim.params_list[1]]['iterator']]

                        elif optim.type == 'Skewing':
                            first_dim_index=self.schedule_object.it_dict[comp][optim.params_list[0]]['iterator']
                            second_dim_index= self.schedule_object.it_dict[comp][optim.params_list[1]]['iterator']
                            first_factor=optim.params_list[2]
                            second_factor=optim.params_list[3]

                            sched_dict[comp]["skewing"]["skewed_dims"]=[first_dim_index,second_dim_index]
                            sched_dict[comp]["skewing"]["skewing_factors"]=[first_factor,second_factor]

                        elif optim.type == 'Parallelization':
                            sched_dict[comp]["parallelized_dim"]=self.schedule_object.it_dict[comp][optim.params_list[0]]['iterator']

                        elif optim.type == 'Tiling':
                            #Tiling 2D
                            if len(optim.params_list)==4:
                                sched_dict[comp]["tiling"]["tiling_depth"]=2
                                sched_dict[comp]["tiling"]["tiling_dims"]=[self.schedule_object.it_dict[comp][optim.params_list[0]]['iterator'],self.schedule_object.it_dict[comp][optim.params_list[1]]['iterator']]
                                sched_dict[comp]["tiling"]["tiling_factors"]=[optim.params_list[2],optim.params_list[3]]

                            #Tiling 3D
                            elif len(optim.params_list)==6:
                                sched_dict[comp]["tiling"]["tiling_depth"]=3
                                sched_dict[comp]["tiling"]["tiling_dims"]=[self.schedule_object.it_dict[comp][optim.params_list[0]]['iterator'],self.schedule_object.it_dict[comp][optim.params_list[1]]['iterator'],self.schedule_object.it_dict[comp][optim.params_list[2]]['iterator']]
                                sched_dict[comp]["tiling"]["tiling_factors"]=[optim.params_list[3],optim.params_list[4],optim.params_list[5]]

                        elif optim.type == 'Unrolling':
                            sched_dict[comp]["unrolling_factor"]=optim.params_list[comp][1]

                        elif optim.type == 'Reversal':
                            sched_dict[comp]["reversed_dim"]=self.schedule_object.it_dict[comp][optim.params_list[0]]['iterator']
                        
                        elif optim.type == 'Fusion':
                            pass

                sched_dict[comp]["execution_times"].append(exec_time)
                if not "schedules_list" in self.progs_dict[func].keys():
                    self.progs_dict[func]["schedules_list"]=[sched_dict]
                else:
                    self.progs_dict[func]["schedules_list"].append(sched_dict)
            
    def write_data(self):
        # print("in write data")
        with open(self.programs_file, 'w') as f:
            json.dump(self.progs_dict, f)
        # print("done writing data")
        f.close()