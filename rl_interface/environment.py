# np.set_printoptions(threshold=sys.maxsize)
import json
import random
import sys

import gym
import numpy as np
import ray

# from pyfiglet import Figlet
from rl_interface.action import Action
from tiramisu_programs.schedule import Schedule
from tiramisu_programs.schedule_controller import ScheduleController
from tiramisu_programs.schedule_utils import ScheduleUtils
from tiramisu_programs.tiramisu_program import Tiramisu_Program

import copy
import time
import traceback

import torch

from tiramisu_programs.surrogate_model_utils.modeling import Model_Recursive_LSTM_v2
from tiramisu_programs.cpp_file import CPP_File
from rl_interface.reward import Reward

np.seterr(invalid="raise")


class TiramisuScheduleEnvironment(gym.Env):
    def __init__(
        self,
        config,
        shared_variable_actor
    ):

        # f = Figlet(font='banner3-D')
        # # print(f.renderText("Tiramisu"))
        print("Initializing the environment")

        self.config = config
        self.placeholders = []
        self.speedup = 0
        self.schedule = []
        self.tiramisu_progs = []
        self.progs_annot = {}
        self.programs_file = config.environment.programs_file
        self.measurement_env = None

        print("Récupération des données depuis {} \n".format(config.environment.dataset_path))
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

        self.dataset_path = config.environment.dataset_path
        self.depth = 0
        self.nb_executions = 5
        self.episode_total_time = 0
        self.prog_ind = 0
        
        

    def reset(self, file=None):
        print("\n----------Resetting the environment-----------\n")
        self.episode_total_time = time.time()
        while True:
            try:
                init_indc = random.randint(0, len(self.progs_list) - 1)
                file = CPP_File.get_cpp_file(self.dataset_path, self.progs_list[init_indc])
                self.prog = Tiramisu_Program(self.config,file)
                self.schedule_object = Schedule(self.prog)
                self.schedule_controller = ScheduleController(schedule=self.schedule_object, nb_executions = self.nb_executions, scheds=self.scheds, config=self.config)
                self.obs = self.schedule_object.get_representation()
                if self.config.tiramisu.env_type == "cpu":
                    if self.progs_dict == {} or self.prog.name not in self.progs_dict.keys():
                        print("Getting the intitial exe time by execution")
                        self.prog.initial_execution_time=self.schedule_controller.measurement_env([],'initial_exec', self.nb_executions, self.prog.initial_execution_time)
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
            self.search_time = time.time()
            return self.obs

    def step(self, raw_action):
        action_name = Action.ACTIONS_ARRAY[raw_action]
        print("\n ----> {} [ {} ] \n".format(action_name, self.schedule_object.schedule_str))
        info = {}
        applied_exception = False
        reward = 0
        self.steps += 1

        try:
            action = Action(raw_action, self.schedule_object.it_dict, self.schedule_object.common_it)
            self.obs = copy.deepcopy(
                self.schedule_object.get_representation()
            )
            self.obs, speedup, done, info = self.schedule_controller.apply_action(action) # Should return speedup instead of reward
            reward_object = Reward(speedup)
            reward = reward_object.log_reward()
            
        except Exception as e:
            print("STEP_ERROR: ", e.__class__.__name__, end=" ")
            if applied_exception:
                print("Already Applied exception")
                info = {"more than one time": True}
                done = False
                return self.obs, reward, done, info

            else:
                print("This action yields an error. It won't be applied.")
                done = False
                info = {
                    "depth": self.depth,
                    "error": "ended with error in the step function",
                }
        return self.obs, reward, done, info
