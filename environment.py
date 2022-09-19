# np.set_printoptions(threshold=sys.maxsize)
import json
import random
import sys

import gym
import numpy as np
import ray

# from pyfiglet import Figlet
from action import Action
from optimization import optimization_command
from schedule import Schedule
from tiramisu_program import Tiramisu_Program

np.seterr(invalid="raise")
import copy
import math
import os
import time
import traceback

import torch

from surrogate_model_utils.modeling import Model_Recursive_LSTM_v2
from cpp_file import CPP_File



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
        print("Initialisation de l'environnement")

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
        print("Dataset chargé!\n")

        self.scheds = Schedule.get_schedules_str(
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
        self.lc_total_time = 0
        self.codegen_total_time = 0
        self.prog_ind = 0
        self.model = Model_Recursive_LSTM_v2()
        self.model.load_state_dict(
            torch.load(pretrained_weights_path, map_location="cpu")
        )
        self.schedule_list_model = []

    def reset(self, file=None):
        print("\nRéinitialisation de l'environnement\n")
        self.episode_total_time = time.time()
        self.lc_total_time = 0
        self.codegen_total_time = 0
        while True:
            try:
                init_indc = random.randint(0, len(self.progs_list) - 1)
                file = CPP_File.get_cpp_file(self.dataset_path, self.progs_list[init_indc])
                self.prog = Tiramisu_Program(file)
                self.schedule_object = Schedule(self.prog, self.model, self.args)
                self.obs = self.schedule_object.get_observation()

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
        print("\nL'action {} est choisie".format(action_name))
        print("the curr schedule is: ", len(self.schedule), self.schedule_str)
        exit = False
        done = False
        info = {}
        applied_exception = False
        skew_params_exception = False
        skew_unroll = False
        reward = 0
        self.steps += 1
        first_comp = self.comps[0]

        try:
            action = Action(raw_action, self.it_dict, self.common_it)
            # print("after creating the action")
            self.obs = copy.deepcopy(
                self.schedule_object.get_observation()
            )  # get current observation
            self.obs, reward, done, info = self.schedule_object.apply_action(action)
        except Exception as e:
            print(e.__class__.__name__)
            if applied_exception:
                # reward = -1
                print("applied exception")
                info = {"more than one time": True}
                done = False
                return self.obs, reward, done, info

            else:
                print("ERROR_MODEL", traceback.format_exc())
                ex_type, ex_value, ex_traceback = sys.exc_info()
                if (
                    self.schedule != []
                    and not skew_params_exception
                    and not skew_unroll
                ):
                    self.schedule.pop()
                # reward = -1
                print("\nCette action a généré une erreur, elle ne sera pas appliquée.")
                print("else exception", ex_type.__name__, ex_value, ex_traceback)
                done = False
                info = {
                    "depth": self.depth,
                    "error": "ended with error in the step function",
                }
        return self.obs, reward, done, info
