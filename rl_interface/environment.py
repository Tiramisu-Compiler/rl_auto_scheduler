# np.set_printoptions(threshold=sys.maxsize)
import copy
import sys
import time
import traceback

import gym
import numpy as np
import ray
from rl_interface.action import Action
from rl_interface.reward import Reward
from tiramisu_programs.cpp_file import CPP_File
from tiramisu_programs.tiramisu_program import TiramisuProgram
from tiramisu_programs.schedule import Schedule
from tiramisu_programs.schedule_controller import ScheduleController
from utils.environment_variables import configure_env_variables
from utils.rl_autoscheduler_config import RLAutoSchedulerConfig

np.seterr(invalid="raise")


class TiramisuScheduleEnvironment(gym.Env):
    '''
    The reinforcement learning environment used by the GYM. 
    '''
    SAVING_FREQUENCY = 500

    def __init__(self, config: RLAutoSchedulerConfig, dataset_actor):
        print("Configuring the environment variables")
        configure_env_variables(config)

        print("Initializing the local variables")
        self.config = config
        self.total_steps = 0
        self.placeholders = []
        self.speedup = 0
        self.schedule = []
        self.tiramisu_progs = []
        self.progs_annot = {}
        self.programs_file = config.environment.programs_file
        self.measurement_env = None
        self.cpps_path = config.environment.dataset_path
        self.depth = 0
        self.nb_executions = 5
        self.episode_total_time = 0
        self.prog_ind = 0
        self.steps = 0
        self.previous_cpp_file = None
        self.dataset_actor = dataset_actor

        if config.environment.use_dataset:
            self.cpps_path = config.environment.json_dataset['cpps_path']

        self.action_space = gym.spaces.Discrete(62)

        self.observation_space = gym.spaces.Dict({
            # Computation representation (5 is the MAX computations)
            "representation":
            gym.spaces.Box(low=-np.inf, high=np.inf, shape=(5, 1052)),
            # Mask to hide actions from being taken 62 masks for 62 actions
            "action_mask":
            gym.spaces.Box(low=0, high=1, shape=(62, )),
            # Representation of loops
            "loops_representation":
            gym.spaces.Box(low=-np.inf, high=np.inf, shape=(15, 26)),
            # Loop indices of loops instead in loop i
            "child_list":
            gym.spaces.Box(low=-np.inf, high=np.inf, shape=(12, 11)),
            # Whether loop i has computations or not
            "has_comps":
            gym.spaces.Box(low=-np.inf, high=np.inf, shape=(12, )),
            # Computation indices of all computations inside of a loop (12 loops,5 max computations)
            "computations_indices":
            gym.spaces.Box(low=-np.inf, high=np.inf, shape=(12, 5)),
            # float representation of the padded string format of the program tree
            "prog_tree":
            gym.spaces.Box(low=-np.inf, high=np.inf, shape=(5000,))
        })

    def reset(self, file=None):
        """
        Reset the environment to the intial state. A state is defined as a random program with the schedule applied to it.
        The initial state is defined as a random program with no schedules applied to it.
        the input file is just a placeholder required by the gym.
        Returns: The current intitial state.
        """

        print("\n----------Resetting the environment-----------\n")
        self.episode_total_time = time.time()
        while True:
            try:
                # Clean files of the previous function ran
                if self.config.environment.clean_files and self.previous_cpp_file:
                    CPP_File.clean_cpp_file(self.previous_cpp_file)

                # get the next function
                (function_name, function_dict) = ray.get(
                    self.dataset_actor.get_next_function.remote())

                # Copy the function's files to the dataset copy created
                file = CPP_File.get_cpp_file(
                    self.cpps_path, function_name)

                # Set up the function files to be deleted on the next iteration
                self.previous_cpp_file = function_name

                # Load the tiramisu program from the file
                self.prog = TiramisuProgram(
                    self.config, file, function_dict)

                print(f"Trying with program {self.prog.name}")

                self.schedule_object = Schedule(
                    self.prog)

                self.schedule_controller = ScheduleController(
                    schedule=self.schedule_object,
                    nb_executions=self.nb_executions,
                    config=self.config)

                # Get the gym representation from the annotations
                self.obs = self.schedule_object.get_representation()

                if self.config.tiramisu.env_type == "cpu":
                    print("Getting the initial exe time by execution")
                    self.prog.initial_execution_time = self.schedule_controller.measurement_env(
                        [], 'initial_exec', self.nb_executions,
                        self.prog.initial_execution_time)
                elif self.config.tiramisu.env_type == "model":
                    self.prog.initial_execution_time = 1.0

            except:
                print("RESET_ERROR_STDERR",
                      traceback.format_exc(), file=sys.stderr)
                print("RESET_ERROR_STDOUT",
                      traceback.format_exc(), file=sys.stdout)
                continue

            self.steps = 0
            # unused
            self.search_time = time.time()
            print(f"Choosing program {self.prog.name}")
            return self.obs

    def step(self, raw_action):
        """
        Apply a transformation on a program. If the action raw_action is legal, it is applied. If not, it is ignored and not added to the schedule.
        Returns: The current state after eventually applying the transformation, and the reward that the agent received for taking the action.
        """
        action_name = Action.ACTIONS_ARRAY[raw_action]
        print("\n ----> {} [ {} ] \n".format(
            action_name, self.schedule_object.sched_str))
        info = {}
        applied_exception = False
        reward = 0.0
        speedup = 1.0
        self.steps += 1
        self.total_steps += 1

        try:
            action = Action(raw_action,
                            self.schedule_object.it_dict,
                            self.schedule_object.common_it)
            _, speedup, done, info = self.schedule_controller.apply_action(
                action)
            print("Obtained speedup: ", speedup)

        except Exception as e:
            self.schedule_object.repr["action_mask"][action.id] = 0
            print("STEP_ERROR_STDERR: ",
                  traceback.format_exc(),
                  file=sys.stderr,
                  end=" ")
            print("STEP_ERROR_STDOUT: ",
                  traceback.format_exc(),
                  file=sys.stdout,
                  end=" ")
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

        self.obs = copy.deepcopy(self.schedule_object.get_representation())
        if (self.schedule_controller.depth
                == self.schedule_object.MAX_DEPTH) or (self.steps >= 20):
            done = True
        if done:
            print("\n ************** End of an episode ************")
            try:
                speedup = self.schedule_controller.get_final_score()
            except:
                speedup = 1.0
            # Update dataset with explored legality checks
            self.dataset_actor.update_dataset.remote(
                self.prog.name, self.prog.function_dict)

        reward_object = Reward(speedup)
        reward = reward_object.reward
        print(f"Received a reward: {reward}")
        return self.obs, reward, done, info
