# np.set_printoptions(threshold=sys.maxsize)
import copy
import random
import sys
import time
import traceback

import gym
import numpy as np
import ray
import tiramisu_programs

import rl_interface
from utils.environment_variables import configure_env_variables

np.seterr(invalid="raise")


class TiramisuScheduleEnvironment(gym.Env):
    '''
    The reinforcement learning environment used by the GYM. 
    '''
    SAVING_FREQUENCY = 500

    def __init__(self, config, shared_variable_actor):
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
        self.dataset_path = config.environment.dataset_path
        self.depth = 0
        self.nb_executions = 5
        self.episode_total_time = 0
        self.prog_ind = 0
        self.steps = 0
        self.previous_cpp_file = None
        self.shared_variable_actor = shared_variable_actor

        print("Loading data from {} \n".format(
            config.environment.dataset_path))    # FIX that here

        self.id = ray.get(self.shared_variable_actor.increment.remote())

        # List of function names
        self.progs_list = ray.get(
            self.shared_variable_actor.get_progs_list.remote(self.id))

        # Dict of programs with their annotations, schedules, exectution times and traces
        self.progs_dict = ray.get(
            self.shared_variable_actor.get_progs_dict.remote())
        print("Loaded the dataset!")

        # Dict of function with a Dict containing schedules in STR format and their execution time
        self.scheds = tiramisu_programs.schedule_utils.ScheduleUtils.get_schedules_str(
            list(self.progs_dict.keys()),
            self.progs_dict)  # to use it to get the execution time

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
                    tiramisu_programs.cpp_file.CPP_File.clean_cpp_file(
                        self.dataset_path, self.previous_cpp_file)
                # Choose a random program (function)
                random_prog_index = random.randint(0, len(self.progs_list) - 1)

                # Copy the function's files to the dataset copy created
                file = tiramisu_programs.cpp_file.CPP_File.get_cpp_file(
                    self.dataset_path, self.progs_list[random_prog_index])

                # Set up the function files to be deleted on the next iteration
                self.previous_cpp_file = self.progs_list[random_prog_index]

                # Load the tiramisu program from the file
                self.prog = tiramisu_programs.tiramisu_program.TiramisuProgram(
                    self.config, file)

                print(f"Trying with program {self.prog.name}")

                self.schedule_object = tiramisu_programs.schedule.Schedule(self.prog)

                self.schedule_controller = tiramisu_programs.schedule_controller.ScheduleController(
                    schedule=self.schedule_object,
                    nb_executions=self.nb_executions,
                    scheds=self.scheds,
                    config=self.config)

                # Load the legality check list. Starts empty
                lc_data = ray.get(self.shared_variable_actor.get_lc_data.remote())

                self.schedule_controller.load_legality_data(lc_data)

                # Get the gym representation from the annotations
                self.obs = self.schedule_object.get_representation()

                if self.config.tiramisu.env_type == "cpu":
                    if self.progs_dict == {} or self.prog.name not in self.progs_dict.keys():
                        print("Getting the initial exe time by execution")
                        self.prog.initial_execution_time = self.schedule_controller.measurement_env(
                            [], 'initial_exec', self.nb_executions,
                            self.prog.initial_execution_time)
                        self.progs_dict[self.prog.name] = {}
                        self.progs_dict[self.prog.name]["initial_execution_time"] = self.prog.initial_execution_time

                    else:
                        print("The initial execution time exists")
                        self.prog.initial_execution_time = self.progs_dict[
                            self.prog.name]["initial_execution_time"]
                else:
                    self.prog.initial_execution_time = 1.0
                    self.progs_dict[self.prog.name] = {}
                    self.progs_dict[self.prog.name]["initial_execution_time"] = self.prog.initial_execution_time
                self.progs_dict[self.prog.name]["program_annotation"] = self.schedule_object.annotations

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
        action_name = rl_interface.Action.ACTIONS_ARRAY[raw_action]
        print("\n ----> {} [ {} ] \n".format(
            action_name, self.schedule_object.schedule_str))
        info = {}
        applied_exception = False
        reward = 0.0
        speedup = 1.0
        self.steps += 1
        self.total_steps += 1

        try:
            action = rl_interface.Action(raw_action,
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
            ray.get(self.shared_variable_actor.update_lc_data.remote(
                self.schedule_controller.get_legality_data()))
            if "schedule" in self.progs_dict[self.prog.name]:
                self.schedule_object.schedule_dict["speedup"] = speedup
                self.schedule_object.schedule_dict["schedule_str"] = self.schedule_object.schedule_str
                self.progs_dict[self.prog.name]["schedules_list"].append(
                    self.schedule_object.schedule_dict)
            else:
                self.schedule_object.schedule_dict["speedup"] = speedup
                self.schedule_object.schedule_dict["schedule_str"] = self.schedule_object.schedule_str
                self.progs_dict[self.prog.name]["schedules_list"] = [
                    self.schedule_object.schedule_dict]
        reward_object = rl_interface.Reward(speedup)
        reward = reward_object.reward
        print(f"Received a reward: {reward}")

        # Saving data
        if self.total_steps % self.SAVING_FREQUENCY:
            ray.get(self.shared_variable_actor.write_lc_data.remote())
            rl_interface.utils.EnvironmentUtils.write_json_dataset(
                f"worker_{self.id}.json", self.progs_dict)
        return self.obs, reward, done, info
