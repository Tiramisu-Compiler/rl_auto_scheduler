from dataclasses import dataclass
import numpy as np
import re
import sys, os, subprocess
from pathlib import Path
from datetime import datetime
import re
import torch

from tiramisu_programs.schedule_utils import TimeOutException

class CPP_File(object):

    @classmethod
    def compile_and_run_tiramisu_code(cls, config,file_path, log_message="No message"):
        # print("inside compile and run")
        os.environ["FUNC_DIR"] = (
            "/".join(Path(file_path).parts[:-1]) if len(Path(file_path).parts) > 1 else "."
        ) + "/"
        os.environ["FILE_PATH"] = file_path

        failed = cls.launch_cmd(config.tiramisu.compile_tiramisu_cmd, file_path)
        if failed:
            print(f"Error occured while compiling {file_path}")
            with open(file_path) as file:
                print(file.read(), file=sys.stderr, flush=True)
            return False
        else:
            failed = cls.launch_cmd(config.tiramisu.run_tiramisu_cmd, file_path)
            if failed:
                print(f"Error occured while running {file_path}")
                # with open(file_path) as file: print(file.read(), file=sys.stderr,flush=True)
                return False
        return True

    @classmethod
    def launch_cmd(cls,step_cmd, file_path, cmd_type=None, nb_executions=None, initial_exec_time=None):
        failed = False
        try:
            if cmd_type == "initial_exec":
                out = subprocess.run(
                    step_cmd,
                    check=True,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=15 * nb_executions,
                )
                # print("after running initial exec")
            elif cmd_type == "sched_eval":
                out = subprocess.run(
                    step_cmd,
                    check=True,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=15 + 10 * nb_executions * initial_exec_time / 1000,
                )
                # print("after running sched eval")

            else:
                out = subprocess.run(
                    step_cmd,
                    check=True,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )

        except subprocess.TimeoutExpired:
            raise TimeOutException

        except Exception as e:
            print(
                f"\n# {str(datetime.now())} ---> Error running {step_cmd} \n"
                + e.stderr.decode("UTF-8"),
                file=sys.stderr,
                flush=True,
            )
            out = e
            failed = True
        else:  # no exception rised
            if "error" in out.stderr.decode("UTF-8"):
                print(
                    f"\n# {str(datetime.now())} ---> Error running {step_cmd} \n"
                    + out.stderr.decode("UTF-8"),
                    file=sys.stderr,
                    flush=True,
                )
                failed = True
        if failed:
            func_folder = (
                "/".join(Path(file_path).parts[:-1])
                if len(Path(file_path).parts) > 1
                else "."
            ) + "/"
            with open(func_folder + "error.txt", "a") as f:
                f.write(
                    "\nError running "
                    + step_cmd
                    + "\n---------------------------\n"
                    + out.stderr.decode("UTF-8")
                    + "\n"
                )
        return failed

    @classmethod
    def get_comp_name(cls,file):
        f = open(file, "r+")
        comp_name = ""
        for l in f.readlines():
            if not "//" in l:
                l = l.split()
                if "computation" in l or "tiramisu::computation" in l:
                    l = l[1].split("(")
                    comp_name = l[0]
                    break
        f.close()
        return comp_name

    @classmethod
    def get_cpp_file(cls,Dataset_path, func_name):

        file_name = func_name + "_generator.cpp"
        original_path = Dataset_path + "/" + func_name + "/" + file_name
        dc_path = Path(Dataset_path).parts[:-1]
        target_path = "{}/Dataset_copies/{}".format(".", func_name)

        if not os.path.isdir("./Dataset_copies/"):
            os.mkdir("./Dataset_copies/")
            
        if os.path.isdir(target_path):
            os.system("rm -r {}".format(target_path))
            # print("directory removed")

        os.mkdir(target_path)
        os.system("cp -r {} {}".format(original_path, target_path))
        return target_path + "/" + file_name

