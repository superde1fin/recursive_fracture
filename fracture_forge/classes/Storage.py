import sys, os, shutil
from mpi4py import MPI

class Data:
    units_data = {
                "real" : {"timestep" : 1e-15,}, 
                "metal" : {"timestep" : 1e-12},
                }
    non_inter_cutoff = 10
    initial_types = 2
    type_groups = 5

class SystemParams:
    parameters = {
            "simulation_temp" : 300
            }

class Lmpfunc:
    def __init__(self, func):
        self._func = func

    def __call__(self, *args, **kwargs):
        MPI.COMM_WORLD.Barrier()
        return self._func(*args, **kwargs)

class Helper:
    action_proc = 0
    @staticmethod
    def print(*args):
        if MPI.COMM_WORLD.Get_rank() == Helper.action_proc:
            print(*args)
            sys.stdout.flush()

    @staticmethod
    def command(*args):
        if MPI.COMM_WORLD.Get_rank() == Helper.action_proc:
            os.system(*args)

    @staticmethod
    def chdir(*args):
        if MPI.COMM_WORLD.Get_rank() == Helper.action_proc:
            os.chdir(*args)

    @staticmethod
    def mkdir(*args):
        if MPI.COMM_WORLD.Get_rank() == Helper.action_proc:
            os.mkdir(*args)

    @staticmethod
    def rmtree(*args, **kwargs):
        if MPI.COMM_WORLD.Get_rank() == Helper.action_proc:
            shutil.rmtree(*args, **kwargs)

    @staticmethod
    def convert_timestep(lmp, step): #ns  - step
        return int((step*1e-9)/(lmp.eval("dt")*Data.units_data[SystemParams.parameters["units"]]["timestep"]))

    output_ctr = 0

class Fake_lmp:
    def __init__(self, energy):
        self.energy = energy

    def eval(self, handle):
        if handle == "pe":
            return self.energy
        else:
            return float("inf")

    def __bool__(self):
        return False

    def close(self):
        pass

