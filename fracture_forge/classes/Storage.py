import sys, os, shutil
from mpi4py import MPI

class Data:
    units_data = {
                "real" : {"timestep" : 1e-15,}, 
                "metal" : {"timestep" : 1e-12},
                }
    non_inter_cutoff = 10
    initial_types = 2

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

    output_ctr = 0

