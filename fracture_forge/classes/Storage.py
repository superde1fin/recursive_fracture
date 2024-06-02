import sys, os, shutil
from mpi4py import MPI

class Data:
    units_data = {
                "real" : {"timestep" : 1e-15,}, 
                "metal" : {"timestep" : 1e-12},
                }
    non_inter_cutoff = 10

class SystemParams:
    simulation_temp = 300
    dr = 3
    error = 0.1
    interactions = [(1, 2), (1, 3), (1, 4), (1, 5), (2, 4), (3, 5), (4, 5)]
    dtheta = 30
    pivot_type = 2
    neigh_num = 2

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

