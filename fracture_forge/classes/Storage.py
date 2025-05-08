import sys, os, shutil, functools
from mpi4py import MPI

class Data:
    units_data = {
            "real" : {"timestep" : 1e-15, "energy": 6.9e-21, "length": 1e-10}, 
            "metal" : {"timestep" : 1e-12, "energy": 1.6e-19, "length": 1e-10},
                }
    boltzman = 0.00198716 #kcal/mol*kelvin
    verbosity = 0

class SystemParams:
    load_margin = 0
    simulation_temp = 300
    dr = 1
    error = 0.1
    default_units = "real"
    max_verbosity = 5
    default_verbosity = 1

class Helper:
    action_proc = 0
    @staticmethod
    def mpi_print(*args, verbosity = None):
        if MPI.COMM_WORLD.Get_rank() == Helper.action_proc:
            if verbosity is None:
                verbosity = SystemParams.default_verbosity
            if verbosity <= Data.verbosity:
                print(*args)
                sys.stdout.flush()
    """
    @staticmethod
    def mpi_print(*args):
        print(f"{MPI.COMM_WORLD.Get_rank()}: ", *args)
        sys.stdout.flush()
    """


    @staticmethod
    def print(*args):
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

    @staticmethod
    def action(command, *args, **kwargs):
        if MPI.COMM_WORLD.Get_rank() == Helper.action_proc:
            return command(*args, **kwargs)
        else:
            return None

    @staticmethod
    def linear_func(func):
        @functools.wraps(func)
        def decorator(*args, **kwargs):
            if MPI.COMM_WORLD.Get_rank() == 0:
                return func(*args, **kwargs)
            else:
                return None
        return decorator
