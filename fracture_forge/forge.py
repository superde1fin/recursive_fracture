from lammps import PyLammps
import re, glob, copy, os
from classes.Storage import Data, SystemParams
import numpy as np
import time, random

def copy_variables(new_lmp, lmp):
    new_lmp.variable(f"home_dir string {lmp.variables['home_dir'].value}")
    new_lmp.variable(f"surface_area equal {lmp.variables['surface_area'].value}")

def disable_interactions(lmp, coords, dr, theta):
    lmp.minimize(f"1.0e-8 1.0e-8 {convert_timestep(lmp, 0.01)} {convert_timestep(lmp, 0.1)}")
    return 0


def system_parameters_initialization(lmp, units = "metal"):
    lmp.units(units)
    SystemParams.parameters["units"] = units
    lmp.atom_style("charge")
    lmp.boundary("p p p")
    #lmp.comm_modify("model single vel yes")
    lmp.neighbor("2.0 bin")
    lmp.neigh_modify("every 1 delay 0")


def convert_timestep(lmp, step): #ns  - step
    return int((step*1e-9)/(lmp.eval("dt")*Data.units_data[SystemParams.parameters["units"]]["timestep"]))

def vizualization(lmp, thermo_step = 0.1, dump_step = 0.1): #ns
    thermo_step = convert_timestep(lmp, thermo_step)
    dump_step = convert_timestep(lmp, dump_step)

    lmp.thermo(thermo_step)
    lmp.thermo_style("custom step temp etotal pe vol density pxx pyy pzz")
    lmp.thermo_modify("flush yes")

    #Computes
    lmp.compute("stress_pa all stress/atom NULL")
    lmp.compute("pe_pa all pe/atom")



def copy_lmp(lmp, potfile):
    new_lmp = PyLammps(verbose = False)

    copy_variables(new_lmp, lmp)

    system_parameters_initialization(new_lmp, units = "real")
    new_lmp.timestep(1)

    new_lmp.region("my_simbox block", lmp.system.xlo, lmp.system.xhi, lmp.system.ylo, lmp.system.yhi, lmp.system.zlo, lmp.system.zhi)
    new_lmp.create_box(lmp.system.ntypes, "my_simbox")
    vizualization(new_lmp)
    new_lmp.include(potfile)
    
    for i in range(len(lmp.atoms)):
        position = " ".join(map(str, lmp.atoms[i].position))
        new_lmp.create_atoms(lmp.atoms[i].type, "single", position)

    return new_lmp


def quazi_static(lmp, dr_frac = 0.01, dtheta = 1):
    print("start")
    start_coords = (lmp.eval("lx")/2 + lmp.system.xlo, lmp.system.ylo)
    filename = glob.glob("glass_*.structure")[-1]
    name_handle = re.search(r"(?<=glass_).+(?=\.structure)", filename).group()
    potfile = os.path.abspath(f"pot_{name_handle}.FF")
    starting_dir = "initial"
    if not os.path.isdir(starting_dir):
        os.system(f"mkdir {starting_dir}")
    elif os.listdir(starting_dir):
        os.system(f"rm -r {starting_dir}/*")
    os.chdir(starting_dir)
    new_lmp = QSR(lmp = lmp, coords = start_coords, dr = dr_frac*max(lmp.eval("lx"), lmp.eval("ly"), lmp.eval("lz")), dtheta = dtheta, norm = (1, 0), theta = None, potfile = potfile)

    return (lmp.eval("pe") - new_lmp.eval("pe"))/new_lmp.variables["surface_area"].value


def QSR(lmp, coords, dr, dtheta, norm, theta, potfile):
    os.system(f"echo '{coords}' >> path.txt")
    if coords[0] > lmp.system.xhi or coords[0] < lmp.system.xlo or coords[1] > lmp.system.yhi or coords[1] < lmp.system.ylo:
        return lmp

    if not theta is None:
        lmp = copy_lmp(lmp, potfile)
        lmp.variable(f"surface_area equal {lmp.variables['surface_area'].value + dr*lmp.eval('lz')}")
    else:
        lmp.variable(f"surface_area equal {dr*lmp.eval('lz')}")

    #Output
    print("Lammps: ", lmp)
    print("Coordinates: ", coords)
    os.system("pwd")

    disable_interactions(lmp, coords, dr, theta)

    lowest_pot = float("infinity")
    res_lmp = None
    prev_theta = theta
    for theta in np.linspace(0, np.pi, int(180/dtheta) + 1):
        if prev_theta is None or theta - prev_theta != np.pi and theta - prev_theta != -np.pi:
            print("Theta: ", str(theta))
            os.system(f"mkdir {theta}")
            os.chdir(f"{theta}")
            tmp_lmp =  QSR(lmp, coords = (coords[0] + dr*np.cos(theta), coords[1] + dr*np.sin(theta)), dr = dr, dtheta = dtheta, norm = (np.cos(theta - np.pi/2), np.sin(theta - np.pi/2)), theta = theta, potfile = potfile)
            os.chdir("..")
            new_pe = tmp_lmp.eval("pe")
            print(f"lowest pe: {new_pe}")
            if new_pe < lowest_pot:
                #Note: Check combination of log files
                if os.path.isfile(f"{theta}/log.lammps"):
                    os.system(f"cp {theta}/log.lammps .")
                os.system(f"cat {theta}/path.txt > tmp_path.txt")
                selected_angle = theta
                lowest_pot = new_pe
                if res_lmp and res_lmp != lmp:
        #            res_lmp.finalize()
                    res_lmp.close()
                res_lmp = tmp_lmp
            elif tmp_lmp != lmp:
    #            tmp_lmp.finalize()
                tmp_lmp.close()
            os.system(f"rm -r {theta}")
    os.system("cat tmp_path.txt >> path.txt")
    return res_lmp


def main():
    lmp = PyLammps(verbose = False)

    system_parameters_initialization(lmp, units = "real")


    filename = glob.glob("glass_*.structure")[-1]
    lmp.read_data(filename)

    lmp.timestep(1)

    name_handle = re.search(r"(?<=glass_).+(?=\.structure)", filename).group()

    lmp.variable(f"home_dir string {os.path.abspath('.')}")
    lmp.include(f"pot_{name_handle}.FF")

    vizualization(lmp)

    simulation_temp = 300 #K
    #Minimization
    lmp.reset_atoms("id")
    lmp.velocity(f"all create {simulation_temp} 12345 dist gaussian")
    lmp.minimize(f"1.0e-8 1.0e-8 {convert_timestep(lmp, 0.01)} {convert_timestep(lmp, 0.1)}")



    """
    #Run
    initial_relax = convert_timestep(lmp, 0.1)
    lmp.fix(f"initial_relax all npt temp {simulation_temp} {simulation_temp} {100*lmp.eval('dt')} iso 1 1 {1000*lmp.eval('dt')}")
    lmp.run(initial_relax)
    lmp.unfix("initial_relax")
    """

#    quazi_static(lmp)
    quazi_static(lmp, 0.5, 90)

    return lmp

if __name__ == "__main__":
    main()#.finalize()
