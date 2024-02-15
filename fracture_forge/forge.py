from lammps import PyLammps
import re, glob, copy, os
from classes.Storage import Data, SystemParams
import numpy as np
import time, random


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

def vizualization(lmp, thermo_step = 0.1, dump_step = 0.1, name = ""): #ns
    thermo_step = convert_timestep(lmp, thermo_step)
    dump_step = convert_timestep(lmp, dump_step)

    lmp.thermo(thermo_step)
    lmp.thermo_style("custom step temp etotal pe vol density pxx pyy pzz")
    lmp.thermo_modify("flush yes")

    #Computes
    lmp.compute("stress_pa all stress/atom NULL")
    lmp.compute("pe_pa all pe/atom")

    lmp.dump("stress_dump all custom", dump_step, f"./dumps/dump_{name}.*.dump id type x y z c_stress_pa[*] c_pe_pa")


def copy_lmp(lmp):
    os.system("touch log.lammps && pwd > log.lammps")
    """
    new_lmp = PyLammps(verbose = False)
    system_parameters_initialization(new_lmp, units = "real")
    new_lmp.timestep(1)
    filename = glob.glob("glass_*.structure")[-1]
    name_handle = re.search(r"(?<=glass_).+(?=\.structure)", filename).group()
    new_lmp.include(f"pot_{name_handle}.FF")
    vizualization(new_lmp, name = name_handle)

    new_lmp.region("my_simbox block", lmp.system.xlo, lmp.system.xhi, lmp.system.ylo, lmp.system.yhi, lmp.system.zlo, lmp.system.zhi)
    new_lmp.create_box(lmp.system.ntypes, "my_simbox")
    
    for atom in lmp.atoms:
        new_lmp.create_atoms(atom.type, "single", " ".join(map(str, atom.position)))

    return new_lmp
    """
    return lmp


def quazi_static(lmp, dr_frac = 0.01, dtheta = 1):
    print("start")
    start_coords = (lmp.eval("lx")/2 + lmp.system.xlo, lmp.system.ylo)
    starting_dir = "initial"
    if not os.path.isdir(starting_dir):
        os.system(f"mkdir {starting_dir}")
    else:
        os.system(f"rm -r {starting_dir}/*")
    os.chdir("initial")
#    new_lmp = QSR(lmp = lmp, coords = start_coords, dr = dr_frac*max(lmp.eval("lx"), lmp.eval("ly"), lmp.eval("lz")), dtheta = dtheta, norm = (1, 0), theta = None, first = True)
    new_lmp = QSR(lmp = float("infinity"), coords = start_coords, dr = dr_frac*max(lmp.eval("lx"), lmp.eval("ly"), lmp.eval("lz")), dtheta = dtheta, norm = (1, 0), theta = None, first = True)

    return (lmp.eval("pe") - new_lmp.eval("pe"))/new_lmp.variables["surface_area"].value


def QSR(lmp, coords, dr, dtheta, norm, theta, first = False):
    if theta:
        lmp = copy_lmp(lmp)
#        lmp.variable(f"surface_area equals {dr*lmp.eval('lz')}")
    print(coords)
    os.system("pwd")

    if (coords[0] > 43 or coords[0] < 3.4 or coords[1] > 43 or coords[1] < 3.4) and not first:
        print("hit border")
        return random.randint(20, 100)

    """
    if coords[0] > lmp.system.xhi or coords[0] < lmp.system.xlo or coords[1] > system.yhi or coords[1] < lmp.system.ylo:
        return lmp
    """

#    lmp.variable(f"surface_area equals {lmp.eval('surface_area') + dr*lmp.eval('lz')}")

    """
    lmp.fix(f"probe all npt temp 300 300 {lmp.eval('dt')*1000} iso 1 1 {lmp.eval('dt')*1000}")
    lmp.region(f"probe_region {coords[0]} {coords[1]} 0.0 {norm[0]} {norm[1]} 0")
    #Turn off interactions manually
    lmp.fix(f"probe_wall all wall/region probe_region lj126 1.0 1.0 2.5")
    probe_time = convert_timestep(lmp, 0.05)
    lmp.run(probe_time)
    lmp.unfix("probe")
    """


    lowest_pot = float("infinity")
    res_lmp = lmp
    for theta in np.linspace(0, np.pi, int(180/dtheta) + 1):
        print(str(theta))
        os.system(f"mkdir {theta}")
        os.chdir(f"{theta}")
        tmp_lmp =  QSR(lmp, coords = (coords[0] + dr*np.cos(theta), coords[1] + dr*np.sin(theta)), dr = dr, dtheta = dtheta, norm = (np.cos(theta - np.pi/2), np.sin(theta - np.pi/2)), theta = theta)
        os.chdir("..")
#        new_pe = tmp_lmp.eval("pe")
        new_pe = tmp_lmp
        print(f"lowest pe: {new_pe}")
        if new_pe < lowest_pot:
            os.system(f"cp {theta}/log.lammps .")
            selected_angle = theta
            lowest_pot = new_pe
#            res_lmp.finalize()
#            res_lmp.close()
            res_lmp = tmp_lmp
        else:
            pass
#            tmp_lmp.finalize()
#            tmp_lmp.close()
        os.system(f"rm -r {theta}")

    """
    for log_name in glob.glob(f"log_{lvl_ctr + 1}_*.lammps"):
        if log_name != f"log_{lvl_ctr + 1}_{best_step}.lammps":
            os.system(f"rm {log_name}")
    """
    return res_lmp


def main():
    lmp = PyLammps(verbose = False)

    system_parameters_initialization(lmp, units = "real")


    filename = glob.glob("glass_*.structure")[-1]
    lmp.read_data(filename)

    lmp.timestep(1)

    name_handle = re.search(r"(?<=glass_).+(?=\.structure)", filename).group()

    lmp.include(f"pot_{name_handle}.FF")

    vizualization(lmp, name = name_handle)

    """
    simulation_temp = 300 #K
    #Minimization
    lmp.reset_atoms("id")
    lmp.velocity(f"all create {simulation_temp} 12345 dist gaussian")
    lmp.minimize(f"1.0e-8 1.0e-8 {convert_timestep(lmp, 0.01)} {convert_timestep(lmp, 0.1)}")

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
