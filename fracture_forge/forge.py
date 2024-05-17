from mpi4py import MPI
from lammps import PyLammps
import ctypes as ct
import  glob, copy, os, sys
from classes.Storage import Data, SystemParams, Helper, Lmpfunc
import numpy as np
import time, random
import regex as re


def near_surface(coords, atom_pos, angles, dr, a, side):
    prev_theta, theta = angles
    x0, y0 = coords #Tip of the division vector

    #Tail of the division vector
    x1 = x0 - dr*np.cos(theta)
    y1 = y0 - dr*np.sin(theta)

    #Tail of the parallel transport of the division vector to the left by length 'a'
    x2 = x1 - a*np.sin(theta)
    y2 = y1 + a*np.cos(theta)

    #Tail of the parallel transport of the division vector to the right by length 'a'
    x3 = x1 + a*np.sin(theta)
    y3 = y1 - a*np.cos(theta)

    #Line through point (x0, y0) at angle theta
    f01 = np.poly1d([np.tan(theta), y0 - x0*np.tan(theta)])

    #Line through point (x0, y0) perpendicular to theta
    f02 = np.poly1d([np.tan(theta + np.pi/2), y0 - x0*np.tan(theta + np.pi/2)])

    #Line through point (x2, y2) at angle theta
    f21 = np.poly1d([np.tan(theta), y2 - x2*np.tan(theta)])

    #Line through point (x1, y1) perpendicular to theta
    f12 = np.poly1d([np.tan(theta + np.pi/2), y1 - x1*np.tan(theta + np.pi/2)])

    #Line through point (x3, y3) at angle theta
    f31 = np.poly1d([np.tan(theta), y3 - x3*np.tan(theta)])

    #Line through point (x0, y0) at angle perpendicular to prev_theta
    f99 = np.poly1d([np.tan(prev_theta + np.pi/2), y1 - x1*np.tan(prev_theta + np.pi/2)])

    for x in [atom_pos[0], side + atom_pos[0], atom_pos[0] - side]:
        if (np.sqrt((x - x1)**2 + (atom_pos[1] - y1)**2) < a) and (atom_pos[1] < np.polyval(f12, x)) and (atom_pos[1] > np.polyval(f99, x)):
            if (prev_theta > theta):
                return 1
            elif (prev_theta < theta):
                return -1
        if theta <= np.pi/2:
            if (atom_pos[1] <= np.polyval(f02, x) and atom_pos[1] <= np.polyval(f21, x) and atom_pos[1] >= np.polyval(f12, x) and atom_pos[1] >= np.polyval(f01, x)):
                return 1
            elif (atom_pos[1] <= np.polyval(f01, x) and atom_pos[1] <= np.polyval(f02, x) and atom_pos[1] >= np.polyval(f12, x) and atom_pos[1] >= np.polyval(f31, x)):
                return -1
        else:
            if (atom_pos[1] <= np.polyval(f02, x) and atom_pos[1] >= np.polyval(f21, x) and atom_pos[1] >= np.polyval(f12, x) and atom_pos[1] <= np.polyval(f01, x)):
                return 1
            elif (atom_pos[1] >= np.polyval(f01, x) and atom_pos[1] <= np.polyval(f02, x) and atom_pos[1] >= np.polyval(f12, x) and atom_pos[1] <= np.polyval(f31, x)):
                return -1
    return 0


def modify_potfile(lmp, potfile):
    ntypes = lmp.system.ntypes
    text = open(potfile, 'r').read()
    name = re.sub(r"(?<=\/[^/]+)\.(?=.+$)", "_new.", potfile)
    new_potfile = open(name, 'w')
    new_text = text + "\n\n#-------------------------\n\n"
    for t in range(1, ntypes + 1):
        mass_re = re.compile(f"^mass\s+{t}\s+.+$", re.MULTILINE)
        mass_line = mass_re.findall(text)[-1]
        new_text = mass_re.sub(mass_line + '\n' + re.sub(f"(?<=^mass\s+){t}(?=\s+.+$)", str(ntypes + t), mass_line) + '\n', new_text)
        mass_re = re.compile(f"^mass\s+{ntypes + t}\s+.+$", re.MULTILINE)
        mass_line = mass_re.findall(new_text)[-1]
        new_text = mass_re.sub(mass_line + '\n' + re.sub(f"(?<=^mass\s+){ntypes + t}(?=\s+.+$)", str(2*ntypes + t), mass_line) + '\n', new_text)

        
        coeff_lines = re.compile(f"(?:^pair_coeff\s+(?:(?:{t}\s+[0-9]+)|(?:[0-9]\s+{t}))\s+.+$)", re.MULTILINE).findall(text) 
        for line in coeff_lines:
            new_text += '\n' + re.sub(f"((?<=^pair_coeff\s+){t}(?=\s+[0-9]+.+$))|((?<=^pair_coeff\s+[0-9]\s+){t}(?=\s+.+$))", str(ntypes + t), line)
            new_text += '\n' + re.sub(f"((?<=^pair_coeff\s+){t}(?=\s+[0-9]+.+$))|((?<=^pair_coeff\s+[0-9]\s+){t}(?=\s+.+$))", str(2*ntypes + t), line)

        for j in range(t + 1, ntypes + 1):
            line = re.compile(f"^pair_coeff\s+{ntypes + t}\s+{j}\s+.+$", re.MULTILINE).findall(new_text)[-1]
            new_text += '\n' + re.sub(f"(?<=^pair_coeff\s+{ntypes + t}\s+){j}(?=\s+.+$)", str(ntypes + j), line)
            line = re.compile(f"^pair_coeff\s+{2*ntypes + t}\s+{j}\s+.+$", re.MULTILINE).findall(new_text)[-1]
            new_text += '\n' + re.sub(f"(?<=^pair_coeff\s+{2*ntypes + t}\s+){j}(?=\s+.+$)", str(2*ntypes + j), line)


        self_coeff_line = re.compile(f"^pair_coeff\s+{t}\s+{t}\s+.+$", re.MULTILINE).findall(text)[-1]
        new_text += '\n' + re.sub(f"(?<=^pair_coeff\s+{t}\s+){t}(?=\s+.+$)", str(ntypes + t), self_coeff_line)
        new_text += '\n' + re.sub(f"(?<=^pair_coeff\s+{t}\s+){t}(?=\s+.+$)", str(2*ntypes + t), self_coeff_line)


    general_type_re = re.compile(f"(?<=pair_coeff\s+\*\s+\*.+)(\s+\S+){{{ntypes}}}$", re.MULTILINE)
    type_names = general_type_re.search(new_text).group()*3
    new_text = general_type_re.sub(type_names, new_text)

    new_text += '\n\n'

    for i in range(1, ntypes + 1):
        for j in range(1, ntypes + 1):
            new_text += f"\npair_coeff {ntypes + i} {2*ntypes + j} table ${{table_path}} NoNo 10"
            

    


    new_potfile.write(new_text)

    return name
        
@Lmpfunc
def copy_variables(new_lmp, lmp):
    new_lmp.variable(f"home_dir string {lmp.variables['home_dir'].value}")
    new_lmp.variable(f"surface_area equal {lmp.variables['surface_area'].value}")


@Lmpfunc
def system_parameters_initialization(lmp, units = "metal"):
    lmp.units(units)
    SystemParams.parameters["units"] = units
    lmp.atom_style("charge")
    lmp.boundary("p p p")
    lmp.comm_modify("mode single vel yes")
    lmp.neighbor("2.0 bin")
    lmp.neigh_modify("every 1 delay 0")


def convert_timestep(lmp, step): #ns  - step
    return int((step*1e-9)/(lmp.eval("dt")*Data.units_data[SystemParams.parameters["units"]]["timestep"]))

@Lmpfunc
def vizualization(lmp, thermo_step = 0.1, dump_step = 0.1): #ns
    thermo_step = convert_timestep(lmp, thermo_step)
    dump_step = convert_timestep(lmp, dump_step)

    lmp.thermo(thermo_step)
    lmp.thermo_style("custom step temp etotal pe vol density pxx pyy pzz")
    lmp.thermo_modify("flush yes")

    #Computes
    lmp.compute("stress_pa all stress/atom NULL")
    lmp.compute("pe_pa all pe/atom")

def create_surface(lmp):
    min_y = float("inf")
    max_y = float("-inf")
    my_atoms = np.array(lmp.lmp.gather_atoms("x", 1, 3), dtype = ct.c_double).reshape((-1, 3))
    for atom in my_atoms:
        if atom[1] < min_y:
            min_y = atom[1]
        if atom[1] > max_y:
            max_y = atom[1]
    SystemParams.parameters["old_bounds"] = (min_y, max_y)
    Helper.print("Old bounds:", SystemParams.parameters["old_bounds"])
    #Testing
    """
    lmp.change_box(f"all y delta {-Data.non_inter_cutoff} {Data.non_inter_cutoff}")
    lmp.fix(f"surface_relax all npt temp {SystemParams.parameters['simulation_temp']} {SystemParams.parameters['simulation_temp']} {100*lmp.eval('dt')} iso 1 1 {1000*lmp.eval('dt')}")
    lmp.run(convert_timestep(lmp, 0.1))
    lmp.unfix("surface_relax")
    """



@Lmpfunc
def copy_lmp(lmp, potfile, angles, dr, coords):
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    Helper.print("starting copy")
    new_lmp = PyLammps(verbose = False)

    copy_variables(new_lmp, lmp)

    system_parameters_initialization(new_lmp, units = "real")
    new_lmp.timestep(1)

    new_lmp.region("my_simbox block", lmp.system.xlo, lmp.system.xhi, lmp.system.ylo, lmp.system.yhi, lmp.system.zlo, lmp.system.zhi)
    new_lmp.create_box(Data.initial_types*3, "my_simbox")
    vizualization(new_lmp)
    new_lmp.include(potfile)
    start_time = time.time()
    my_atoms = np.array(lmp.lmp.gather_atoms("x", 1, 3), dtype = ct.c_double).reshape((-1, 3))
    types = np.array(lmp.lmp.gather_atoms("type", 0, 1), dtype = ct.c_double)
    comm.Barrier()
    Helper.print("Time for gathering atoms:", time.time() - start_time)
    start_time = time.time()

    numatoms = len(lmp.atoms)
    proc_atoms = int(numatoms/size)
    new_atoms = []
    if rank == size - 1:
        upper_bound = numatoms
    else:
        upper_bound = proc_atoms*(rank + 1)
    sys.stdout.flush()
#    for i in range(len(lmp.atoms)):
    side = lmp.eval('lx')
    for i in range(rank*proc_atoms, upper_bound):
        float_pos = my_atoms[i]
        if types[i] > Data.initial_types:
            new_type = int(types[i])
        else:
            surface_pos = near_surface(coords, float_pos[:-1], angles, dr, Data.non_inter_cutoff, side)
            if surface_pos == -1:
                new_type = int(types[i]) + Data.initial_types
            elif surface_pos == 1:
                new_type = int(types[i]) + 2*Data.initial_types
            else:
                new_type = int(types[i])

        position = " ".join(map(str, float_pos))
        new_atoms.append((new_type, position))

    comm.Barrier()
    full_atoms = []
    for i in range(size):
        full_atoms += comm.bcast(new_atoms, root = i)
    for new_type, position in full_atoms:
        new_lmp.create_atoms(new_type, "single", position)

    Helper.print("Time for atom scan:", time.time() - start_time)
    #Testing
    new_lmp.run(0)
#    new_lmp.minimize(f"1.0e-8 1.0e-8 {convert_timestep(lmp, 0.01)} {convert_timestep(lmp, 0.1)}")
    new_lmp.write_data(f"output.{Helper.output_ctr}.structure")
    Helper.output_ctr += 1
    return new_lmp


@Lmpfunc
def quazi_static(lmp, dr_frac = 0.1, dtheta = 10):
    Helper.print("start")
    dr = dr_frac*min(lmp.eval("lx"), lmp.eval("ly"), lmp.eval("lz"))
    Helper.print(f"dr value: {dr}")
    create_surface(lmp)
    filename = glob.glob("glass_*.structure")[-1]
    name_handle = re.search(r"(?<=glass_).+(?=\.structure)", filename).group()
    potfile = os.path.abspath(f"pot_{name_handle}.FF")
    potfile = modify_potfile(lmp, potfile)
    start_coords = (lmp.eval("lx")/2 + lmp.system.xlo, SystemParams.parameters["old_bounds"][0] - 1)
    Data.initial_types = lmp.system.ntypes

    starting_dir = "initial"
    if not os.path.isdir(starting_dir):
        Helper.command(f"mkdir {starting_dir}")
    elif os.listdir(starting_dir):
        Helper.command(f"rm -r {starting_dir}/*")
    Helper.chdir(starting_dir)

    #Check so that surface regions don't overlap through a periodix x boundary
    x_side = lmp.eval("lx")
    if dtheta*(np.pi/180) < np.arcsin(2*Data.non_inter_cutoff/x_side):
        angle_span = (2*dtheta*(np.pi/180), np.pi - 2*dtheta*(np.pi/180), int(180/dtheta) - 3)
    else:
        angle_span = (dtheta*(np.pi/180), np.pi - dtheta*(np.pi/180), int(180/dtheta) - 1)
        
    new_lmp = QSR(lmp = lmp, coords = start_coords, dr = dr, span = angle_span, angles = (np.pi/2, None), potfile = potfile, in_glass = False)

    new_lmp.write_data(f"output.{Helper.output_ctr}.structure")
    Helper.output_ctr = 0
    
    result = 0.69*(abs(lmp.eval("pe")) - abs(new_lmp.eval("pe")))/new_lmp.variables["surface_area"].value

    Helper.print(new_lmp.variables["surface_area"].value)

    new_lmp.close()
    Helper.mkdir("../QS_results")
    for filename in os.listdir():
        if os.path.isfile(filename):
            Helper.command(f"mv {filename} ../QS_results")

    return result


@Lmpfunc
def QSR(lmp, coords, dr, span, angles, potfile, in_glass):
    prev_theta, theta = angles
    prev_theta = np.pi/2 if not prev_theta else prev_theta
    Helper.command(f"echo '{coords}' >> path.txt")
    Helper.print(f"Recursion step: {Helper.output_ctr}")

    #Output
    Helper.print("Coordinates: ", coords)
    Helper.print(f"In {os.getcwd().split(os.path.commonprefix([potfile, os.getcwd()]))[-1]}")

    if not in_glass and coords[1] >= SystemParams.parameters["old_bounds"][0]:
        in_glass = True
        lmp.variable(f"surface_area equal 0")

    if in_glass:
        lmp = copy_lmp(lmp, potfile, (prev_theta, theta), dr, coords)
        if coords[0] > lmp.system.xhi:
            dist = dr - (coords[0] - lmp.system.xhi)/np.cos(theta)
        elif coords[0] < lmp.system.xlo:
            dist = dr - (lmp.system.xlo - coords[0])/np.sin(np.pi - theta)
        elif coords[1] > SystemParams.parameters["old_bounds"][1]:
            dist = dr - (coords[1] - SystemParams.parameters["old_bounds"][1])/np.sin(theta)
        elif coords[1] - dr*np.sin(theta) < SystemParams.parameters["old_bounds"][0]:
            dist = (coords[1] - SystemParams.parameters["old_bounds"][0])/np.sin(theta)
        else:
            dist = dr

        lmp.variable(f"surface_area equal {lmp.variables['surface_area'].value + dist*lmp.eval('lz')}")

        if coords[0] >= lmp.system.xhi or coords[0] <= lmp.system.xlo or coords[1] >= SystemParams.parameters["old_bounds"][1]:
            return lmp


    lowest_pot = float("infinity")
    res_lmp = None
    prev_theta = theta

    for theta in np.linspace(*span):
        if prev_theta is None or np.pi - prev_theta + theta >= np.pi/2 and prev_theta + np.pi - theta >= np.pi/2:
            Helper.print("Theta: ", str(theta))
            Helper.mkdir(f"{theta}")
            Helper.chdir(f"{theta}")
            tmp_lmp =  QSR(lmp, coords = (coords[0] + dr*np.cos(theta), coords[1] + dr*np.sin(theta)), dr = dr, span = span, angles = (prev_theta, theta), potfile = potfile, in_glass = in_glass)
            Helper.chdir("..")
            Helper.command(f"mv {theta}/output.*.structure .")
            new_pe = abs(tmp_lmp.eval("pe"))
            Helper.print(f"lowest pe: {new_pe}")
            if new_pe < lowest_pot:
                if os.path.isfile(f"{theta}/log.lammps"):
                    Helper.command(f"mv {theta}/log.lammps tmp_log.lammps")
                Helper.command(f"mv {theta}/path.txt tmp_path.txt")
                selected_angle = theta
                lowest_pot = new_pe
                if res_lmp and res_lmp != lmp:
                    res_lmp.close()
                res_lmp = tmp_lmp
            elif tmp_lmp != lmp:
                tmp_lmp.close()

            Helper.rmtree(f"{theta}", ignore_errors = True)
    Helper.command("cat tmp_path.txt >> path.txt")
    Helper.command("cat tmp_log.lammps >> log.lammps")
    Helper.command("rm tmp_path.txt")
    Helper.command("rm tmp_log.lammps")
    return res_lmp


def main():
    lmp = PyLammps(verbose = False)

    system_parameters_initialization(lmp, units = "real")


    filename = glob.glob("glass_*.structure")[-1]
    lmp.read_data(filename)

    lmp.timestep(1)

    name_handle = re.search(r"(?<=glass_).+(?=\.structure)", filename).group()

    lmp.variable(f"home_dir string {os.getcwd()}")
    lmp.include(f"pot_{name_handle}.FF")

    vizualization(lmp)

    #Minimization
    lmp.run(0)
    lmp.velocity(f"all create {SystemParams.parameters['simulation_temp']} 12345 dist gaussian")


    Helper.print("\n\nG:", quazi_static(lmp))
#    Helper.print("\n\nG:", quazi_static(lmp, 0.25, 45))

    return lmp

if __name__ == "__main__":
    main().lmp.finalize()
#    MPI.COMM_WORLD.Barrier()
#    MPI.Finalize()
