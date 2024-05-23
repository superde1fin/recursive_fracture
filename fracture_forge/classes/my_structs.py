from lammps import lammps
from classes.Storage import SystemParams, Data, Helper
import glob, os
import numpy as np
import ctypes as ct
import regex as re

class FracTree:
    def __init__(self, error = 0.1, start_buffer = 0.5):
        self.__head = Node(is_head = True)
        self.__grid_size = error/np.sqrt(2)
        self.__node_ctr = 1
        head_lmp = self.__head.get_lmp()

        """Surface creation"""
        min_y = float("inf")
        max_y = float("-inf")
        my_atoms = np.array(head_lmp.gather_atoms("x", 1, 3), dtype = ct.c_double).reshape((-1, 3))
        for atom in my_atoms:
            if atom[1] < min_y:
                min_y = atom[1]
            if atom[1] > max_y:
                max_y = atom[1]
        SystemParams.parameters["old_bounds"] = (min_y, max_y)

        Helper.print("Old bounds:", SystemParams.parameters["old_bounds"])
        #head_lmp.command(f"change_box all y delta {-Data.non_inter_cutoff} {Data.non_inter_cutoff}")
        #head_lmp.command(f"fix surface_relax all npt temp {SystemParams.parameters['simulation_temp']} {SystemParams.parameters['simulation_temp']} {100*lmp.eval('dt')} iso 1 1 {1000*lmp.eval('dt')}")
        #head_lmp.command(f"run {Helper.convert_timestep(head_lmp, 0.1)}")
        #head_lmp.command("unfix surface_relax")
        """End of surface creation"""

        box = head_lmp.extract_box()
        self.__box = box
        x_side = box[1][0] - box[0][0]
        start_pos = (x_side/2 + box[0][0], SystemParams.parameters["old_bounds"][0] - start_buffer)
        self.__head.set_tip(self.__discretize(start_pos))
        self.__nodes_hash = {self.__head.get_pos() : self.__head}


    def build(self, dtheta, dr, interactions = "default"):
        x_side = self.__box[1][0] - self.__box[0][0]
        Data.initial_types = self.__head.get_lmp().extract_global("ntypes")
        self.__modify_potfile(interactions, groups = Data.type_groups)
        self.__dr = dr
        #Check so that surface regions don't overlap through a periodix x boundary
        if dtheta*(np.pi/180) < np.arcsin(2*Data.non_inter_cutoff/x_side):
            self.__angle_span = (2*dtheta*(np.pi/180), np.pi - 2*dtheta*(np.pi/180), int(180/dtheta) - 3)
        else:
            self.__angle_span = (dtheta*(np.pi/180), np.pi - dtheta*(np.pi/180), int(180/dtheta) - 1)

        
        self.__generate_nodes(coords = self.__head.get_pos())


    def calculate(self, save_dir = "out_structs"):
        Helper.lowest_path_energy = float("inf")
        Helper.fracture_memory = {}
        Helper.output_ctr = 0
        self.__save_dir = save_dir
        self.__head.activate()
        starting_pe = self.__head.get_lmp().get_thermo("pe")

        lowest_pe = self.__calc_subtree(self.__head)

        node.get_lowest_leaf().get_lmp().comand("write_data result.structure")

        return (lowest_pe - starting_pe)/self.__surface_area

    def __calc_subtree(self, node):
        #print("On node with coords:", node.get_pos())
        if node.is_leaf():
            return node.get_path_pe()
        else:
            lowest_pot = float("inf")
            for child in node.get_children():
                if not child.is_active():
                    child.activate()
                    child.get_lmp().command(f"write_data {self.__save_dir}/output.{Helper.output_ctr}.structure")
                    Helper.output_ctr += 1
                    res_pe = self.__calc_subtree(child)
                else:
                    res_pe = child.get_path_pe()
                if res_pe < lowest_pot:
                    lowest_pot = res_pe
                    lowest_child = child
            node.set_path_pe(lowest_pot)
            node.set_lowest_leaf(lowest_child.get_lowest_leaf())
            return lowest_pot


    def __generate_nodes(self, coords, parent = None, theta = None, in_glass = False):
        if not in_glass and coords[1] >= SystemParams.parameters["old_bounds"][0]:
            in_glass = True
            self.__surface_area = 0

        if in_glass:
            #print("Attaching node at:", coords)
            parent = self.attach(coords, node = parent, angle = theta)

            if coords[0] > self.__box[1][0]:
                dist = self.__dr - (coords[0] - self.__box[1][0])/np.cos(theta)
            elif coords[0] < self.__box[0][0]:
                dist = self.__dr - (self.__box[0][0] - coords[0])/np.sin(np.pi - theta)
            elif coords[1] > SystemParams.parameters["old_bounds"][1]:
                dist = self.__dr - (coords[1] - SystemParams.parameters["old_bounds"][1])/np.sin(theta)
            elif coords[1] - self.__dr*np.sin(theta) < SystemParams.parameters["old_bounds"][0]:
                dist = (coords[1] - SystemParams.parameters["old_bounds"][0])/np.sin(theta)
            else:
                dist = self.__dr

            self.__surface_area += dist*(self.__box[1][2] - self.__box[0][2])

        if coords[0] >= self.__box[1][0] or coords[0] <= self.__box[0][0] or coords[1] >= SystemParams.parameters["old_bounds"][1]:
            return

        for theta in np.linspace(*self.__angle_span):
            self.__generate_nodes(coords = (coords[0] + self.__dr*np.cos(theta), coords[1] + self.__dr*np.sin(theta)), parent = parent, theta = theta, in_glass = in_glass)

    def get_box(self):
        return self.__box

    def get_head(self):
        return self.__head


    def __len__(self):
        return self.__node_ctr


    def __discretize(self, coords):
        return (self.__grid_size*round(coords[0]/self.__grid_size), self.__grid_size*round(coords[1]/self.__grid_size))
        #return coords

    def attach(self, coords, node = None, angle = None):
        if not node:
            node = self.__head
        if self.__discretize(coords) in self.__nodes_hash:
            new_node = self.__nodes_hash[self.__discretize(coords)]
        else:
            self.__node_ctr += 1
            new_node = Node(tip = self.__discretize(coords), parent = node, angle = angle)
            self.__nodes_hash[self.__discretize(coords)] = new_node
        node.attach(new_node)
        return new_node

    def flatten(self):
        return self.__nodes_hash.values()

    def __modify_potfile(self, interactions, groups = 2):
        ntypes = Data.initial_types
        if interactions == "default":
            interactions = []
            for g in range(2, groups + 1):
                interactions.append((1, g))
                interactions.append((g, 1))
        else:
            for i in range(len(interactions)):
                interactions.append(interactions[i][::-1])
            
        text = open(self.__head.potfile, 'r').read()
        name = re.sub(r"(?<=\/[^/]+)\.(?=.+$)", "_new.", self.__head.potfile)
        new_potfile = open(name, 'w')
        new_text = text + "\n\n#-------------------------\n\n"
        for t in range(1, ntypes + 1):
            for g in range(groups - 1):
                mass_re = re.compile(f"^mass\s+{ntypes*g + t}\s+.+$", re.MULTILINE)
                mass_line = mass_re.findall(new_text)[-1]
                new_text = mass_re.sub(mass_line + '\n' + re.sub(f"(?<=^mass\s+){ntypes*g + t}(?=\s+.+$)", str(ntypes*(g + 1) + t), mass_line) + '\n', new_text)


                for j in range(t, ntypes + 1):
                    coeff_line = re.compile(f"^pair_coeff\s+{t}\s+{j}\s+.+$", re.MULTILINE).findall(new_text)[-1]
                    new_text += '\n' + re.sub(f"(?<=^pair_coeff\s+){t}\s+{j}(?=\s+.+$)", f"{ntypes*(g + 1) + t} {ntypes*(g + 1) + j}", coeff_line) + f"\t#Groups ({g + 2}, {g + 2}) for types ({t}, {j})"


                    for group_iter in range(g + 2, groups + 1):
                        if (g + 1, group_iter) in interactions:
                            new_text += '\n' + re.sub(f"(?<=^pair_coeff\s+){t}\s+{j}(?=\s+.+$)", f"{ntypes*g + t} {ntypes*(group_iter - 1) + j}", coeff_line) + f"\t#Groups ({g + 1}, {group_iter}) for types ({t}, {j})"
                            if t != j:
                                new_text += '\n' + re.sub(f"(?<=^pair_coeff\s+){t}\s+{j}(?=\s+.+$)", f"{ntypes*(group_iter - 1) + t} {ntypes*g + j}", coeff_line) + f"\t#Groups ({group_iter}, {g + 1}) for types ({t}, {j})"
                        else:
                            tmp_line = re.sub(f"(?<=^pair_coeff\s+){t}\s+{j}(?=\s+.+$)", f"{ntypes*g + t} {ntypes*(group_iter - 1) + j}", coeff_line)
                            new_text += '\n' + re.sub(f"(?<=^pair_coeff\s+{ntypes*g + t}\s+{ntypes*(group_iter - 1) + j}.+\}}\s+)\w+(?=\s+.+)", "NoNo", tmp_line) + f"\t#Groups ({g + 1}, {group_iter}) for types ({t}, {j})"
                            if t != j:
                                tmp_line = re.sub(f"(?<=^pair_coeff\s+){t}\s+{j}(?=\s+.+$)", f"{ntypes*(group_iter - 1) + t} {ntypes*g + j}", coeff_line)
                                new_text += '\n' + re.sub(f"(?<=^pair_coeff\s+{ntypes*(group_iter - 1) + t}\s+{ntypes*g + j}.+\}}\s+)\w+(?=\s+.+)", "NoNo", tmp_line) + f"\t#Groups ({group_iter}, {g + 1}) for types ({t}, {j})"
                
        general_type_re = re.compile(f"(?<=pair_coeff\s+\*\s+\*.+)(\s+\S+){{{ntypes}}}$", re.MULTILINE)
        type_names = general_type_re.search(new_text).group()*groups
        new_text = general_type_re.sub(type_names, new_text)
        new_potfile.write(new_text)

        self.__head.potfile = name

        

class Node:
    def __init__(self, parent = None, is_head = False, tip = None, units = "real", angle = np.pi/2):
        self.__active = False
        self.__is_head = is_head
        self.__children = list()
        self.__tip = tip
        self.__parent = parent
        self.__theta = angle
        self.__prev_theta = self.get_parent_angle()

        if self.__is_head:
            #Testing
            #self.__lmp = lammps(cmdargs = ["-log", "none", "-screen", "none"])
            self.__lmp = lammps(cmdargs = ["-log", f"logs/log.{Helper.output_ctr}.lammps"])
            self.__system_parameters_initialization(units = units)
            filename = glob.glob("glass_*.structure")[-1]
            self.__lmp.command(f"read_data {filename}")
            name_handle = re.search(r"(?<=glass_).+(?=\.structure)", filename).group()
            self.potfile = f"pot_{name_handle}.FF"

    def get_parent_angle(self):
        if self.__parent:
            return self.__parent.__theta
        else:
            return np.pi/2

    def get_parent(self):
        return self.__parent

    def is_head(self):
        return self.__is_head

    def set_tip(self, coords):
        if self.__is_head:
            self.__tip = coords

    def attach(self, node):
        self.__children.append(node)
        return self

    def get_children(self):
        return self.__children

    def get_pos(self):
        return self.__tip

    def is_leaf(self):
        return not len(self.__children)

    def is_active(self):
        return self.__active

    def activate(self, old_lmp = None, potfile = None, timestep = 1, units = "real", thermo_step = 1000, dump_step = 1000, dr = None):
        self.__active = True
        if self.__is_head:
            self.__lmp.command(f"variable home_dir string {os.getcwd()}")
            self.__lmp.command(f"include {self.potfile}")
        else:
            #Testing
            #self.__lmp = lammps(cmdargs = ["-log", "none", "-screen", "none"])
            self.__lmp = lammps(cmdargs = ["-log", f"logs/log.{Helper.output_ctr}.lammps"])
            self.__system_parameters_initialization(units = units)
            self.dr = dr

            self.__transfer_vars(old_lmp)
            self.__system_parameters_initialization(units = units)
            self.__lmp.command(f"timestep {timestep}")
            box = old.lmp.extract_box()
            self.__lmp.command(f"region my_simbox {box[0][0]} {box[1][0]} {box[0][1]} {box[1][1]} {box[0][2]} {box[1][2]}")
            self.__lmp.command(f"create_box {Data.initial_types*Data.type_groups} my_simbox")
            self.__vizualization(thermo_step, dump_step)
            self.__lmp.command(f"include {potfile}")

            my_atoms = np.array(old_lmp.gather_atoms("x", 1, 3), dtype = ct.c_double).reshape((-1, 3))
            types = np.array(old_lmp.gather_atoms("type", 0, 1), dtype = ct.c_double)
            self.box_side = box[1][0] - box[0][0]
            
            for i in range(self.__lmp.get_natoms()):
                float_pos = my_atoms[i]
                if types[i] <= Data.initial_types:
                    group = near_surface(float_pos[:-1], Data.non_inter_cutoff)
                    new_type = int(types[i]) + (group - 1)*Data.initial_types
                elif types[i] > Data.initial_types and types[i] <= 3*Data.initial_types:
                    new_type = int(types[i])
                else:
                    group = near_surface(float_pos[:-1], Data.non_inter_cutoff)
                    tp = int(types[i])%Data.initial_types
                    tp = tp if tp else tp + Data.initial_types
                    new_type = tp + (group - 1)*Data.initial_types

                position = " ".join(map(str, float_pos))
                self.__lmp.command(f"create_atoms {new_type} single {position}")


        self.__lmp.command(f"velocity all create {SystemParams.parameters['simulation_temp']} 12345 dist gaussian")
        self.__lmp.command("run 0")

        self.potfile = potfile

        if self.is_leaf():
            self.__lowest_leaf = self
            self.__path_pe = self.__lmp.get_thermo("pe")


    def set_path_pe(self, pe):
        self.__path_pe = pe

    def get_path_pe(self):
        return self.__path_pe

    def set_lowest_leaf(self, node):
        self.__lowest_leaf = node

    def get_lowest_leaf(self):
        return self.__lowest_leaf

    def __near_surface(self, atom_pos, cutoff):
        x0, y0 = self.__tip #Tip of the division vector

        #Tail of the division vector
        x1 = x0 - self.dr*np.cos(self.__theta)
        y1 = y0 - self.dr*np.sin(self.__theta)

        #Tail of the parallel transport of the division vector to the left by length 'cutoff'
        x2 = x1 - cutoff*np.sin(self.__theta)
        y2 = y1 + cutoff*np.cos(self.__theta)

        #Tail of the parallel transport of the division vector to the right by length 'cutoff'
        x3 = x1 + cutoff*np.sin(self.__theta)
        y3 = y1 - cutoff*np.cos(self.__theta)

        #Line through point (x0, y0) at angle theta
        f01 = np.poly1d([np.tan(self.__theta), y0 - x0*np.tan(self.__theta)])

        #Line through point (x0, y0) perpendicular to theta
        f02 = np.poly1d([np.tan(self.__theta + np.pi/2), y0 - x0*np.tan(self.__theta + np.pi/2)])

        #Line through point (x2, y2) at angle theta
        f21 = np.poly1d([np.tan(self.__theta), y2 - x2*np.tan(self.__theta)])

        #Line through point (x1, y1) perpendicular to theta
        f12 = np.poly1d([np.tan(self.__theta + np.pi/2), y1 - x1*np.tan(self.__theta + np.pi/2)])

        #Line through point (x3, y3) at angle self.__theta
        f31 = np.poly1d([np.tan(self.__theta), y3 - x3*np.tan(self.__theta)])

        #Line through point (x1, y1) at angle perpendicular to prev_theta
        f99 = np.poly1d([np.tan(self.__prev_theta + np.pi/2), y1 - x1*np.tan(self.__prev_theta + np.pi/2)])

        for x in [atom_pos[0], self.box_side + atom_pos[0], atom_pos[0] - self.box_side]:
        #Tail of the cut coverage at turns
            if (np.sqrt((x - x1)**2 + (atom_pos[1] - y1)**2) < cutoff) and (atom_pos[1] < np.polyval(f12, x)) and (atom_pos[1] > np.polyval(f99, x)):
                if (self.__prev_theta > self.__theta):
                    return 2
                elif (self.__prev_theta < self.__theta):
                    return 3

            if self.__theta <= np.pi/2:
                if (atom_pos[1] <= np.polyval(f02, x) and atom_pos[1] <= np.polyval(f21, x) and atom_pos[1] >= np.polyval(f12, x) and atom_pos[1] >= np.polyval(f01, x)):
                    return 2
                elif (atom_pos[1] <= np.polyval(f01, x) and atom_pos[1] <= np.polyval(f02, x) and atom_pos[1] >= np.polyval(f12, x) and atom_pos[1] >= np.polyval(f31, x)):
                    return 3
                elif ((np.sqrt((x - x0)**2 + (atom_pos[1] - y0)**2) <= cutoff) and atom_pos[1] >= np.polyval(f01, x) and atom_pos[1] >= np.polyval(f02, x)):
                    return 4
                elif ((np.sqrt((x - x0)**2 + (atom_pos[1] - y0)**2) <= cutoff) and atom_pos[1] <= np.polyval(f01, x) and atom_pos[1] >= np.polyval(f02, x)):
                    return 5
            else:
                if (atom_pos[1] <= np.polyval(f02, x) and atom_pos[1] >= np.polyval(f21, x) and atom_pos[1] >= np.polyval(f12, x) and atom_pos[1] <= np.polyval(f01, x)):
                    return 2
                elif (atom_pos[1] >= np.polyval(f01, x) and atom_pos[1] <= np.polyval(f02, x) and atom_pos[1] >= np.polyval(f12, x) and atom_pos[1] <= np.polyval(f31, x)):
                    return 3
                elif ((np.sqrt((x - x0)**2 + (atom_pos[1] - y0)**2) <= cutoff) and atom_pos[1] <= np.polyval(f01, x) and atom_pos[1] >= np.polyval(f02, x)):
                    return 4
                elif ((np.sqrt((x - x0)**2 + (atom_pos[1] - y0)**2) <= cutoff) and atom_pos[1] >= np.polyval(f01, x) and atom_pos[1] >= np.polyval(f02, x)):
                    return 5

        return 1
        

    def __transfer_vars(self, lmp):
        self.__lmp.variable(f"home_dir string {lmp.extract_variable('home_dir')}")

    def __system_parameters_initialization(self, units):
        self.__lmp.command(f"units {units}")
        SystemParams.parameters["units"] = units
        self.__lmp.command("atom_style charge")
        self.__lmp.command("boundary p p p")
        self.__lmp.command("comm_modify mode single vel yes")
        self.__lmp.command("neighbor 2.0 bin")
        self.__lmp.command("neigh_modify every 1 delay 0")

    def __vizualization(self, thermo_step, dump_step):
        #self.__lmp.command(f"thermo {thermo_step}")
        #self.__lmp.command("thermo_style custom step temp etotal pe vol density pxx pyy pzz")
        #self.__lmp.command("thermo_modify flush yes")

        #Computes
        self.__lmp.command("compute pe_pa all pe/atom")
    #    self.__lmp.command(f"dump my_dump all atom {dump_step} positions.{Helper.output_ctr}.dump")
    def get_lmp(self):
        return self.__lmp
