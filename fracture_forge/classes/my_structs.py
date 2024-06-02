from lammps import lammps
from classes.Storage import SystemParams, Helper, Data
import glob, os
import numpy as np
import ctypes as ct
import regex as re

class FracGraph:
    def __init__(self, connection_radius, error = 0.1, start_buffer = 0.5, test_mode = False, simulation_temp = 300, ):
        self.__dr = connection_radius
        self.__test_mode = test_mode
        self.__head = Node(is_head = True, test_mode = self.__test_mode)
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
        Data.old_bounds = (min_y, max_y)

        Helper.print("Old bounds:", Data.old_bounds)
        #head_lmp.command(f"change_box all y delta {-Data.non_inter_cutoff} {Data.non_inter_cutoff}")
        #head_lmp.command(f"fix surface_relax all npt temp {simulation_temp} {simulation_temp} {100*lmp.eval('dt')} iso 1 1 {1000*lmp.eval('dt')}")
        #head_lmp.command(f"run {Helper.convert_timestep(head_lmp, 0.1)}")
        #head_lmp.command("unfix surface_relax")
        """End of surface creation"""

        box = head_lmp.extract_box()
        self.__box = box
        x_side = box[1][0] - box[0][0]
        start_pos = (x_side/2 + box[0][0], Data.old_bounds[0] - start_buffer)
        self.__head.set_tip(self.__discretize(start_pos))
        self.__node_hash = {self.__discretize(self.__head.get_pos()) : self.__head}

    #Getters
    def get_box(self):
        return self.__box

    def get_head(self):
        return self.__head

    def __len__(self):
        return self.__node_ctr

    def flatten(self):
        return self.__node_hash.values()

    def get_node_coords(self):
        return np.array(list(self.__node_hash.keys()))

    #Main behavior
    def __discretize(self, coords):
        return (self.__grid_size*round(coords[0]/self.__grid_size), self.__grid_size*round(coords[1]/self.__grid_size))
        #return coords

    def build_mid_atom(self, pivot_atom_type, num_neighs, interactions = "default"):
        if interactions == "default":
            Data.type_groups = 1
        else:
            Data.type_groups = max(sum(interactions, ()))

        head_lmp = self.__head.get_lmp()

        #Calculate simulation region sides
        sides = np.array([self.__box[1][0] - self.__box[0][0], self.__box[1][1] - self.__box[0][1], self.__box[1][2] - self.__box[0][2]])


        positions = np.array(head_lmp.gather_atoms("x", 1, 3), dtype = ct.c_double).reshape((-1, 3))
        types = np.array(head_lmp.gather_atoms("type", 0, 1), dtype = ct.c_int)
        ids = np.array(head_lmp.gather_atoms("id", 0, 1), dtype = ct.c_int)

        self.__node_pos = np.array(list())

        #Cycle through the local ids of atoms with the oxygen type
        for pid in np.where(types == pivot_atom_type)[0]:
            #Get the distance vector components
            diff = positions - positions[pid]
            #Enforce the periodic boundary condition
            diff -= np.around(diff/sides)*sides
            #Calculate the distance vector norm
            distances = np.linalg.norm(diff, axis = 1)
            #Mask out the unwanted neighbor candidates (oxygens)
            distances = np.where(types == pivot_atom_type, float("inf"), distances)
            #Get local ids of two closest oxygen neighbors
            neigh_ids = np.where(np.isin(distances, np.partition(distances, num_neighs - 1)[:num_neighs]))[0]
            #Calculate mid-bond position
            mid_positions = positions[pid] + diff[neigh_ids]/2
            #Enforce periodic boundaries
            mid_positions += (mid_positions < self.__box[0])*sides
            mid_positions -= (mid_positions > self.__box[1])*sides
            for mid_pos in mid_positions:
                self.attach(coords = mid_pos[:-1])




    def build(self, dtheta, dr, interactions = "default"):
        if interactions == "default":
            Data.type_groups = 1
        else:
            Data.type_groups = max(sum(interactions, ()))
        x_side = self.__box[1][0] - self.__box[0][0]
        Data.initial_types = self.__head.get_lmp().extract_global("ntypes")
        self.__modify_potfile(interactions)
        self.__dr = dr
        #Check so that surface regions don't overlap through a periodix x boundary
        if dtheta*(np.pi/180) < np.arcsin(2*Data.non_inter_cutoff/x_side):
            self.__angle_span = (2*dtheta*(np.pi/180), np.pi - 2*dtheta*(np.pi/180), int(180/dtheta) - 3)
        else:
            self.__angle_span = (dtheta*(np.pi/180), np.pi - dtheta*(np.pi/180), int(180/dtheta) - 1)

        
        self.__generate_nodes(coords = self.__head.get_pos())

    def __generate_nodes(self, coords, parent = None, theta = None, in_glass = False):
        if not in_glass and coords[1] >= Data.old_bounds[0]:
            in_glass = True

        if in_glass:
            if coords[0] >= self.__box[1][0] or coords[0] <= self.__box[0][0]:
                return
            parent = self.attach(coords, node = parent, angle = theta)


        if coords[1] >= Data.old_bounds[1]:
            return

        if theta:
            for theta in np.linspace(*self.__angle_span):
                self.__generate_nodes(coords = (coords[0] + self.__dr*np.cos(theta), coords[1] + self.__dr*np.sin(theta)), parent = parent, theta = theta, in_glass = in_glass)
        else:
            for x in np.linspace(self.__box[0][0] + self.__dr*0.2, self.__box[1][0] - self.__dr*0.2, int((self.__box[1][0] - self.__box[0][0] - self.__dr*0.4)/self.__dr)):
                self.__generate_nodes(coords = (x, self.__box[0][1] + self.__dr*0.2), parent = parent, theta = np.pi/2, in_glass = in_glass)

    def calculate(self, save_dir = "out_structs", outp_freq = 1):
        if os.path.isdir(save_dir):
            os.system(f"rm -r {save_dir}")
        os.mkdir(save_dir)

        self.__outp_freq = outp_freq
        self.__lowest_path_energy = float("inf")
        self.__save_dir = save_dir
        self.__head.activate()
        starting_pe = self.__head.get_lmp().get_thermo("pe")

        lowest_pe = self.__calc_subtree(self.__head)

        lowest_leaf = self.__head.get_lowest_leaf()
        lowest_leaf.get_lmp().command("write_data result.structure")
        try:
            res = (lowest_pe - starting_pe)/lowest_leaf.get_surface_area()
        except:
            res = 0

        return res

    def __calc_subtree(self, node):
        if node.is_leaf():
            pe = node.get_path_pe()
            if pe < self.__lowest_path_energy:
                self.__lowest_path_energy = pe
            return pe
        else:
            lowest_pot = float("inf")
            to_terminate = list()
            children = node.get_children()
            lowest_child = None
            for child in children:
                if not child.is_active():
                    child.set_parent(node)
                    child.activate(dr = self.__dr, potfile = self.__new_potfile, test_mode = self.__test_mode)
                    if not child.get_id()%self.__outp_freq:
                        child.get_lmp().command(f"write_data {self.__save_dir}/output.{child.get_id()}.structure")
                    if child.get_lmp().get_thermo("pe") > self.__lowest_path_energy:
                        res_pe = float("inf")
                        child.set_path_pe(res_pe)
                    else:
                        res_pe = self.__calc_subtree(child)
                else:
                    res_pe = child.get_path_pe()
                if res_pe < lowest_pot:
                    lowest_pot = res_pe
                    lowest_child = child

            for child in children:
                if child is not lowest_child:
                    child.get_lmp().close()
            node.set_path_pe(lowest_pot)
            if lowest_child:
                node.set_lowest_leaf(lowest_child.get_lowest_leaf())
            else:
                node.set_lowest_leaf(None)
            return lowest_pot

    def attach(self, coords):
        disc_coords = self.__discretize(coords)

        if disc_coords in self.__node_hash:
            new_node = self.__node_hash[disc_coords]
        else:
            new_node = Node(tip = disc_coords, node_ctr = self.__node_ctr)
            self.__node_hash[disc_coords] = new_node
            self.__node_ctr += 1

        num_bins = int(np.floor(self.__dr/self.__grid_size))
        for x in np.linspace(disc_coords[0] - self.__grid_size*num_bins, disc_coords[0] + self.__grid_size*num_bins, num_bins*2):
            for y in np.linspace(disc_coords[1] - self.__grid_size*num_bins, disc_coords[1] + self.__grid_size*num_bins, num_bins*2):
                if (x, y) != disc_coords and (x, y) in self.__node_hash:
                    self.__node_hash[(x, y)].attach(new_node)
                    new_node.attach(self.__node_hash[(x, y)])



        return new_node

    def __modify_potfile(self, interactions):
        groups = Data.type_groups
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

        self.__new_potfile = name

        

class Node:
    def __init__(self, is_head = False, tip = None, units = "real", test_mode = False, node_ctr = 0):
        self.__id = node_ctr
        self.__active = False
        self.__is_head = is_head
        self.__children = list()
        self.__tip = tip
        self.__surface_area = 0

        if self.__is_head:
            if test_mode:
                if os.path.isdir("logs"):
                    os.system("rm -r logs")
                os.mkdir("logs")
                self.__lmp = lammps(cmdargs = ["-log", f"logs/log.0.lammps"])
            else:
                self.__lmp = lammps(cmdargs = ["-log", "none", "-screen", "none"])
            self.__system_parameters_initialization(units = units)
            if Data.structure_file:
                filename = Data.structure_file
            else:
                filename = glob.glob("glass_*.structure")[-1]
            self.__lmp.command(f"read_data {filename}")
            self.__structure_file = filename
            if Data.potfile:
                self.potfile = Data.potfile
            else:
                name_handle = re.search(r"(?<=glass_).+(?=\.structure)", filename).group()
                self.potfile = os.path.abspath(f"pot_{name_handle}.FF")
            

    #Setters
    def set_path_pe(self, pe):
        self.__path_pe = pe

    def set_tip(self, coords):
        if self.__is_head:
            self.__tip = coords

    def attach(self, node):
        self.__children.append(node)
        return self

    def set_parent(self, node):
        self.__parent = node

    def set_lowest_leaf(self, node):
        self.__lowest_leaf = node

    def get_parent_angle(self):
        if self.__parent:
            return self.__parent.__theta
        else:
            return np.pi/2

    #Getters
    def get_struct_filename(self):
        return self.__structure_file

    def get_id(self):
        return self.__id

    def get_parent(self):
        return self.__parent

    def get_surface_area(self):
        return self.__surface_area

    def get_children(self):
        return self.__children

    def get_pos(self):
        return self.__tip

    def get_path_pe(self):
        return self.__path_pe

    def get_lowest_leaf(self):
        return self.__lowest_leaf

    def get_lmp(self):
        return self.__lmp

    #State functions
    def is_head(self):
        return self.__is_head

    def is_leaf(self):
        return not len(self.__children)

    def is_active(self):
        return self.__active

    #Main behavior
    def __calc_angle(self):
        if not self.__parent:
            return np.pi/2
        else:
            pass

    def __system_parameters_initialization(self, units):
        self.__lmp.command(f"units {units}")
        SystemParams.units = units
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


    def activate(self, potfile = None, timestep = 1, units = "real", thermo_step = 1000, dump_step = 1000, dr = None, test_mode = False, parent = None):
        self.__parent = parent
        self.__theta = self.__calc_angle()
        self.__active = True
        if self.__is_head:
            self.__lmp.command(f"include {self.potfile}")
            print("Head node activated")
        else:

            old_lmp = self.__parent.get_lmp()
            if test_mode:
                self.__lmp = lammps(cmdargs = ["-log", f"logs/log.{self.__id}.lammps"])
            else:
                self.__lmp = lammps(cmdargs = ["-log", "none", "-screen", "none"])

            self.__system_parameters_initialization(units = units)
            self.dr = dr
            self.__prev_theta = self.get_parent_angle()

            self.__system_parameters_initialization(units = units)
            self.__lmp.command(f"timestep {timestep}")
            box = old_lmp.extract_box()

            if self.__tip[0] > box[1][0]:
                dist = dr - (self.__tip[0] - box[1][0])/np.cos(self.__theta)
            elif self.__tip[0] < box[0][0]:
                dist = dr - (box[0][0] - self.__tip[0])/np.sin(np.pi - self.__theta)
            elif self.__tip[1] > Data.old_bounds[1]:
                dist = dr - (self.__tip[1] - Data.old_bounds[1])/np.sin(self.__theta)
            elif self.__tip[1] - dr*np.sin(self.__theta) < Data.old_bounds[0]:
                dist = (self.__tip[1] - Data.old_bounds[0])/np.sin(self.__theta)
            else:
                dist = dr

            self.__surface_area = self.__parent.get_surface_area() + dist*(box[1][2] - box[0][2])

            self.__lmp.command(f"region my_simbox block {box[0][0]} {box[1][0]} {box[0][1]} {box[1][1]} {box[0][2]} {box[1][2]}")
            self.__lmp.command(f"create_box {Data.initial_types*Data.type_groups} my_simbox")
            self.__vizualization(thermo_step, dump_step)
            self.__lmp.command(f"include {potfile}")

            my_atoms = np.array(old_lmp.gather_atoms("x", 1, 3), dtype = ct.c_double).reshape((-1, 3))
            types = np.array(old_lmp.gather_atoms("type", 0, 1), dtype = ct.c_double)
            self.box_side = box[1][0] - box[0][0]

            self.__cut(my_atoms, types, old_lmp.get_natoms())
            print(f"Node {self.__id} activated at x = {round(self.__tip[0], 3)}, y = {round(self.__tip[1], 3)}")
            

        self.__lmp.command("run 0")
        self.potfile = potfile

        if self.is_leaf():
            self.__lowest_leaf = self
            self.__path_pe = self.__lmp.get_thermo("pe")


    def __cut(self, my_atoms, types, natoms):
        for i in range(natoms):
            float_pos = my_atoms[i]
            if types[i] <= Data.initial_types:
                group = self.__near_surface(float_pos[:-1])
                if group <= Data.type_groups:
                    new_type = int(types[i]) + (group - 1)*Data.initial_types
                else:
                    new_type = int(types[i])
            elif types[i] > Data.initial_types and types[i] <= 3*Data.initial_types:
                new_type = int(types[i])
            else:
                group = self.__near_surface(float_pos[:-1])
                if group <= Data.type_groups:
                    tp = int(types[i])%Data.initial_types
                    tp = tp if tp else tp + Data.initial_types
                    new_type = tp + (group - 1)*Data.initial_types
                else:
                    new_type = int(types[i])

            position = " ".join(map(str, float_pos))
            self.__lmp.command(f"create_atoms {new_type} single {position}")



    def __near_surface(self, atom_pos):
        cutoff = Data.non_inter_cutoff
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
        

