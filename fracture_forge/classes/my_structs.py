from lammps import lammps
from classes.Storage import SystemParams, Helper, Data
import glob, os, sys
import numpy as np
import ctypes as ct
import regex as re
import heapq
from scipy import interpolate

class FracGraph:
    def __init__(self, connection_radius, error = 0.1, start_buffer = 0.5, test_mode = False, simulation_temp = 300, ):
        self.__dr = connection_radius
        self.__test_mode = test_mode
        self.__head = Node(is_head = True, test_mode = self.__test_mode)
        self.__head.set_parent(None)
        self.__tail = Node(is_tail = True, test_mode = self.__test_mode, node_ctr = 1)
        self.__node_ctr = 2
        self.__paths = dict()
        self.__grid_size = error/np.sqrt(2)
        if self.__dr < self.__grid_size:
            self.__dr = self.__grid_size*1.5
            Helper.print("Reset probe radius to allow for fracture graph connectivity")
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
        self.__head.set_tip(start_pos)
        self.__tail.set_tip((x_side/2 + box[0][0], Data.old_bounds[1] + start_buffer))
        self.__head.energy = 0
        self.__tail.energy = 0
        self.__node_hash = {self.__head.get_pos() : self.__head, self.__tail.get_pos() : self.__tail}

    #Getters
    def get_box(self):
        atom_box = tuple(self.__box)
        atom_box[0][1] = Data.old_bounds[0]
        atom_box[1][1] = Data.old_bounds[1]
        return atom_box

    def get_head(self):
        return self.__head

    def __len__(self):
        return self.__node_ctr

    def flatten(self):
        return self.__node_hash.values()

    def get_node_coords(self):
        return np.array(list(self.__node_hash.keys()))
        #return np.array([neigh.get_pos() for neigh in self.__tail.get_neighbors()])

    def __trunc(self, values, dec = 0):
        return np.trunc(np.array(values)*(10**dec))/(10**dec)

    #Main behavior
    def __discretize(self, coords):
        rel_coords = np.array(coords) - self.__box[0][:-1]
        grid_coords = self.__grid_size*np.around(rel_coords/self.__grid_size) + self.__box[0][:-1]
        grid_coords -= (grid_coords > self.__box[1][:-1])*self.__grid_size
        return tuple(self.__trunc(grid_coords, 3))
        return tuple(grid_coords)

    def build_test(self, interactions):
        if interactions == "default":
            Data.type_groups = 1
        else:
            Data.type_groups = max(sum(interactions, ()))


        Data.initial_types = self.__head.get_lmp().extract_global("ntypes")
        self.__modify_potfile(interactions)
        self.__modify_struct()
        node = self.attach(coords = (25.315, 4.176))
        self.__head.attach(node)
        node1 = self.attach(coords = (24.325, 4.106))
        node.attach(node1)
        node1.attach(self.__tail)

    def __build_potmap(self):
        file = open(self.__head.potfile, "r")
        text = file.read()
        table_file = re.findall(pattern = r"(?<=^variable\s+table_path.+\").+(?=\")", string = text, flags = re.MULTILINE)[-1]
        pair_coeff_lines = re.findall(pattern = r"(?<=^pair_coeff\s+)((?:[^\s]+\s+){5})", string = text, flags = re.MULTILINE)
        anchor_map = dict()
        anchors = {line.split()[-1] for line in pair_coeff_lines}
        with open(table_file, "r") as tfile:
            file_iter = iter(enumerate(tfile))
            for i, line in file_iter:
                stripped = line.strip()
                if stripped in anchors:
                    num_entries = int(next(file_iter)[-1].split()[-1])
                    table_vals = np.loadtxt(table_file, skiprows = i + 3, max_rows = num_entries)
                    dists = table_vals[:, 1]
                    anchor_map[stripped] = interpolate.interp1d(table_vals[:, 1], table_vals[:, -2], kind = "linear")
        type_map = dict()
        for line in pair_coeff_lines[1:]:
            type1, type2, _, _, anchor = line.split()
            type_key = tuple(sorted((int(type1), int(type2))))
            if not type_key in type_map:
                if anchor in anchor_map.keys():
                    type_map[type_key] = anchor_map[anchor]

        file.close()
        return type_map



    def build(self, pivot_atom_type, num_neighs, interactions = "default"):
        if interactions == "default":
            Data.type_groups = 1
        else:
            Data.type_groups = max(sum(interactions, ()))

        Helper.print("Number of type groups:", Data.type_groups)


        Data.initial_types = self.__head.get_lmp().extract_global("ntypes")
        head_lmp = self.__head.get_lmp()
        self.__head.activate()
        forcefield = self.__build_potmap()

        #Calculate simulation region sides
        sides = np.array([self.__box[1][0] - self.__box[0][0], self.__box[1][1] - self.__box[0][1], self.__box[1][2] - self.__box[0][2]])

        head_divs = int(np.ceil(sides[0]/self.__dr))
        head_step = sides[0]/head_divs
        head_nodes = dict()
        tail_nodes = dict()

        #Gather per-atom information
        positions = np.array(head_lmp.gather_atoms("x", 1, 3), dtype = ct.c_double).reshape((-1, 3))
        types = np.array(head_lmp.gather_atoms("type", 0, 1), dtype = ct.c_int)
        neigh_lists = head_lmp.numpy.get_neighlist(head_lmp.find_pair_neighlist("table"))
        tags = head_lmp.extract_atom("id")
        for local_id, nlist in neigh_lists:
            tag1 = tags[local_id]
            type1 = types[tag1 - 1]
            for id2 in nlist:
                tag2 = tags[id2]
                type2 = types[tag2 - 1]
                delta = positions[tag1 - 1] - positions[tag2 - 1]
                delta -= np.around(delta/sides)*sides
                dist = np.linalg.norm(delta)
                type_key = tuple(sorted((type1, type2)))
                if dist >= forcefield[type_key].x.min() and dist <= forcefield[type_key].x.max():
                    pair_energy = forcefield[type_key](dist)
                    mid_pos = positions[tag2 - 1] + delta/2
                    mid_pos += (mid_pos < self.__box[0])*sides
                    mid_pos -= (mid_pos > self.__box[1])*sides
                    node = self.attach(coords = mid_pos[:-1], energy = pair_energy)
                    disc_coords = node.get_pos()
                    node_pos = int(np.floor((disc_coords[0] - self.__box[0][0])/head_step))
                    if node_pos in head_nodes:
                        if disc_coords[1] < head_nodes[node_pos].get_pos()[1]:
                            head_nodes[node_pos] = node
                    else:
                        head_nodes[node_pos] = node

                    if node_pos in tail_nodes:
                        if disc_coords[1] > tail_nodes[node_pos].get_pos()[1]:
                            tail_nodes[node_pos] = node
                    else:
                        tail_nodes[node_pos] = node

        for head_neigh in head_nodes.values():
            self.__head.attach(head_neigh)

        for tail_neigh in tail_nodes.values():
            self.__tail.attach(tail_neigh)

    def calculate(self, save_dir = "out_structs", outp_freq = 1):
        if os.path.isdir(save_dir):
            os.system(f"rm -r {save_dir}")
        os.mkdir(save_dir)

        self.__outp_freq = outp_freq
        self.__save_dir = save_dir
        starting_pe = self.__head.activate()

        energies = {node : float("inf") for node in self.flatten()}
        energies[self.__head] = 0

        priority_queue = [(0, self.__head, None)]
        scan_ctr = 0
        while priority_queue:
            current_energy, current, parent_id = heapq.heappop(priority_queue)

            if current_energy > energies[current]:
                continue

            Helper.print("-----------------------------------------------------")
            current.reset_lowest(parent_id)
            Helper.print("Lowest node:", current.get_id(), "Eng:", current_energy, "Parent:", parent_id, "Pos:", current.get_pos())
            scan_ctr += 1

            if current.is_tail():
                self.__tail.reset_tip()
                self.__head.reset_tip()
                return current_energy/current.get_surface_area()

            neighbors = current.get_neighbors()
            for neighbor in neighbors:
                if not neighbor.is_head() and neighbor.get_id() != parent_id:
                    path_energy = current_energy - neighbor.activate(parent = current)
                    Helper.print("Looking at node:", neighbor.get_id(), "Eng:", path_energy, "Pos:", neighbor.get_pos())
                    if path_energy < energies[neighbor]:
                        self.__paths[neighbor] = current
                        energies[neighbor] = path_energy
                        heapq.heappush(priority_queue, (path_energy, neighbor, current.get_id()))


        self.__tail.reset_tip()
        self.__head.reset_tip()
        return float("inf")

    def __rec_path_search(self, node):
        if node.is_head():
            return [node.get_pos()]
        else:
            ancestors = self.__rec_path_search(self.__paths[node])
            ancestors.insert(0, node.get_pos())
            return ancestors

    def get_paths(self):
        out = list()
        for node in self.__paths.keys():
            if node not in self.__paths.values():
                path = self.__rec_path_search(node)
                if node.is_tail():
                    path[0] = (path[1][0], path[0][1])
                path[-1] = (path[-2][0], path[-1][1])
                out.append(path)

        sorted_paths = sorted(out, key = lambda node_lst : len(node_lst), reverse = True)
        max_length = len(sorted_paths[0])
        return list(filter(lambda x: len(x)/max_length > 0.8, sorted_paths))


    def attach(self, coords, energy):
        disc_coords = self.__discretize(coords)

        if disc_coords in self.__node_hash:
            new_node = self.__node_hash[disc_coords]
        else:
            new_node = Node(tip = disc_coords, node_ctr = self.__node_ctr)
            self.__node_hash[disc_coords] = new_node
            Helper.print("Created a new node", self.__node_ctr)
            self.__node_ctr += 1

        new_node.energy += energy
        num_bins = int(np.floor(self.__dr/self.__grid_size))
        for x in np.linspace(disc_coords[0] - self.__grid_size*num_bins, disc_coords[0] + self.__grid_size*num_bins, num_bins*2 + 1):
            for y in np.linspace(disc_coords[1] - self.__grid_size*num_bins, disc_coords[1] + self.__grid_size*num_bins, num_bins*2 + 1):
                neigh_coords = self.__discretize((x, y))
                if neigh_coords != disc_coords and neigh_coords in self.__node_hash:
                    new_node.attach(self.__node_hash[neigh_coords])



        return new_node

        

class Node:
    def __init__(self, is_head = False, tip = None, units = "real", test_mode = False, node_ctr = 0, is_tail = False, timestep = 1):
        self.__id = node_ctr
        self.__active = False
        self.__is_head = is_head
        self.__is_tail = is_tail
        self.__neighbors = set()
        self.__tip = tip
        self.__surface_area = 0
        self.__test_mode = test_mode
        self.__old_tip = None
        self.__versions = dict()
        self.energy = 0

        if self.__is_head:
            self.__units = units
            if test_mode:
                if os.path.isdir("logs"):
                    os.system("rm -r logs")
                os.mkdir("logs")
                self.__lmp = lammps(cmdargs = ["-log", f"logs/log.0.lammps"])
            else:
                self.__lmp = lammps(cmdargs = ["-log", "none", "-screen", "none"])
            self.__system_parameters_initialization(units = units)
            self.__lmp.command(f"timestep {timestep}")
            if Data.structure_file:
                filename = Data.structure_file
            else:
                filename = glob.glob("glass_*.structure")[-1]
            self.__lmp.command(f"read_data {filename}")
            self.structure_file = os.path.abspath(filename)
            if Data.potfile:
                self.potfile = Data.potfile
            else:
                name_handle = re.search(r"(?<=glass_).+(?=\.structure)", filename).group()
                self.potfile = os.path.abspath(f"pot_{name_handle}.FF")
            

    #Setters

    def set_tip(self, coords):
        if self.__is_head or self.__is_tail:
            self.__tip = coords
        if not self.__old_tip and (self.__is_tail or self.__is_head):
            self.__old_tip = coords

    def reset_tip(self):
        if self.__is_tail or self.__is_head:
            self.__tip = self.__old_tip

    def attach(self, node):
        self.__neighbors.add(node)
        node.__neighbors.add(self)
        return self

    def set_parent(self, node):
        if node:
            node_pos = node.get_pos()
            if not (self.__tip[0] - node_pos[0]):
                if self.__tip[1] > node_pos[1]:
                    self.__theta = np.pi/2
                else:
                    self.__theta = -np.pi/2
            else:
                x = self.__tip[0] - node_pos[0]
                if x > 0:
                    self.__theta = np.arctan((self.__tip[1] - node_pos[1])/x)
                else:
                    self.__theta = np.arctan((self.__tip[1] - node_pos[1])/x) + np.pi


            if self.is_tail():
                self.__theta = np.pi/2
                par_pos = node.get_pos()
                self.set_tip((par_pos[0], self.get_pos()[1]))

            if node.is_head():
                self.__theta = np.pi/2
                node.set_tip((self.__tip[0], node.get_pos()[1]))

            #Helper.print(f"Node {self.__id} angle: {self.__theta*180/np.pi} with node {node.get_id()} as parent")
        self.__parent = node



    #Getters
    def get_parent_angle(self):
        if self.__parent and not self.__parent.is_head():
            return self.__parent.__theta
        else:
            return np.pi/2

    def get_pe(self):
        return self.__pe

    def get_id(self):
        return self.__id

    def get_parent_id(self):
        if self.__is_head:
            return None
        else:
            return self.__parent.__id

    def get_parent(self):
        return self.__parent

    def get_surface_area(self):
        return self.__surface_area

    def get_neighbors(self):
        return list(self.__neighbors)

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

    def is_tail(self):
        return self.__is_tail

    def is_active(self):
        return self.__active

    #Built-in reassignment
    def __str__(self):
        return f"id: {self.__id}, position: {self.get_pos()}"

    def __repr__(self):
        return f"{self.__id}"

    def __lt__(self, node):
        return self.__id < node.__id

    #Main behavior
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

    def reset_lowest(self, parent_id):
        if not self.is_head():
            for pid in self.__versions.keys():
                if pid == parent_id:
                    self.__surface_area, self.__theta = self.__versions[parent_id]
            self.__versions = dict()


    def __save_state(self):
        #Helper.print("Saving state of node:", self.__id, "with parent:", self.__parent_id)
        self.__versions[self.__parent.__id] = (self.__surface_area, self.__theta)

    def __reset(self):
        neighs = self.__neighbors
        theta = self.__theta
        old_tip = self.__old_tip
        parent = self.__parent
        versions = self.__versions
        self.__init__(is_head = self.__is_head, is_tail = self.__is_tail, node_ctr = self.__id, tip = self.__tip)
        self.__neighbors = neighs
        self.__theta = theta
        self.__parent = parent
        self.__old_tip = old_tip
        self.__versions = versions




    def activate(self, parent = None):
        if self.__is_head:
            if not self.__active:
                self.__lmp.command("clear")
                self.__system_parameters_initialization(units = self.__units)
                self.__lmp.command(f"atom_modify map yes")
                self.__lmp.command(f"read_data {self.structure_file}")
                self.__lmp.command(f"include {self.potfile}")
                Helper.print("Head node activated")
                self.__lmp.command("run 0")
        else:
            if self.__active:
                self.__save_state()
            self.set_parent(parent)
            if self.__active:
                self.__reset()

            self.__lmp = self.__parent.__lmp
            self.__prev_theta = self.get_parent_angle()

            box = self.__lmp.extract_box()

            par_pos = self.__parent.get_pos()
            y_dist = self.__tip[1] - par_pos[1]
            if self.__parent.is_head():
                y_dist -= (Data.old_bounds[0] - par_pos[1])
            if self.__is_tail:
                y_dist -= (self.__tip[1] - Data.old_bounds[1])

            dist = np.sqrt((self.__tip[0] - par_pos[0])**2 + y_dist**2)
            self.__surface_area = self.__parent.get_surface_area() + dist*(box[1][2] - box[0][2])
            

            try:
                grandparent = self.__parent.get_parent()
                del grandparent
            except:
                pass

        self.__active = True
        return self.energy
