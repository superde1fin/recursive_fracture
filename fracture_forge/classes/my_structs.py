from lammps import lammps
from classes.Storage import SystemParams, Helper, Data
import glob, os, sys, heapq, random
import numpy as np
import ctypes as ct
import regex as re
from mpi4py import MPI
import mpmath as mp

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
proc_self_comm = comm

class FracGraph:
    def __init__(self, connection_radius, error = 0.1, start_buffer = 0.5, test_mode = False, simulation_temp = 300, nono_table = "", load_margin = 0, units = "real", minimize = False):
        self.__units = units
        self.__load_margin = load_margin
        self.__nono_table = nono_table
        self.__step_energies = dict()
        self.__dr = connection_radius
        self.__test_mode = test_mode
        self.__head = Node(is_head = True, test_mode = self.__test_mode, units = self.__units, minimize = minimize)
        self.__head.set_parent(None)
        self.__tail = Node(is_tail = True, test_mode = self.__test_mode, node_ctr = 1, units = self.__units)
        self.__node_ctr = 2
        self.__paths = dict()
        self.__id_node_map = list()
        self.__grid_size = error/np.sqrt(2)
        if self.__dr < self.__grid_size:
            self.__dr = self.__grid_size*1.5
            Helper.mpi_print("Reset probe radius to allow for fracture graph connectivity")
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

        Helper.mpi_print("Old bounds:", Data.old_bounds)
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
        self.__node_hash = {self.__head.get_pos() : self.__head, self.__tail.get_pos() : self.__tail}
        self.__id_node_map.append(self.__head)
        self.__id_node_map.append(self.__tail)

        self.__step_energies[self.__head.get_id()] = [(0, 0)]

    #Getters

    def get_box(self):
        atom_box = tuple(self.__box)
        atom_box[0][1] = Data.old_bounds[0]
        atom_box[1][1] = Data.old_bounds[1]
        return np.array(atom_box[:2])

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

    def build_test(self, interactions):
        if interactions == "default":
            Data.type_groups = 1
        else:
            Data.type_groups = max(sum(interactions, ()))


        Data.initial_types = self.__head.get_lmp().extract_global("ntypes")
        """
        node1 = self.attach(coords = (5, 5))
        node2 = self.attach(coords = (30, 20))
        node3 = self.attach(coords = (5, 39))
        self.__head.attach(node1)
        node1.attach(node2)
        node2.attach(node3)
        node3.attach(self.__tail)
        """
        self.__head.attach(self.__tail)

    def build_arbitrary(self, interactions = "default"):
        if interactions == "default":
            Data.type_groups = 1
        else:
            Data.type_groups = max(sum(interactions, ()))

        Helper.mpi_print("Number of type groups:", Data.type_groups)

        Data.initial_types = self.__head.get_lmp().extract_global("ntypes")
        head_lmp = self.__head.get_lmp()

        box = self.get_box()

        sides = box[1] - box[0]

        head_divs = int(np.ceil(sides[0]))
        head_step = sides[0]/head_divs
        head_nodes = dict()
        tail_nodes = dict()

        node_grid = self.__dr/np.sqrt(2)

        num_grid_points = np.floor(sides[:-1]/node_grid).astype(int)
        margins = (sides[:-1] - num_grid_points*node_grid)/2
        x_span = np.linspace(box[0][0] + margins[0], box[1][0] - margins[0], num_grid_points[0] + 1)
        y_span = np.linspace(box[0][1] + margins[1], box[1][1] - margins[1], num_grid_points[1] + 1)
        min_y = y_span[0]
        max_y = y_span[-1]
        np.random.shuffle(x_span)
        np.random.shuffle(y_span)
        for y in y_span:
            for x in x_span:
                node = self.attach(coords = np.array((x, y)))
                if y == min_y:
                    self.__head.attach(node)
                if y == max_y:
                    self.__tail.attach(node)
                disc_coords = node.get_pos()
                node_pos = int(np.floor((disc_coords[0] - box[0][0])/head_step))

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




    def build(self, pivot_atom_type, num_neighs, interactions = "default"):
        if interactions == "default":
            Data.type_groups = 1
        else:
            Data.type_groups = max(sum(interactions, ()))

        Helper.mpi_print("Number of type groups:", Data.type_groups)


        Data.initial_types = self.__head.get_lmp().extract_global("ntypes")
        head_lmp = self.__head.get_lmp()

        #Calculate simulation region sides
        sides = np.array([self.__box[1][0] - self.__box[0][0], self.__box[1][1] - self.__box[0][1], self.__box[1][2] - self.__box[0][2]])

        #head_divs = int(np.ceil(sides[0]/self.__dr))
        head_divs = int(np.ceil(sides[0]))
        head_step = sides[0]/head_divs
        head_nodes = dict()
        tail_nodes = dict()

        #Gather per-atom information
        positions = np.array(head_lmp.gather_atoms("x", 1, 3), dtype = ct.c_double).reshape((-1, 3))
        types = np.array(head_lmp.gather_atoms("type", 0, 1), dtype = ct.c_int)
        ids = np.array(head_lmp.gather_atoms("id", 0, 1), dtype = ct.c_int)

        Helper.mpi_print(f"Head y position: {self.__head.get_pos()[1]}")
        Helper.mpi_print("Scanning radius:", self.__dr)

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
                node = self.attach(coords = mid_pos[:-1])
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

        head_y_pos = self.__head.get_pos()[1]
        head_neigh_positions = np.array([node.get_pos()[1] - head_y_pos for node in head_nodes.values()])
        mean = np.mean(head_neigh_positions)
        std_dev = np.std(head_neigh_positions)
        for i, head_neigh in enumerate(head_nodes.values()):
            if std_dev != 0:
                Z = (head_neigh_positions[i] - mean)/std_dev
            else:
                Z = 0
            if Z < 2:
                self.__head.attach(head_neigh)

        tail_y_pos = self.__tail.get_pos()[1]
        tail_neigh_positions = np.array([tail_y_pos - node.get_pos()[1] for node in tail_nodes.values()])
        mean = np.mean(tail_neigh_positions)
        std_dev = np.std(tail_neigh_positions)
        for i, tail_neigh in enumerate(tail_nodes.values()):
            if std_dev != 0:
                Z = (tail_neigh_positions[i] - mean)/std_dev
            else:
                Z = 0
            if Z < 2:
                self.__tail.attach(tail_neigh)

    def __node_info_transform(self, node_info):
        if isinstance(node_info, dict):
            return (node_info["strongest_link"], node_info["node_id"], node_info["parent_id"], node_info["surface_area"], node_info["theta"], node_info["pe"], node_info["walls"])
        elif isinstance(node_info, tuple):
            return {"strongest_link" : node_info[0], "node_id" : node_info[1], "parent_id" : node_info[2], "surface_area" : node_info[3], "theta" : node_info[4], "pe": node_info[5], "walls": node_info[6]}
        else:
            raise RuntimeError(f"ERROR: Expected type dict or tuple not {type(node_info)}")

    @Helper.linear_func
    def recalculate_path(self, path, interactions = "default"):
        if interactions == "default":
            Data.type_groups = 1
        else:
            Data.type_groups = max(sum(interactions, ()))

        Data.initial_types = self.__head.get_lmp().extract_global("ntypes")

        Helper.mpi_print("Number of type groups:", Data.type_groups)
        starting_pe = self.__head.activate(box = self.get_box())
        prev_node = self.__head
        for node_pos in path:
            node = Node(tip =  node_pos[:-1], node_ctr = self.__node_ctr, units = self.__units)
            self.__node_ctr += 1
            node.activate(parent = prev_node, box = self.get_box())
            prev_node = node

        path_eng = self.__tail.activate(parent = node, box = self.get_box()) - starting_pe


        self.__tail.get_lmp().command(f"write_data {Data.non_inter_cutoff}_surface.structure")
        Helper.mpi_print("Surface area created:", 2*self.__tail.get_surface_area())
        Helper.mpi_print("Energy change:", path_eng)

        return path_eng/(2*self.__tail.get_surface_area())

    def __find_rand_path(self):
        node = self.__head
        prev_node = None
        head_stored = False
        visited = list()
        while not node.is_tail():
            path_eng = node.activate(box = self.get_box(), parent = prev_node)
            visited.append(node.get_id())
            if not head_stored:
                start_eng = path_eng
                head_stored = True
            neighbors = node.get_neighbors()
            found_node = False
            neigh_ctr = 0
            num_neghs = len(neighbors)
            prev_node = node
            while not found_node:
                node = random.choice(neighbors)
                if not node.get_id() in visited and node.get_pos()[1] > prev_node.get_pos()[1]:
                    found_node = True
                else:
                    neigh_ctr += 1
                    neighbors.remove(node)
                if neigh_ctr == num_neghs:
                    raise RuntimeError("Unavoidable loop in a random path, please increase the probe radius")
        path_eng = node.activate(box = self.get_box(), parent = prev_node)
        return (path_eng - start_eng)/(2*node.get_surface_area())


    def get_random_paths(self, num_paths):
        path_energies = list()
        to_do = num_paths // size
        if rank < num_paths%size:
            to_do += 1

        for i in range(to_do):
            path_energies.append(self.__find_rand_path())
            self.__head.deactivate(box = self.get_box())


        gathered_values = comm.gather(path_energies, root = 0)
        if rank == 0:
            return [item for sublist in gathered_values for item in sublist]
        else:
            return None

    def __extend_neighbors(self, node):
        print("Extenging neighbors for", node)
        disc_coords = node.get_pos()
        box_x_side = self.__box[1][0] - self.__box[0][0]
        found_new = False
        max_y = -float("inf")
        for neigh in node.get_neighbors():
            neigh_y = neigh.get_pos()[1]
            if neigh_y > max_y:
                max_y = neigh_y

        if max_y <= disc_coords[1]:
            prev_radius = self.__dr
        else:
            prev_radius = max_y - disc_coords[1]

        num_bins = int(np.floor(prev_radius/self.__grid_size))
        max_extension = 2*self.__dr


        print("Previous radius:", prev_radius)
        while prev_radius < max_extension and not found_new:
            prev_radius += self.__grid_size
            num_bins += 1
            print("Scanning radius:", prev_radius)

            new_neigh_layer = list()

            x = disc_coords[0] - self.__grid_size*num_bins
            for y in np.linspace(disc_coords[1], disc_coords[1] + self.__grid_size*(num_bins - 1), num_bins):
                new_neigh_layer.append((x, y))
            y = disc_coords[1] + self.__grid_size*(num_bins - 1)
            for x in np.linspace(disc_coords[0] - self.__grid_size*num_bins, disc_coords[0] + self.__grid_size*num_bins, 2*num_bins + 1):
                new_neigh_layer.append((x, y))
            x = disc_coords[0] + self.__grid_size*num_bins
            for y in np.linspace(disc_coords[1], disc_coords[1] + self.__grid_size*(num_bins - 1), num_bins):
                new_neigh_layer.append((x, y))

            for x, y in new_neigh_layer:
                neigh_coords = tuple(self.__trunc(np.array([x, y]), 3))

                if neigh_coords != disc_coords and neigh_coords in self.__node_hash and neigh_coords[1] > disc_coords[1] and not self.__node_hash[neigh_coords].is_discarded():
                    neigh_node =  self.__node_hash[neigh_coords]
                    Helper.mpi_print("Adding neighbor:", neigh_node.get_id(), "at pos:", neigh_coords)
                    node.attach(neigh_node)
                    found_new = True


        return found_new



    def calculate(self, save_dir = "out_structs", outp_freq = 1):
        node = self.__head
        box = self.get_box()
        node.activate(box = box)

        while not node.is_tail():
            current_pos = node.get_pos()
            neighbors = node.get_neighbors()
            selected_neighs = list()
            Z = 0
            num_accepted = 0
            for neigh in neighbors:
                if neigh.get_pos()[1] > current_pos[1]:
                    weight = neigh.activate(box = box, parent = node)
                    selected_neighs.append((weight, neigh))
                    Z += weight
                    num_accepted += 1

            cumul_prob = 0
            rand_selector = random.random()
            i = 0
            found_next = False
            while i < num_accepted and not found_next:
                p = selected_neighs[i][0]/Z
                if (rand_selector > cumul_prob) and (rand_selector <= cumul_prob + p):
                    node = selected_neighs[i][1]
                    found_next = True
                else:
                    cumul_prob += p

                i += 1


        if rank == 0:
            E = node.get_pe() - self.__head.get_pe()
            G = 0.69*(E)/node.get_surface_area()
            Helper.print("Energy diff:", E)
            Helper.print("G:", G)
                    



    @Helper.linear_func
    def __rec_path_search(self, node_id):
        if self.__id_node_map[node_id].is_head():
            return [[(*self.__id_node_map[node_id].get_pos(), self.__step_energies[node_id][0])]]

        local_paths = list()
        for i, parent in enumerate(self.__paths[node_id]):
            for par_path in self.__rec_path_search(parent):
                local_paths.append([(*self.__id_node_map[node_id].get_pos(), self.__step_energies[node_id][i])] + par_path)
        
        return local_paths


    @Helper.linear_func
    def get_paths(self):
        out = list()
        desired_rec_depth = len(self.__paths)*2
        if desired_rec_depth > sys.getrecursionlimit():
            sys.setrecursionlimit(desired_rec_depth)
        for node_id in self.__paths.keys():
            if node_id not in self.__paths.values():
                leaf_paths = self.__rec_path_search(node_id)
                for path in leaf_paths:
                    if self.__id_node_map[node_id].is_tail():
                        path[0] = (path[1][0], path[0][1], path[0][2])
                    path[-1] = (path[-2][0], path[-1][1], path[-1][2])
                out += leaf_paths

        sorted_paths = sorted(out, key = lambda node_lst : len(node_lst), reverse = True)
        return sorted_paths


    def attach(self, coords):
        disc_coords = self.__discretize(coords)

        if disc_coords in self.__node_hash:
            new_node = self.__node_hash[disc_coords]
        else:
            new_node = Node(tip = disc_coords, node_ctr = self.__node_ctr, units = self.__units)
            self.__id_node_map.append(new_node)
            self.__node_hash[disc_coords] = new_node
            Helper.mpi_print("Created a new node", self.__node_ctr, "at pos:", disc_coords)
            self.__node_ctr += 1


            num_bins = int(np.floor(self.__dr/self.__grid_size))
            for x in np.linspace(disc_coords[0] - self.__grid_size*num_bins, disc_coords[0] + self.__grid_size*num_bins, num_bins*2 + 1):
                for y in np.linspace(disc_coords[1] - self.__grid_size*num_bins, disc_coords[1] + self.__grid_size*num_bins, num_bins*2 + 1):
                    neigh_coords = self.__discretize((x, y))


                    if neigh_coords != disc_coords and neigh_coords in self.__node_hash:
                        neigh_node =  self.__node_hash[neigh_coords]
                        Helper.mpi_print("Adding neighbor:", neigh_node.get_id(), "at pos:", neigh_coords)
                        new_node.attach(neigh_node)



        return new_node

        
        

class Node:
    def __init__(self, is_head = False, tip = None, units = "real", test_mode = False, node_ctr = 0, is_tail = False, timestep = 1, minimize = False):
        self.__id = node_ctr
        self.__active = False
        self.__is_head = is_head
        self.__is_tail = is_tail
        self.__neighbors = set()
        self.__tip = tip
        self.__surface_area = 0
        self.__test_mode = test_mode
        self.__old_tip = None
        self.minimize = minimize
        self.__discarded = False
        self.wall_list = list()
        self.current_wall_count = 0

        if self.__is_head:
            self.__units = units
            self.current_wall_count = 0
            if test_mode:
                if os.path.isdir("logs"):
                    Helper.action(os.system, "rm -r logs")
                Helper.action(os.mkdir, "logs")
                comm.Barrier()
                self.__lmp = lammps(cmdargs = ["-log", f"logs/log.{rank}.lammps"], comm = proc_self_comm)
            else:
                self.__lmp = lammps(cmdargs = ["-log", "none", "-screen", "none"], comm = proc_self_comm)
            self.__system_parameters_initialization(units = units)
            self.__lmp.command(f"timestep {timestep}")
            if Data.structure_file:
                filename = Data.structure_file
            else:
                filename = glob.glob("glass_*.structure")[-1]
            self.structure_file = os.path.abspath(filename)
            if Data.potfile:
                self.potfile = Data.potfile
            else:
                name_handle = re.search(r"(?<=glass_).+(?=\.structure)", filename).group()
                self.potfile = os.path.abspath(f"pot_{name_handle}.FF")
            self.__lmp.command(f"read_data {filename}")
            self.__lmp.command(f"variable pot_dir string {'/'.join(self.potfile.split(r'/')[:-1])}/../")
            self.__theta = np.pi/2
        else:
            self.atom_positions = None

            

    #Setters
    def discard(self):
        self.__discarded = True

    def set_surface_area(self, dA):
        self.__surface_area = self.__parent.__surface_area + dA

    def deactivate(self):
        self.__active = False

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


            self.__lmp = node.get_lmp()
            self.__prev_theta = node.get_theta()
            self.wall_list = node.wall_list.copy()
            self.__lmp.scatter_atoms("x", ct.c_int(1), ct.c_int(3), (self.__lmp.get_natoms()*3*ct.c_double)(*node.atom_positions))
            self.minimize = node.minimize


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
    def get_atom_positions(self):
        if self.atom_positions is not None:
            return self.atom_positions
        else:
            return np.array(self.__lmp.gather_atoms("x", 1, 3), dtype = ct.c_double)


    def get_parent_angle(self):
        if self.__parent:
            return self.__parent.__theta
        else:
            return np.pi/2


    def get_pe(self):
        return self.__pe

    def get_id(self):
        return self.__id

    def get_theta(self):
        return self.__theta

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
        return sorted(list(self.__neighbors), key = lambda x: x.__id)

    def get_pos(self):
        return self.__tip

    def get_path_pe(self):
        return self.__path_pe

    def get_lowest_leaf(self):
        return self.__lowest_leaf

    def get_lmp(self):
        return self.__lmp


    def get_cut_length(self):
        return self.__cut_length

    #State functions
    def is_discarded(self):
        return self.__discarded

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
        self.__lmp.command("neighbor 0.0 bin")
        self.__lmp.command("neigh_modify every 1 delay 0")

    def reset_lowest(self, surface_area, theta, head, pe, walls):
        self.__lmp = head.__lmp
        self.__pe = pe
        self.current_wall_count = 0
        if not self.is_head():
            self.__surface_area = surface_area
            self.__theta = theta
            self.atom_positions = np.copy(head.atom_positions)
            self.__lmp.scatter_atoms("x", ct.c_int(1), ct.c_int(3), (self.__lmp.get_natoms()*3*ct.c_double)(*self.atom_positions))
            self.place_walls(walls)

    def take_walls_down(self):
        for i in range(self.current_wall_count):
            self.__lmp.command(f"region slab_{i} delete")
            self.__lmp.command(f"unfix wall_{i}")

    def activate(self, box, parent = None):
        if self.__is_head:
            if not self.__active:
                self.__lmp.command("clear")
                self.__system_parameters_initialization(units = self.__units)
                self.__lmp.command(f"atom_modify map yes")
                self.__lmp.command(f"read_data {self.structure_file}")
                self.__lmp.command(f"include {self.potfile}")
                Helper.mpi_print("Head node activated")
        else:
            self.set_parent(parent)

            par_pos = self.__parent.get_pos()
            y_dist = self.__tip[1] - par_pos[1]
            x_dist = self.__tip[0] - par_pos[0]
            if self.__parent.is_head():
                y_dist = self.__tip[1] - Data.old_bounds[0]
                x__dist = 0
            if self.__is_tail:
                y_dist = Data.old_bounds[1] - par_pos[1]
                x__dist = 0

            z_dim = box[1][2] - box[0][2]
            self.__cut_length = np.sqrt(x_dist**2 + y_dist**2)
            self.__surface_area = self.__parent.get_surface_area() + self.__cut_length*z_dim

            self.wall_list.append({"center": f"{self.__tip[0] - x_dist/2} {self.__tip[1] - y_dist/2} {(box[1][2] + box[0][2])/2}", "side1": f"{x_dist} {y_dist} 0", "side2": f"0 0 {z_dim}"})
            self.current_wall_count = self.__parent.current_wall_count + 1
            Helper.mpi_print(f"Node {self.__id} wall {self.current_wall_count} {self.wall_list[-1]}")
            self.add_wall(wall = self.wall_list[-1])



            try:
                grandparent = self.__parent.get_parent()
                del grandparent
            except:
                pass


        self.__lmp.command("run 0")
        my_prerelax = self.__lmp.get_thermo("pe") 
        if self.minimize:
            self.__lmp.command("minimize 1.0e-8 1.0e-8 100000 10000000")
        self.__active = True
        self.__pe = self.__lmp.get_thermo("pe")
        self.atom_positions = np.array(self.__lmp.gather_atoms("x", 1, 3), dtype = ct.c_double)

        if self.__is_head:
            return 1
        else:
            E = my_prerelax - self.__parent.get_pe()

            self.__lmp.command(f"region slab_{self.current_wall_count} delete")
            self.__lmp.command(f"unfix wall_{self.current_wall_count}")
            #self.__lmp.scatter_atoms("x", ct.c_int(1), ct.c_int(3), (self.__lmp.get_natoms()*3*ct.c_double)(*self.__parent.atom_positions))

        return mp.exp(-E/(Data.boltzman*SystemParams.simulation_temp))


    def place_walls(self, wall_list):
        for i, wall in enumerate(wall_list):
            self.add_wall(wall_num = i, wall = wall)
            self.current_wall_count += 1

    def add_wall(self, wall, wall_num = None):
        if wall_num is None:
            wall_num = self.current_wall_count

        self.__lmp.command(f"region slab_{wall_num} slab center {wall['center']} side1 {wall['side1']} side2 {wall['side2']}")
        self.__lmp.command(f"fix wall_{wall_num} all wall/ghost/region slab_{wall_num} -1")
