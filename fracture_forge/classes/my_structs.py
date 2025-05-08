from lammps import lammps
from classes.Storage import SystemParams, Helper, Data
import glob, os, sys, heapq, random, time
import numpy as np
import ctypes as ct
import regex as re
from mpi4py import MPI
import mpmath as mp
import gc

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

class FracGraph:
    def __init__(self, connection_radius, error = 0.1, start_buffer = 0.5, test_mode = False, simulation_temp = 300, units = "real", minimize = False):
        sys.stdout.flush()
        sys.stderr.flush()
        time.sleep(1)
        comm.Barrier()

        self.__units = units
        self.__step_energies = dict()
        self.__dr = connection_radius
        self.__test_mode = test_mode
        self.__minimize = minimize
        self.__simulation_temp = simulation_temp

        self.__head = Node(is_head = True, test_mode = self.__test_mode, units = self.__units, minimize = self.__minimize, simulation_temp = self.__simulation_temp)
        self.__head.set_parent(None)

        self.__tail = Node(is_tail = True, test_mode = self.__test_mode, node_ctr = 1, units = self.__units, minimize = self.__minimize, simulation_temp = self.__simulation_temp)

        self.__node_ctr = 2

        self.__grid_size = error/np.sqrt(2)
        if self.__dr < self.__grid_size:
            self.__dr = self.__grid_size*1.5
            Helper.mpi_print("Reset probe radius to allow for fracture graph connectivity", verbosity = 1)

        self.__head.activate()
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

        Helper.mpi_print("Old bounds:", Data.old_bounds, verbosity = 1)
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
        self.__path = [(*self.__head.get_pos(), self.__head.get_deltaE())]
        self.__tail.set_tip((x_side/2 + box[0][0], Data.old_bounds[1] + start_buffer))
        self.__node_hash = {self.__head.get_pos() : self.__head, self.__tail.get_pos() : self.__tail}

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

    def build_test(self):
        """
        #
        node1 = self.attach(coords = (5, 5))
        node2 = self.attach(coords = (30, 20))
        node3 = self.attach(coords = (5, 39))
        self.__head.attach(node1)
        node1.attach(node2)
        node2.attach(node3)
        node3.attach(self.__tail)
        #
        self.__head.set_tip((self.__box[0][0], Data.old_bounds[0]))
        self.__tail.set_tip((self.__box[1][0], Data.old_bounds[1]))
        self.__head.attach(self.__tail)
        #
        node1 = self.attach(coords = (10, 10))
        self.__head.attach(node1)
        node1.attach(self.__tail)
        #
        node1 = self.attach(coords = (20, 10))
        node2 = self.attach(coords = (10, 20))
        node3 = self.attach(coords = (30, 20))
        node4 = self.attach(coords = (20, 30))

        self.__head.attach(node1)
        node1.attach(node2)
        node1.attach(node3)
        node2.attach(node4)
        node3.attach(node4)
        node4.attach(self.__tail)
        #
        """
        node1 = self.attach(coords = (20, 10))
        node2 = self.attach(coords = (10, 20))
        node3 = self.attach(coords = (30, 20))
        node4 = self.attach(coords = (20, 30))

        self.__head.attach(node1)
        node1.attach(node2)
        node1.attach(node3)
        node2.attach(node4)
        node3.attach(node4)
        node4.attach(self.__tail)


    def build_arbitrary(self, grid_size = None):
        if grid_size is None:
            grid_size = self.__dr

        self.__grid_size = grid_size


        box = self.get_box()

        sides = (box[1] - box[0])[:-1]

        num_grid_points = np.floor(sides/self.__grid_size).astype(int)
        right_margins = sides - num_grid_points*self.__grid_size
        x_span = np.linspace(box[0][0], box[1][0] - right_margins[0], num_grid_points[0] + 1)
        y_span = np.linspace(Data.old_bounds[0], Data.old_bounds[1] - right_margins[1], num_grid_points[1] + 1)
        min_y = y_span[0]
        max_y = y_span[-1]
        if rank == 0:
            np.random.shuffle(x_span)
            np.random.shuffle(y_span)

        comm.Bcast(x_span, root = 0)
        comm.Bcast(y_span, root = 0)

        for y in y_span:
            for x in x_span:
                node = self.attach(coords = np.array((x, y)))
                if y == min_y:
                    self.__head.attach(node)
                if y == max_y:
                    self.__tail.attach(node)
        Helper.mpi_print("Head neighs:", self.__tail.get_neighbors(), verbosity = 4)
        Helper.mpi_print("Tail neighs:", self.__tail.get_neighbors(), verbosity = 4)

    def build(self):
        head_lmp = self.__head.get_lmp()

        sides = np.array(self.__box[1]) - np.array(self.__box[0])

        head_divs = int(np.ceil(sides[0]/self.__dr))
        head_step = sides[0]/head_divs
        head_nodes = dict()
        tail_nodes = dict()

        #Gather per-atom information
        positions = np.array(head_lmp.gather_atoms("x", 1, 3), dtype = ct.c_double).reshape((-1, 3))

        Helper.mpi_print(f"Head y position: {self.__head.get_pos()[1]}", verbosity = 2)
        Helper.mpi_print("Scanning radius:", self.__dr, verbosity = 2)

        #Cycle through the local ids of atoms with the oxygen type
        for atom_pos in positions:
            diffs = positions - atom_pos
            diffs -= np.around(diffs/sides)*sides

            distances = np.sum(diffs**2, axis = 1)
            sorted_indices = np.argsort(distances)
            proximity_sort = positions[sorted_indices][1:]
            distances = np.sqrt((distances[sorted_indices])[1:])

            done = False
            atom_iter = 0
            min_dist = distances[atom_iter]
            while atom_iter < proximity_sort.shape[0] and not done:
                if distances[atom_iter] - min_dist > self.__dr:
                    done = True
                else:
                    node = self.attach(((proximity_sort[atom_iter] + atom_pos)/2)[:-1])
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

                    atom_iter += 1

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



    @Helper.linear_func
    def recalculate_path(self, path, interactions = "default"):
        if interactions == "default":
            Data.type_groups = 1
        else:
            Data.type_groups = max(sum(interactions, ()))

        Data.initial_types = self.__head.get_lmp().extract_global("ntypes")

        starting_pe = self.__head.activate(box = self.get_box())
        prev_node = self.__head
        for node_pos in path:
            node = Node(tip =  node_pos[:-1], node_ctr = self.__node_ctr, units = self.__units, minimize = self.__minimize, test_mode = self.__test_mode, simulation_temp = self.__simulation_temp)
            self.__node_ctr += 1
            node.activate(parent = prev_node, box = self.get_box())
            prev_node = node

        path_eng = self.__tail.activate(parent = node, box = self.get_box()) - starting_pe


        self.__tail.get_lmp().command(f"write_data {Data.non_inter_cutoff}_surface.structure")
        Helper.mpi_print("Surface area created:", 2*self.__tail.get_surface_area(), verbosity = 1)
        Helper.mpi_print("Energy change:", path_eng, verbosity = 1)

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


    def calculate(self):
        node = self.__head

        pathE = 0

        while not node.is_tail():
            Helper.mpi_print("-------", verbosity = 1)
            current_pos = node.get_pos()
            neighbors = node.get_neighbors()
            selected_neighs = list()
            Z = 0
            num_accepted = 0
            for neigh in neighbors:
                if neigh.get_pos()[1] > current_pos[1]:
                    weight = neigh.activate(parent = node)
                    selected_neighs.append((weight, neigh))
                    Z += weight
                    num_accepted += 1

            if not num_accepted:
                raise RuntimeError("Node without neighbors. Increase spanning radius (Keep in mind that more unrealistically large cuts will be available) or use arbitrary grid.")

            cumul_prob = 0
            if rank == 0:
                rand_selector = random.random()
            else:
                rand_selector = None

            rand_selector = comm.bcast(rand_selector, root = 0)
            i = 0
            found_next = False
            while i < num_accepted and not found_next:
                p = selected_neighs[i][0]/Z
                if (rand_selector > cumul_prob) and (rand_selector <= cumul_prob + p):
                    node = selected_neighs[i][1]
                    found_next = True
                    Helper.mpi_print("Selected Node:", node, verbosity = 1)
                else:
                    cumul_prob += p

                i += 1
            node_deltaE = node.get_deltaE()
            pathE += node_deltaE
            self.__path.append((*node.get_pos(), node_deltaE))


        crack_surface_area = (Data.units_data[self.__units]["length"]**2)*node.get_surface_area()
        experimental_surface_area = (Data.units_data[self.__units]["length"]**2)*(Data.old_bounds[1] - Data.old_bounds[0])*(self.__box[1][2] - self.__box[0][2]) 

        Helper.mpi_print("Surface Area (m^2)", crack_surface_area, verbosity = 1)
        Helper.mpi_print("Experimental Area (m^2)", experimental_surface_area, verbosity = 2)
        
        self.__head.wall_list = node.wall_list.copy()
        energy_difference = Data.units_data[self.__units]["energy"]*(self.__head.reset_lowest(reset_head = True) - self.__head.get_pe())

        energy_release_rate = energy_difference/crack_surface_area
        experimental_energy_release_rate = energy_difference/experimental_surface_area
        step_eng_rel = Data.units_data[self.__units]["energy"]*pathE/crack_surface_area
        exp_step_G = Data.units_data[self.__units]["energy"]*pathE/experimental_surface_area

        Helper.mpi_print("Energy diff (J):", energy_difference, verbosity = 1)
        Helper.mpi_print("Step energy diff (J):", pathE, verbosity = 1)
        Helper.mpi_print("G (J/m^2):", energy_release_rate, verbosity = 1)
        Helper.mpi_print("Experimental G (J/m^2):", experimental_energy_release_rate, verbosity = 2)
        Helper.mpi_print("Step G (J/m^2):", step_eng_rel, verbosity = 2)
        Helper.mpi_print("EStep G (J/m^2):", exp_step_G, verbosity = 2)

        if os.path.isdir(".states"):
            Helper.action(os.system, "rm -r .states")
                    




    @Helper.linear_func
    def get_path(self):
        self.__path[0] = (self.__path[1][0], self.__path[0][1], self.__path[0][2])
        return self.__path[::-1]


    def attach(self, coords):
        disc_coords = self.__discretize(coords)

        if disc_coords in self.__node_hash:
            new_node = self.__node_hash[disc_coords]
        else:
            new_node = Node(tip = disc_coords, node_ctr = self.__node_ctr, units = self.__units, minimize = self.__minimize, test_mode = self.__test_mode, simulation_temp = self.__simulation_temp)
            self.__node_hash[disc_coords] = new_node
            Helper.mpi_print("Created a new node", self.__node_ctr, "at pos:", disc_coords, verbosity = 2)
            self.__node_ctr += 1


            num_bins = int(np.floor(self.__dr/self.__grid_size))
            for x in np.linspace(disc_coords[0] - self.__grid_size*num_bins, disc_coords[0] + self.__grid_size*num_bins, num_bins*2 + 1):
                for y in np.linspace(disc_coords[1] - self.__grid_size*num_bins, disc_coords[1] + self.__grid_size*num_bins, num_bins*2 + 1):
                    neigh_coords = self.__discretize((x, y))


                    if neigh_coords != disc_coords and neigh_coords in self.__node_hash:
                        neigh_node =  self.__node_hash[neigh_coords]
                        Helper.mpi_print("Adding neighbor:", neigh_node.get_id(), "at pos:", neigh_coords, verbosity = 4)
                        new_node.attach(neigh_node)



        return new_node

        
        

class Node:
    def __init__(self, is_head = False, tip = None, units = "real", test_mode = False, node_ctr = 0, is_tail = False, timestep = 1, minimize = False, simulation_temp = 300):
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
        self.__temp = simulation_temp
        self.__units = units
        self.__test_mode = test_mode
        self.__minimize = minimize
        self.__E = 0

        if self.__is_head:
            self.__theta = np.pi/2
            if os.path.isdir(".states"):
                Helper.action(os.system, "rm -r .states")
            Helper.action(os.mkdir, ".states")
            if self.__test_mode:
                if os.path.isdir("logs"):
                    Helper.action(os.system, "rm -r logs")
                Helper.action(os.mkdir, "logs")

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


    #Getters

    def get_deltaE(self):
        return self.__deltaE

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

    def get_energy_change(self):
        return self.__E

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
    def __post_structure(self):
        self.__lmp.command(f"include {self.potfile}")
        self.__lmp.command(f"velocity all create {self.__temp} {random.randint(0, 1000000)}")
        #self.__lmp.command("min_style sd")
        #self.__lmp.commands_string("""# Define variables for simulation box limits
        ## Define the radius of the cylinders
        #variable radius equal 10.0

        ## Create the first cylindrical region at (xlo, ylo)
        #region cyl1 cylinder z $(xlo) $(ylo) $(v_radius) $(zlo) $(zhi)

        ## Create the second cylindrical region at (xhi, ylo)
        #region cyl2 cylinder z $(xhi) $(ylo) $(v_radius) $(zlo) $(zhi)

        #group cyl1 region cyl1
        #group cyl2 region cyl2

        #fix pull all addforce 0.0 0.0 10
        #""")
        if self.__test_mode:
            self.__lmp.command("thermo_style custom step temp etotal pe vol density press")
            self.__lmp.command("thermo 1")

    def __pre_structure(self):
        self.__lmp.command(f"units {self.__units}")
        SystemParams.units = self.__units
        self.__lmp.command("atom_style charge")
        self.__lmp.command("boundary p p p")
        self.__lmp.command("comm_modify mode single vel yes")
        self.__lmp.command("neighbor 2.0 bin")
        #self.__lmp.command("neigh_modify every 1 delay 0 check no")
        self.__lmp.command("neigh_modify every 1 delay 0")
        self.__lmp.command("atom_modify map yes")
        self.__lmp.command(f"variable pot_dir string {'/'.join(self.potfile.split(r'/')[:-1])}/../")
        #self.__lmp.command("min_modify dmax 0.01")

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


            self.__prev_theta = node.get_theta()
            self.wall_list = node.wall_list.copy()
            self.__start_lammps()
            self.__lmp.command(f"read_restart .states/system.{node.__id}.state")
            self.__post_structure()
            for i, wall in enumerate(self.wall_list):
                self.add_wall(wall, i + 1)

            self.__lmp.command("run 0")
            self.__parent_pe  = self.__lmp.get_thermo('pe')
            Helper.mpi_print(f"AFTER RESET ENG({node.__id}): {self.__parent_pe}")


            if self.is_tail():
                self.__theta = np.pi/2
                par_pos = node.get_pos()
                self.set_tip((par_pos[0], self.get_pos()[1]))

            if node.is_head():
                self.__theta = np.pi/2
                node.set_tip((self.__tip[0], node.get_pos()[1]))

            #Helper.print(f"Node {self.__id} angle: {self.__theta*180/np.pi} with node {node.get_id()} as parent")
        self.__parent = node

    def reset_lowest(self, reset_head = False):
        if not self.is_head() or reset_head:
            Helper.mpi_print("Resetting lowest:", self.__id)
            self.__start_lammps()
            self.__lmp.command(f"read_restart .states/system.{self.__id}.state")
            self.__post_structure()
            for i, wall in enumerate(self.wall_list):
                self.add_wall(wall, i + 1)

            self.__lmp.command("run 0")
            after_reset_pe = self.__lmp.get_thermo('pe')
            Helper.mpi_print(f"AFTER RESET ENG({self.__id}): {after_reset_pe}")
            return after_reset_pe



    def __start_lammps(self):
        Helper.mpi_print("Starting lammps for node", self.__id)
        if self.__test_mode:
            self.__lmp = lammps(cmdargs = ["-log", f"logs/log.{self.__id}.lammps"], comm = comm)
        else:
            self.__lmp = lammps(cmdargs = ["-log", "none", "-screen", "none"], comm = comm)

        self.__pre_structure()



    def activate(self, parent = None):
        if self.__is_head:
            if not self.__active:
                self.__start_lammps()
                self.__lmp.command(f"read_data {self.structure_file}")
                self.__post_structure()
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

            box = self.__lmp.extract_box()
            z_dim = box[1][2] - box[0][2]
            self.__cut_length = np.sqrt(x_dist**2 + y_dist**2)
            self.__surface_area = self.__parent.get_surface_area() + self.__cut_length*z_dim

            self.wall_list.append({"center": f"{self.__tip[0] - x_dist/2} {self.__tip[1] - y_dist/2} {(box[1][2] + box[0][2])/2}", "side1": f"{x_dist} {y_dist} 0", "side2": f"0 0 {z_dim}"})
            self.current_wall_count = self.__parent.current_wall_count + 1
            self.add_wall(wall = self.wall_list[-1])
            #self.__lmp.command("compute paf all property/atom fx fy fz")
            #self.__lmp.command(f"dump atm_dump all custom 1 pos.{self.__id}.dump id type x y z c_paf[1] c_paf[2] c_paf[3]")



        self.__lmp.command("run 0")
        self.prerelax_pe = self.__lmp.get_thermo("pe") 
        Helper.mpi_print(f"BEFORE MIN ENG({self.__id}): {self.prerelax_pe}")
        if self.__minimize and not self.__is_tail:
            self.__lmp.command("minimize 1.0e-8 1.0e-8 100000 10000000")

        self.__active = True
        self.__pe = self.__lmp.get_thermo("pe")
        Helper.mpi_print(f"AFTER MIN ENG({self.__id}): {self.__pe}")

        if not self.__is_tail:
            self.__lmp.command(f"write_restart .states/system.{self.__id}.state")
            if not self.__is_head:
                self.__lmp.close()
                del self.__lmp
                gc.collect()


        if self.__is_head:
            self.__deltaE = 0
            Helper.mpi_print(f"Node {self.__id}, activated (HEAD)")
            Helper.mpi_print("Selected Node:", self)
        else:
            #self.__lmp.close()
            self.__deltaE = self.prerelax_pe - self.__parent_pe
            self.__E = self.__parent.__E + self.__deltaE

            Helper.mpi_print(f"Node {self.__id}, activated at x = {round(self.__tip[0], 3)}, y = {round(self. __tip[1], 3)}, Parent: {self.__parent.__id}, dE: {self.__deltaE}")

        return mp.exp(-self.__deltaE/(Data.boltzman*self.__temp))


    def place_walls(self, wall_list):
        for i, wall in enumerate(wall_list):
            self.add_wall(wall_num = i, wall = wall)
            self.current_wall_count += 1

    def add_wall(self, wall, wall_num = None):
        if wall_num is None:
            wall_num = self.current_wall_count

        if wall_num != 1:
            append = "append wall_1"
        else:
            append = ""

        Helper.mpi_print(f"Adding wall {wall_num}", verbosity = 5)

        self.__lmp.command(f"region slab_{wall_num} slab center {wall['center']} side1 {wall['side1']} side2 {wall['side2']}")
        self.__lmp.command(f"fix wall_{wall_num} all wall/ghost/region slab_{wall_num} {append}")
