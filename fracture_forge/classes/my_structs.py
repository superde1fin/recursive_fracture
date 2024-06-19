from lammps import lammps
from classes.Storage import SystemParams, Helper, Data
import glob, os, sys
import numpy as np
import ctypes as ct
import regex as re
import heapq

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
        self.__node_hash = {self.__head.get_pos() : self.__head, self.__tail.get_pos() : self.__tail}

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
        self.__head.attach(self.__tail)
        """
        node = self.attach(coords = (30, 15))
        self.__head.attach(node)
        node1 = self.attach(coords = (27, 20))
        node.attach(node1)
        node2 = self.attach(coords = (33, 20))
        node.attach(node2)
        """



    def build(self, pivot_atom_type, num_neighs, interactions = "default"):
        if interactions == "default":
            Data.type_groups = 1
        else:
            Data.type_groups = max(sum(interactions, ()))

        Helper.print("Number of type groups:", Data.type_groups)


        Data.initial_types = self.__head.get_lmp().extract_global("ntypes")
        self.__modify_potfile(interactions)
        self.__modify_struct()
        head_lmp = self.__head.get_lmp()

        #Calculate simulation region sides
        sides = np.array([self.__box[1][0] - self.__box[0][0], self.__box[1][1] - self.__box[0][1], self.__box[1][2] - self.__box[0][2]])

        head_divs = int(np.ceil(sides[0]/self.__dr))
        head_step = sides[0]/head_divs
        head_nodes = dict()
        tail_nodes = dict()

        #Gather per-atom information
        positions = np.array(head_lmp.gather_atoms("x", 1, 3), dtype = ct.c_double).reshape((-1, 3))
        types = np.array(head_lmp.gather_atoms("type", 0, 1), dtype = ct.c_int)
        ids = np.array(head_lmp.gather_atoms("id", 0, 1), dtype = ct.c_int)

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

            Helper.print("Lowest node:", current.get_id(), "Eng:", current_energy, "Parent:", parent_id, "Pos:", current.get_pos())
            current.reset_lowest(parent_id)
            Helper.print("Saving datafile for node:", current.get_id(), "Ctr:", scan_ctr, "Ltid:", current.get_lmp().extract_global("current_typeset"), "TID:", current.get_tid())
            current.get_lmp().command(f"write_data {save_dir}/out.{scan_ctr}.struct")
            scan_ctr += 1

            if current.is_tail():
                self.__tail.reset_tip()
                self.__head.reset_tip()
                return current_energy/current.get_surface_area()

            neighbors = current.get_neighbors()
            for neighbor in neighbors:
                if not neighbor.is_head() and neighbor.get_id() != parent_id:
                    path_energy = neighbor.activate(parent = current) - starting_pe
                    #Helper.print("Looking at node:", neighbor.get_id(), "Eng:", path_energy, "Pos:", neighbor.get_pos())
                    if path_energy < energies[neighbor]:
                        self.__paths[neighbor] = current
                        energies[neighbor] = path_energy
                        heapq.heappush(priority_queue, (path_energy, neighbor, current.get_id()))
            current.deactivate()


        self.__tail.reset_tip()
        self.__head.reset_tip()
        return float("inf")

    def __rec_path_search(self, node):
        if node.is_head():
            return [node]
        else:
            ancestors = self.__rec_path_search(self.__paths[node])
            ancestors.insert(0, node)
            return ancestors

    def get_paths(self):
        out = list()
        for node in self.__paths.keys():
            if node not in self.__paths.values():
                out.append(self.__rec_path_search(node))

        return sorted(out, key = lambda node_lst : len(node_lst), reverse = True)


    def attach(self, coords):
        disc_coords = self.__discretize(coords)

        if disc_coords in self.__node_hash:
            new_node = self.__node_hash[disc_coords]
        else:
            new_node = Node(tip = disc_coords, node_ctr = self.__node_ctr)
            self.__node_hash[disc_coords] = new_node
            Helper.print("Created a new node", self.__node_ctr)
            self.__node_ctr += 1

        num_bins = int(np.floor(self.__dr/self.__grid_size))
        for x in np.linspace(disc_coords[0] - self.__grid_size*num_bins, disc_coords[0] + self.__grid_size*num_bins, num_bins*2 + 1):
            for y in np.linspace(disc_coords[1] - self.__grid_size*num_bins, disc_coords[1] + self.__grid_size*num_bins, num_bins*2 + 1):
                neigh_coords = self.__discretize((x, y))
                if neigh_coords != disc_coords and neigh_coords in self.__node_hash:
                    new_node.attach(self.__node_hash[neigh_coords])



        return new_node

    def __modify_struct(self):
        groups = Data.type_groups
        ntypes = Data.initial_types
        prev_name = self.__head.structure_file
        text = open(prev_name, 'r').read()
        text = re.sub(r"(?<=\s*)\d+(?=\s+atom types)", str(ntypes*groups), text)
        name = re.sub(r"(?<=\/[^/]+)\.(?=.+$)", "_new.", prev_name)
        for t in range(1, ntypes + 1):
            for g in range(groups - 1):
                mass_re = re.compile(fr"^{ntypes*g + t}\s+\d+\.\d+$", re.MULTILINE)
                mass_line = mass_re.findall(text)[-1]
                text = mass_re.sub(mass_line + "\n" + re.sub(r"^\d+(?=\s+)", str(t + ntypes*(g + 1)), mass_line), text)
                
        open(name, "w").write(text)
        self.__head.structure_file = name

        
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

        self.__head.potfile = name

        

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
            if node.__lmp.extract_global("ntype_sets"):
                node.__lmp.change_typeset(node.__typeset_id)
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



    def deactivate(self):
        if self.__active and not self.__is_head and self.__lmp.extract_global("current_typeset") != self.__typeset_id:
            self.__active = False
            self.__lmp.delete_typeset(self.__typeset_id)

    #Getters
    def get_parent_angle(self):
        if self.__parent and not self.__parent.is_head():
            return self.__parent.__theta
        else:
            return np.pi/2

    def get_tid(self):
        return self.__typeset_id

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
            self.__lmp.change_typeset(self.__typeset_id)
            prev_tid = self.__typeset_id
            for pid in self.__versions.keys():
                if pid == parent_id:
                    self.__typeset_id, self.__surface_area = self.__versions[parent_id]
                    Helper.print("Going back to type set:", self.__typeset_id)
                    self.__lmp.change_typeset(self.__typeset_id)
                    Helper.print("Removing typest:", prev_tid)
                    self.__lmp.delete_typeset(prev_tid)
                else:
                    Helper.print("Removing typest:",self.__versions[pid][0])
                    self.__lmp.delete_typeset(self.__versions[pid][0])


    def __save_state(self):
        #Helper.print("Saving state of node:", self.__id, "with parent:", self.__parent_id)
        self.__versions[self.__parent.__id] = (self.__typeset_id, self.__surface_area)

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
                self.__typeset_id = 0
                Helper.print("Head node activated")
        else:
            if self.__active:
                self.__save_state()
            self.set_parent(parent)
            if self.__active:
                self.__reset()

            self.__lmp = self.__parent.get_lmp()
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

            my_atoms = np.array(self.__lmp.gather_atoms("x", 1, 3), dtype = ct.c_double).reshape((-1, 3))
            types = np.array(self.__lmp.gather_atoms("type", 0, 1), dtype = ct.c_int)
            self.box_side = box[1][0] - box[0][0]

            old_tid = self.__lmp.extract_global("current_typeset")

            new_types = self.__new_types(my_atoms, types, self.__lmp.get_natoms())
            self.__typeset_id = self.__lmp.add_typeset(new_types)
            self.__lmp.change_typeset(self.__typeset_id)
            Helper.print(f"Node {self.__id} activated at x = {round(self.__tip[0], 3)}, y = {round(self.__tip[1], 3)}, Type set id: {self.__typeset_id}, Old TID: {old_tid}")
            

            try:
                grandparent = self.__parent.get_parent()
                del grandparent
            except:
                pass

        self.__lmp.command("run 0")
        self.__active = True
        self.__pe = self.__lmp.get_thermo("pe")
        return self.__pe


    def __new_types(self, my_atoms, types, natoms):
        prev_node = self.__parent.get_pos()
        new_types_lst = list()
        for i in range(natoms):
            float_pos = my_atoms[i]
            group = self.__near_surface(float_pos[:-1], prev_node = prev_node)
            if types[i] <= Data.initial_types:
                if group <= Data.type_groups:
                    new_type = types[i] + (group - 1)*Data.initial_types
                else:
                    new_type = types[i]
            elif types[i] > Data.initial_types and types[i] <= 3*Data.initial_types:
                if self.__theta > self.__prev_theta + np.pi/2 and group == 3 or self.__theta < self.__prev_theta -np.pi/2 and group == 2:
                    tp = types[i]%Data.initial_types
                    tp = tp if tp else tp + Data.initial_types
                    new_type = tp + (group - 1)*Data.initial_types
                else:
                    new_type = types[i]
            else:
                if group <= Data.type_groups:
                    tp = types[i]%Data.initial_types
                    tp = tp if tp else tp + Data.initial_types
                    new_type = tp + (group - 1)*Data.initial_types
                else:
                    new_type = types[i]

            new_types_lst.append(new_type)
        return new_types_lst



    def __near_surface(self, atom_pos, prev_node):
        cutoff = Data.non_inter_cutoff
        x0, y0 = self.__tip #Tip of the division vector

        #Tail of the division vector
        x1, y1 = prev_node

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
        #for x in [atom_pos[0]]:

            # -90
            if self.__theta == -np.pi/2:
                if (atom_pos[1] >= np.polyval(f02, x) and atom_pos[1] <= np.polyval(f12, x) and x > x1 and x <= x2):
                    return 2
                elif (atom_pos[1] >= np.polyval(f02, x) and atom_pos[1] <= np.polyval(f12, x) and x <= x1 and x >= x3):
                    return 3
                elif ((np.sqrt((x - x0)**2 + (atom_pos[1] - y0)**2) <= cutoff) and atom_pos[1] <= np.polyval(f02, x) and x >= x1):
                    return 4
                elif ((np.sqrt((x - x0)**2 + (atom_pos[1] - y0)**2) <= cutoff) and atom_pos[1] <= np.polyval(f02, x) and x < x1):
                    return 5
            # (-90, 0)
            elif self.__theta > -np.pi/2 and self.__theta < 0:
                if (atom_pos[1] >= np.polyval(f02, x) and atom_pos[1] <= np.polyval(f21, x) and atom_pos[1] <= np.polyval(f12, x) and atom_pos[1] >= np.polyval(f01, x)):
                    return 2
                elif (atom_pos[1] <= np.polyval(f01, x) and atom_pos[1] >= np.polyval(f02, x) and atom_pos[1] <= np.polyval(f12, x) and atom_pos[1] >= np.polyval(f31, x)):
                    return 3
                elif ((np.sqrt((x - x0)**2 + (atom_pos[1] - y0)**2) <= cutoff) and atom_pos[1] >= np.polyval(f01, x) and atom_pos[1] <= np.polyval(f21, x)):
                    return 4
                elif ((np.sqrt((x - x0)**2 + (atom_pos[1] - y0)**2) <= cutoff) and atom_pos[1] <= np.polyval(f01, x) and atom_pos[1] >= np.polyval(f31, x)):
                    return 5
            # 0
            elif self.__theta == 0:
                if (atom_pos[1] <= np.polyval(f21, x) and atom_pos[1] >= np.polyval(f01, x) and x >= x1 and x <= x0):
                    return 2
                elif (atom_pos[1] <= np.polyval(f01, x) and atom_pos[1] >= np.polyval(f31, x) and x >= x1 and x <= x0):
                    return 3
                elif ((np.sqrt((x - x0)**2 + (atom_pos[1] - y0)**2) <= cutoff) and atom_pos[1] >= np.polyval(f01, x) and x >= x0):
                    return 4
                elif ((np.sqrt((x - x0)**2 + (atom_pos[1] - y0)**2) <= cutoff) and atom_pos[1] <= np.polyval(f01, x) and x >= x0):
                    return 5
            # (0, 90)
            elif self.__theta > 0 and self.__theta < np.pi/2:
                if (atom_pos[1] <= np.polyval(f02, x) and atom_pos[1] <= np.polyval(f21, x) and atom_pos[1] >= np.polyval(f12, x) and atom_pos[1] >= np.polyval(f01, x)):
                    return 2
                elif (atom_pos[1] <= np.polyval(f01, x) and atom_pos[1] <= np.polyval(f02, x) and atom_pos[1] >= np.polyval(f12, x) and atom_pos[1] >= np.polyval(f31, x)):
                    return 3
                elif ((np.sqrt((x - x0)**2 + (atom_pos[1] - y0)**2) <= cutoff) and atom_pos[1] >= np.polyval(f01, x) and atom_pos[1] >= np.polyval(f02, x)):
                    return 4
                elif ((np.sqrt((x - x0)**2 + (atom_pos[1] - y0)**2) <= cutoff) and atom_pos[1] <= np.polyval(f01, x) and atom_pos[1] >= np.polyval(f02, x)):
                    return 5
            # 90
            elif self.__theta == np.pi/2:
                if (atom_pos[1] >= np.polyval(f12, x) and atom_pos[1] <= np.polyval(f02, x) and x <= x1 and x > x2):
                    return 2
                elif (atom_pos[1] >= np.polyval(f12, x) and atom_pos[1] <= np.polyval(f02, x) and x > x1 and x < x3):
                    return 3
                elif ((np.sqrt((x - x0)**2 + (atom_pos[1] - y0)**2) <= cutoff) and atom_pos[1] >= np.polyval(f02, x) and x <= x0):
                    return 4
                elif ((np.sqrt((x - x0)**2 + (atom_pos[1] - y0)**2) <= cutoff) and atom_pos[1] >= np.polyval(f02, x) and x >= x0):
                    return 5

            # (90, 180)
            elif self.__theta > np.pi/2 and self.__theta < np.pi:
                if (atom_pos[1] <= np.polyval(f02, x) and atom_pos[1] >= np.polyval(f21, x) and atom_pos[1] >= np.polyval(f12, x) and atom_pos[1] <= np.polyval(f01, x)):
                    return 2
                elif (atom_pos[1] >= np.polyval(f01, x) and atom_pos[1] <= np.polyval(f02, x) and atom_pos[1] >= np.polyval(f12, x) and atom_pos[1] <= np.polyval(f31, x)):
                    return 3
                elif ((np.sqrt((x - x0)**2 + (atom_pos[1] - y0)**2) <= cutoff) and atom_pos[1] <= np.polyval(f01, x) and atom_pos[1] >= np.polyval(f02, x)):
                    return 4
                elif ((np.sqrt((x - x0)**2 + (atom_pos[1] - y0)**2) <= cutoff) and atom_pos[1] >= np.polyval(f01, x) and atom_pos[1] >= np.polyval(f02, x)):
                    return 5
            # 180
            elif self.__theta == np.pi:
                if (atom_pos[1] >= np.polyval(f21, x) and atom_pos[1] <= np.polyval(f01, x) and x <= x1 and x >= x0):
                    return 2
                elif (atom_pos[1] >= np.polyval(f01, x) and atom_pos[1] <= np.polyval(f31, x) and x <= x1 and x >= x0):
                    return 3
                elif ((np.sqrt((x - x0)**2 + (atom_pos[1] - y0)**2) <= cutoff) and atom_pos[1] <= np.polyval(f01, x) and x <= x0):
                    return 4
                elif ((np.sqrt((x - x0)**2 + (atom_pos[1] - y0)**2) <= cutoff) and atom_pos[1] >= np.polyval(f01, x) and x <= x0):
                    return 5
            # (180, 270)
            elif self.__theta > np.pi and self.__theta < 3*np.pi/2:
                if (atom_pos[1] >= np.polyval(f21, x) and atom_pos[1] <= np.polyval(f01, x) and atom_pos[1] >= np.polyval(f02, x) and atom_pos[1] <= np.polyval(f12, x)):
                    return 2
                elif (atom_pos[1] <= np.polyval(f31, x) and atom_pos[1] >= np.polyval(f01, x) and atom_pos[1] >= np.polyval(f02, x) and atom_pos[1] <= np.polyval(f12, x)):
                    return 3
                elif ((np.sqrt((x - x0)**2 + (atom_pos[1] - y0)**2) <= cutoff) and atom_pos[1] <= np.polyval(f01, x) and atom_pos[1] <= np.polyval(f02, x)):
                    return 4
                elif ((np.sqrt((x - x0)**2 + (atom_pos[1] - y0)**2) <= cutoff) and atom_pos[1] >= np.polyval(f01, x) and atom_pos[1] <= np.polyval(f02, x)):
                    return 5
            # 270
            elif self.__theta == 3*np.pi/2:
                if (atom_pos[1] >= np.polyval(f02, x) and atom_pos[1] <= np.polyval(f12, x) and x < x1 and x >= x3):
                    return 2
                elif (atom_pos[1] >= np.polyval(f02, x) and atom_pos[1] <= np.polyval(f12, x) and x > x1 and x <= x2):
                    return 3
                elif ((np.sqrt((x - x0)**2 + (atom_pos[1] - y0)**2) <= cutoff) and atom_pos[1] <= np.polyval(f02, x) and x >= x1):
                    return 4
                elif ((np.sqrt((x - x0)**2 + (atom_pos[1] - y0)**2) <= cutoff) and atom_pos[1] <= np.polyval(f02, x) and x < x1):
                    return 5

            #Tail of the cut coverage at turns
            if (np.sqrt((x - x1)**2 + (atom_pos[1] - y1)**2) < cutoff) and (self.__theta - self.__prev_theta != np.pi) and (self.__prev_theta - self.__theta != np.pi):
                if (self.__prev_theta > self.__theta and self.__theta > self.__prev_theta - np.pi) or (self.__prev_theta < self.__theta and self.__theta > np.pi + self.__prev_theta):
                    return 2
                elif (self.__prev_theta < self.__theta and self.__theta < np.pi + self.__prev_theta) or (self.__prev_theta > self.__theta and self.__theta < self.__prev_theta - np.pi):
                    return 3



        return 1
        

