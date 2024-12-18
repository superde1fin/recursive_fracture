from lammps import lammps
from classes.Storage import SystemParams, Helper, Data
import glob, os, sys, heapq, math, pickle, random
import numpy as np
import ctypes as ct
import regex as re
from mpi4py import MPI
from classes.type_sets import Holder
import mpmath as mp

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
proc_self_comm = MPI.COMM_SELF

class FracGraph:
    def __init__(self, connection_radius, error = 0.1, start_buffer = 0.5, test_mode = False, simulation_temp = 300, nono_table = ""):
        self.__nono_table = nono_table
        self.__step_energies = dict()
        self.__dr = connection_radius
        self.__test_mode = test_mode
        self.__head = Node(is_head = True, test_mode = self.__test_mode)
        self.__head.set_parent(None)
        self.__tail = Node(is_tail = True, test_mode = self.__test_mode, node_ctr = 1)
        self.__node_ctr = 2
        self.__paths = dict()
        self.__id_node_map = list()
        self.__grid_size = error/np.sqrt(2)
        #self.__sanity_list = list()
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

        self.__step_energies[self.__head.get_id()] = (0, 0)

    #Getters
    """
    def get_eng_bounds(self):
        return max(self.__step_energies.values()), min(self.__step_energies.values())
    """

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
        return tuple(grid_coords)

    def build_test(self, interactions):
        if interactions == "default":
            Data.type_groups = 1
        else:
            Data.type_groups = max(sum(interactions, ()))


        Data.initial_types = self.__head.get_lmp().extract_global("ntypes")
        self.__modify_potfile(interactions)
        self.__modify_struct()
        node = self.attach(coords = (20, 20))
        self.__head.attach(node)
        node.attach(self.__tail)



    def build(self, pivot_atom_type, num_neighs, interactions = "default"):
        if interactions == "default":
            Data.type_groups = 1
        else:
            Data.type_groups = max(sum(interactions, ()))

        Helper.mpi_print("Number of type groups:", Data.type_groups)


        Data.initial_types = self.__head.get_lmp().extract_global("ntypes")
        self.__modify_potfile(interactions)
        self.__modify_struct()
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

    def __node_info_transform(self, node_info):
        if isinstance(node_info, dict):
            return (node_info["path_energy"], node_info["node_id"], node_info["typeset_id"], node_info["parent_rank"], node_info["parent_id"], node_info["typeset_list"], node_info["surface_area"], node_info["theta"], node_info["pe"])
        elif isinstance(node_info, tuple):
            return {"path_energy" : node_info[0], "node_id" : node_info[1], "typeset_id" : node_info[2], "parent_rank" : node_info[3], "parent_id" : node_info[4], "typeset_list" : node_info[5], "surface_area" : node_info[6], "theta" : node_info[7], "pe": node_info[8]}
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
        self.__modify_potfile(interactions)
        self.__modify_struct()
        starting_pe = self.__head.activate(box = self.get_box())
        prev_node = self.__head
        for node_pos in path:
            node = Node(tip =  node_pos[:-1], node_ctr = self.__node_ctr)
            self.__node_ctr += 1
            node.activate(parent = prev_node, box = self.get_box())
            #node.get_lmp().command(f"write_data out.{self.__node_ctr}.struct")
            prev_node = node

        path_eng = self.__tail.activate(parent = node, box = self.get_box()) - starting_pe


        self.__tail.get_lmp().command(f"write_data {Data.non_inter_cutoff}_surface.structure")
        Helper.print("Surface area created:", 2*self.__tail.get_surface_area())
        Helper.print("Energy change:", path_eng)

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
            #Helper.print("Looking for neighbors of node:", prev_node.get_id())
            while not found_node:
                node = random.choice(neighbors)
                #Helper.print("Trying to pick node:", node.get_id())
                if not node.get_id() in visited and node.get_pos()[1] > prev_node.get_pos()[1]:
                    #Helper.print("PICKED")
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
            #self.__head.get_lmp().command(f"write_data test.{rank}.{i}.struct")
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
        #max_extension = max(self.__box[1][1] - disc_coords[1], self.__box[1][0] - disc_coords[0], self.__box[0][0] - disc_coords[0])
        max_extension = 2*self.__dr
        #print("Maximum extension radius:", max_extension)


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
                #print("New neigh coords:", neigh_coords)

                if neigh_coords != disc_coords and neigh_coords in self.__node_hash and neigh_coords[1] > disc_coords[1] and not self.__node_hash[neigh_coords].is_discarded():
                    neigh_node =  self.__node_hash[neigh_coords]
                    Helper.mpi_print("Adding neighbor:", neigh_node.get_id(), "at pos:", neigh_coords)
                    node.attach(neigh_node)
                    found_new = True


        return found_new



    def calculate(self, save_dir = "out_structs", outp_freq = 1):
        def dijkstra_step(energies, current_node, scan_ctr):
            scan_ctr = scan_ctr*size + rank
            box_x_side = self.__box[1][0] - self.__box[0][0]

            current = self.__id_node_map[current_node["node_id"]]

            current_pos = current.get_pos()

            if current_node["path_energy"] > energies[current_node["node_id"]]:
                print("Attomeped node porb:", -current_node["path_energy"], "Existing prob:", energies[current_node["node_id"]])
                return list(), [current_node["node_id"]]

            Helper.print("-----------------------------------------------------")
            Helper.print("Lowest node:", current_node["node_id"], "Path Probability:", current_node["path_energy"], "Parent:", current_node["parent_id"], "Pos:", current_pos, "Rank:", rank)
            current.reset_lowest(current_node["typeset_id"], current_node["parent_rank"], current_node["typeset_list"], current_node["surface_area"], current_node["theta"], self.__head, current_node["pe"])
            Helper.print("Saving datafile for node:", current_node["node_id"], "Ctr:", scan_ctr, "TID:", current.get_tid())
            current.get_lmp().command(f"write_data {save_dir}/out.{scan_ctr}.struct")

            if current.is_tail():
                self.__tail.reset_tip()
                self.__head.reset_tip()
                Helper.print("Rank:", rank, "Surface area created:", current.get_surface_area())
                #Helper.print("PE:", rank, self.__tail.get_pe())
                #Helper.print("Rank:", rank, "Energy change:", current_node["path_energy"])
                #return (self.__tail.get_pe() - self.__head.get_pe() + Data.boltzman*300*np.log(current_node["path_energy"])/(current.get_surface_area())
                return current_node["path_energy"], list()

            new_nodes = list()
            discarded = list()
            node_ids = list()
            probs = list()
            length_probs = 0
            partition = mp.mpf(0)
            neighbors = current.get_neighbors()
            box = self.get_box()
            for i, neighbor in enumerate(neighbors):
                neigh_id = neighbor.get_id()
                if not neighbor.is_head() and neigh_id != current_node["parent_id"]:
                    part_piece = neighbor.activate(box = box, parent = current)
                    node_ids.append(i)
                    #node_data.append((part_piece, neigh_id, neighbor.get_pe(), neighbor.get_tid(), neighbor.get_typeset_list(), neighbor.get_surface_area(), neighbor.get_theta()))
                    partition += part_piece
                    probs.append(part_piece)
                    length_probs += neighbor.get_cut_length()*part_piece

            Helper.print(partition)
            #Expectation value of the fracture propagation length
            if partition != 0:
                L = length_probs/partition
            else:
                L = 0

            #for non_norm_p, neigh_id, pe, tid, tid_list, surface_area, theta in node_data:
            for i, nid in enumerate(node_ids):
                #L = neighbors[nid].get_cut_length()
                neigh_id = neighbors[nid].get_id()
                neighbors[nid].set_surface_area(neighbors[nid].get_cut_length()*(box[1][2] - box[0][2]))
                if partition != 0:
                    step_prob = -probs[i]/partition
                else:
                    step_prob = 0
                path_prob = -energies[current_node["node_id"]]*step_prob
                Helper.print("Node:", neigh_id, energies[current_node["node_id"]], probs[i], step_prob, path_prob)

                neigh_coords = neighbors[nid].get_pos()

                if path_prob < energies[neigh_id] and neighbors[nid].get_pos()[1] > current_pos[1]:
                    self.__paths[neigh_id] = (current_node["node_id"], path_prob)
                    energies[neigh_id] = path_prob
                    self.__step_energies[neigh_id] = (float(step_prob), path_prob)
                    new_nodes.append({"path_energy" : path_prob, "node_id" : neigh_id, "typeset_id" : neighbors[nid].get_tid(), "parent_rank" : rank, "parent_id" : current_node["node_id"], "typeset_list" : neighbors[nid].get_typeset_list(), "surface_area" : neighbors[nid].get_surface_area(), "theta" : neighbors[nid].get_theta(), "pe" : neighbors[nid].get_pe()})
                else:
                    discarded.append(nid)
                    neighbors[nid].discard()

            if not new_nodes:
                new_nodes = [-1]

            return new_nodes, discarded


        if os.path.isdir(save_dir):
            Helper.action(os.system, f"rm -r {save_dir}")
        Helper.action(os.mkdir, save_dir)
        comm.Barrier()

        #Sanity check
#        self.__sanity_list = np.array(self.__sanity_list)
#        check_against = self.__sanity_list*size
#        all_lists = np.empty_like(check_against)
#        comm.Allreduce(self.__sanity_list, all_lists, op = MPI.SUM)
#        sane = np.allclose(all_lists, check_against)
#        if not sane:
#            raise RuntimeError("Different coordinates associated with the same node on different processes")
#        else:
#            Helper.mpi_print("All node positions are equivalent")

            

        self.__outp_freq = outp_freq
        self.__save_dir = save_dir
        self.__head.activate(box = self.get_box())

        scan_ctr = -1
        if rank == 0:
            energies = {node_id : 0 for node_id in range(self.__node_ctr)}
            head = self.__head
            energies[head.get_id()] = -1
            head_data = {"path_energy" : -1, "node_id" : head.get_id(), "typeset_id" : head.get_tid(), "parent_rank" : None, "parent_id" : None, "typeset_list" : list(), "surface_area" : head.get_surface_area(), "theta" : head.get_theta(), "pe": self.__head.get_pe()}
            priority_queue = [self.__node_info_transform(head_data)]
            done = False
            while not done and priority_queue:
                terminal_nodes = list()
                sent_nodes = list()
                own_node = self.__node_info_transform(heapq.heappop(priority_queue))
                heap_ctr = 1
                heap_size = len(priority_queue)
                while heap_ctr < heap_size and heap_ctr < size:
                    current_node = self.__node_info_transform(heapq.heappop(priority_queue))
                    comm.send(pickle.dumps(energies), dest = heap_ctr, tag = 0)
                    comm.send(pickle.dumps(current_node), dest = heap_ctr, tag = 1)
                    heap_ctr += 1
                    sent_nodes.append(current_node)

                scan_ctr += 1
                to_add, discarded = dijkstra_step(energies, own_node, scan_ctr)

                if not isinstance(to_add, list):
                    done = True
                elif to_add == [-1]:
                    terminal_nodes.append(own_node)
                    to_add = list()

                for i in range(1, heap_ctr):
                    Helper.print(f"Waiting for response from rank {i}")
                    answer = pickle.loads(comm.recv(source = i, tag = 2))
                    node_disc = pickle.loads(comm.recv(source = i, tag = 3))
                    for disc_id in node_disc:
                        self.__id_node_map[disc_id].discard()
                    if not done:
                        if isinstance(answer, list):
                            if answer == [-1]:
                                terminal_nodes.append(sent_nodes[i - 1])
                            else:
                                for node_info in answer:
                                    #Check that the path energy passed from a different processer is lower than the existing one for the newly calulated node.
                                    if node_info["path_energy"] < energies[node_info["node_id"]]:
                                        energies[node_info["node_id"]] = node_info["path_energy"]
                                        to_add.append(node_info)
                        else:
                            #to_add = answer
                            done = True


                if not done:
                    for node in to_add:
                        heapq.heappush(priority_queue, self.__node_info_transform(node))

                #Give a second chance to the nodes without any fit neighbors
                if terminal_nodes:
                    for node in terminal_nodes:
                        found_neighs = self.__extend_neighbors(self.__id_node_map[node["node_id"]])
                        if found_neighs:
                            heapq.heappush(priority_queue, self.__node_info_transform(own_node))


            for i in range(1, size):
                comm.send(None, dest = i, tag = 0)


        else:
            done = False
            to_add = list()
            while not done:
                energies = comm.recv(source = 0, tag = 0)
                if not energies:
                    done = True
                else:
                    energies = pickle.loads(energies)
                    current_node = pickle.loads(comm.recv(source = 0, tag = 1))
                    scan_ctr += 1
                    to_add, discarded = dijkstra_step(energies, current_node, scan_ctr)
                    Helper.print(f"Rank {rank} sent result to head rank")
                    comm.send(pickle.dumps(to_add), dest = 0, tag = 2)
                    comm.send(pickle.dumps(discarded), dest = 0, tag = 3)


        comm.Barrier()
        if rank == 0:
            for i in range(1, size):
                paths = comm.recv(source = i, tag = 0)
                steps = comm.recv(source = i, tag = 1)
                for node, parent_pair in paths.items():
                    if not node in self.__paths or parent_pair[-1] < self.__paths[node][-1]:
                        self.__paths[node] = parent_pair
                for node, eng_pair in steps.items():
                    if not node in self.__step_energies or eng_pair[-1] < self.__step_energies[node][-1]:
                        self.__step_energies[node] = eng_pair
            for key, value in self.__paths.items():
                self.__paths[key] = value[0]
            for key, value in self.__step_energies.items():
                self.__step_energies[key] = value[0]
        else:
            comm.send(self.__paths, dest = 0, tag = 0)
            comm.send(self.__step_energies, dest = 0, tag = 1)


        if not isinstance(to_add, list) and self.__tail.is_active():
            gathered = comm.gather((to_add, self.__tail.get_pe(), self.__tail.get_surface_area()), root = 0)
        else:
            self.__tail.reset_tip()
            self.__head.reset_tip()
            gathered = comm.gather(None, root = 0)

        if rank == 0:
            res = next((item for item in gathered if item is not None), None)
            if res is None:
                raise RuntimeError("Could not find a path from head to tail")
            prob, tail_eng, area = res
            Helper.print("Energy diff:", tail_eng - self.__head.get_pe())
            Helper.print("G:", 0.69*(tail_eng - self.__head.get_pe())/area)
            return prob
        else:
            return None


    @Helper.linear_func
    def __rec_path_search(self, node_id, path):
        to_add = (*self.__id_node_map[node_id].get_pos(), self.__step_energies[node_id])
        path.append(to_add)

        if self.__id_node_map[node_id].is_head():
            return
        else:
            self.__rec_path_search(self.__paths[node_id], path)

    @Helper.linear_func
    def get_paths(self):
        out = list()
        desired_rec_depth = len(self.__paths)*2
        if desired_rec_depth > sys.getrecursionlimit():
            sys.setrecursionlimit(desired_rec_depth)
        for node_id in self.__paths.keys():
            if node_id not in self.__paths.values():
                path = list()
                self.__rec_path_search(node_id, path)
                if self.__id_node_map[node_id].is_tail():
                    path[0] = (path[1][0], path[0][1], path[0][2])
                path[-1] = (path[-2][0], path[-1][1], path[-1][2])
                out.append(path)

        sorted_paths = sorted(out, key = lambda node_lst : len(node_lst), reverse = True)
        return sorted_paths
        #max_length = len(sorted_paths[0])
        #return list(filter(lambda x: len(x)/max_length > 0.8, sorted_paths))


    def attach(self, coords):
        disc_coords = self.__discretize(coords)

        if disc_coords in self.__node_hash:
            new_node = self.__node_hash[disc_coords]
        else:
            new_node = Node(tip = disc_coords, node_ctr = self.__node_ctr)
            self.__id_node_map.append(new_node)
            self.__node_hash[disc_coords] = new_node
            #self.__sanity_list.append(disc_coords[0])
            #self.__sanity_list.append(disc_coords[1])
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

    def __modify_struct(self):
        groups = Data.type_groups
        ntypes = Data.initial_types
        prev_name = self.__head.structure_file
        name = os.getcwd() + "/" + re.sub(r"(?<=.+)\.(?=[^\.]+$)", "_new.", prev_name.split("/")[-1])
        #name = re.sub(r"(?<=\/[^/]+)\.(?=.+$)", "_new.", prev_name)
        if not os.path.isfile(name):
            text = open(prev_name, 'r').read()
            text = re.sub(r"(?<=\s*)\d+(?=\s+atom types)", str(ntypes*groups), text)
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
            
        name = os.getcwd() + "/" + re.sub(r"(?<=.+)\.(?=[^\.]+$)", "_new.", self.__head.potfile.split("/")[-1])
        if not os.path.isfile(name):
            text = open(self.__head.potfile, 'r').read()
            new_potfile = open(name, 'w')

            #Check for hybrid potentials
            pair_style_re = re.compile(r"^pair_style\s+hybrid", re.MULTILINE)
            hybrid_handle = ""
            if pair_style_re.findall(text):
                hybrid_handle = "table"

            new_text = text + "\n\n#-------------------------\n\n"
            new_text += f"variable      nono_table_path string \"{self.__nono_table}\""
            #t is type of atom
            for t in range(1, ntypes + 1):
                #print("Looking at type:", t)
                #g is group of types
                for g in range(groups - 1):
                    #print("In group:", g)
                    mass_re = re.compile(f"^mass\s+{ntypes*g + t}\s+.+$", re.MULTILINE)
                    mass_line = mass_re.findall(new_text)[-1]
                    new_text = mass_re.sub(mass_line + '\n' + re.sub(f"(?<=^mass\s+){ntypes*g + t}(?=\s+.+$)", str(ntypes*(g + 1) + t), mass_line) + '\n', new_text)


                    #j is atom type greater than t (current type) used for pair combinations
                    for j in range(t, ntypes + 1):
                        #print("Paired with type:", j)
                        #print(f"^pair_coeff\s+{t}\s+{j}\s+.+$")
                        coeff_line = re.compile(f"^pair_coeff\s+{t}\s+{j}\s+.+$", re.MULTILINE).findall(new_text)[-1]
                        #Add a pair_coeff line within current group
                        new_text += '\n' + re.sub(f"(?<=^pair_coeff\s+){t}\s+{j}(?=\s+.+$)", f"{ntypes*(g + 1) + t} {ntypes*(g + 1) + j}", coeff_line) + f"\t#Groups ({g + 2}, {g + 2}) for types ({t}, {j})"

                        #Cycle through inra group interactions
                        for group_iter in range(g + 2, groups + 1):
                            if (g + 1, group_iter) in interactions:
                                new_text += '\n' + re.sub(f"(?<=^pair_coeff\s+){t}\s+{j}(?=\s+.+$)", f"{ntypes*g + t} {ntypes*(group_iter - 1) + j}", coeff_line) + f"\t#Groups ({g + 1}, {group_iter}) for types ({t}, {j})"
                                if t != j:
                                    new_text += '\n' + re.sub(f"(?<=^pair_coeff\s+){t}\s+{j}(?=\s+.+$)", f"{ntypes*(group_iter - 1) + t} {ntypes*g + j}", coeff_line) + f"\t#Groups ({group_iter}, {g + 1}) for types ({t}, {j})"
                            else:
                                tmp_line = re.sub(f"(?<=^pair_coeff\s+){t}\s+{j}(?=\s+.+$)", f"{ntypes*g + t} {ntypes*(group_iter - 1) + j}", coeff_line)
                                new_text += '\n' + re.sub(f"(?<=^pair_coeff\s+{ntypes*g + t}\s+{ntypes*(group_iter - 1) + j}).+", "\t" + hybrid_handle + "\t${nono_table_path}\tNoNo\t10", tmp_line) + f"\t#Groups ({g + 1}, {group_iter}) for types ({t}, {j})"
                                if t != j:
                                    tmp_line = re.sub(f"(?<=^pair_coeff\s+){t}\s+{j}(?=\s+.+$)", f"{ntypes*(group_iter - 1) + t} {ntypes*g + j}", coeff_line)
                                    new_text += '\n' + re.sub(f"(?<=^pair_coeff\s+{ntypes*(group_iter - 1) + t}\s+{ntypes*g + j}).+", "\t" + hybrid_handle + "\t${nono_table_path}\tNoNo\t10", tmp_line) + f"\t#Groups ({group_iter}, {g + 1}) for types ({t}, {j})"
                    
            general_type_re = re.compile(f"(?<=pair_coeff\s+\*\s+\*.+)(\s+\S+){{{ntypes}}}$", re.MULTILINE)
            if(general_type_re.search(new_text)):
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
        self.__discarded = False

        if self.__is_head:
            self.__units = units
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
            self.type_holder = Holder(self.__lmp)
            self.__theta = np.pi/2
            

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
            if node.type_holder.get_ntype_sets():
                node.type_holder.change_typeset(node.__typeset_id)
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
        if self.__parent:
            return self.__parent.__theta
        else:
            return np.pi/2

    def get_tid(self):
        return self.__typeset_id

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
        return list(self.__neighbors)

    def get_pos(self):
        return self.__tip

    def get_path_pe(self):
        return self.__path_pe

    def get_lowest_leaf(self):
        return self.__lowest_leaf

    def get_lmp(self):
        return self.__lmp

    def get_typeset_list(self):
        return self.type_holder.get_typeset_list()

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

    def path_back(self):
        if self.__theta + np.pi ==  self.__prev_theta:
            return True
        if self.__theta - np.pi == self.__prev_theta:
            return True
        return False

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

    def __visualization(self, thermo_step = 1, dump_step = 1):
        #self.__lmp.command(f"thermo {thermo_step}")
        #self.__lmp.command("thermo_style custom step temp etotal pe vol density pxx pyy pzz")
        #self.__lmp.command("thermo_modify flush yes")

        #Computes
        self.__lmp.command("compute pe_pa all pe/atom")
        if Data.use_pressure:
            self.__lmp.command("compute stress_pa all stress/atom NULL")
            self.__lmp.command("compute stress_total all reduce sum c_stress_pa[1]")

    def reset_lowest(self, typeset_id, parent_rank, typeset_list, surface_area, theta, head, pe):
        self.type_holder = head.type_holder
        self.__lmp = head.__lmp
        self.__pe = pe
        if not self.is_head():
            self.__surface_area = surface_area
            self.__theta = theta
            if parent_rank == rank:
                self.__typeset_id = typeset_id
                self.type_holder.change_typeset(self.__typeset_id)
            else:
                self.__typeset_id = self.type_holder.add_typeset(typeset_list)


    def activate(self, box, parent = None):
        if self.__is_head:
            if not self.__active:
                self.__lmp.command("clear")
                self.__system_parameters_initialization(units = self.__units)
                self.__lmp.command(f"atom_modify map yes")
                self.__lmp.command(f"read_data {self.structure_file}")
                self.__lmp.command(f"include {self.potfile}")
                self.__visualization()
                self.__typeset_id = 0
                Helper.mpi_print("Head node activated")
        else:
            self.set_parent(parent)

            self.__lmp = self.__parent.get_lmp()
            self.type_holder = self.__parent.type_holder
            self.__prev_theta = self.get_parent_angle()

            par_pos = self.__parent.get_pos()
            y_dist = self.__tip[1] - par_pos[1]
            x_dist = self.__tip[0] - par_pos[0]
            if self.__parent.is_head():
                y_dist = self.__tip[1] - Data.old_bounds[0]
                x__dist = 0
            if self.__is_tail:
                y_dist = Data.old_bounds[1] - par_pos[1]
                x__dist = 0

            self.__cut_length = np.sqrt(x_dist**2 + y_dist**2)
            #self.__surface_area = self.__parent.get_surface_area() + dist*(box[1][2] - box[0][2])

            my_atoms = np.array(self.__lmp.gather_atoms("x", 1, 3), dtype = ct.c_double).reshape((-1, 3))
            types = np.array(self.__lmp.gather_atoms("type", 0, 1), dtype = ct.c_int)
            self.box_side = box[1][0] - box[0][0]

            old_tid = self.type_holder.get_current()

            new_types = self.__new_types(my_atoms, types, self.__lmp.get_natoms())
            self.__typeset_id = self.type_holder.add_typeset(new_types, )
            self.type_holder.change_typeset(self.__typeset_id)
            Helper.print(f"Node {self.__id} activated at x = {round(self.__tip[0], 3)}, y = {round(self.__tip[1], 3)}, Type set id: {self.__typeset_id}, Old TID: {old_tid}, Rank: {rank}")
            

            try:
                grandparent = self.__parent.get_parent()
                del grandparent
            except:
                pass

        self.__lmp.command("run 0")
        self.__active = True
        self.__pe = self.__lmp.get_thermo("pe")

        if self.__is_head:
            return 1
        else:
            pot_diff = self.__pe - self.__parent.__pe
            print("Parent eng:", self.__parent.__pe, "Self eng:", self.__pe, "Pot diff:", pot_diff, -pot_diff/(Data.boltzman*SystemParams.simulation_temp))
            return mp.exp(-pot_diff/(Data.boltzman*SystemParams.simulation_temp))


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
                    res = 2
                    if res <= Data.type_groups:
                        return res
                elif (atom_pos[1] >= np.polyval(f02, x) and atom_pos[1] <= np.polyval(f12, x) and x <= x1 and x >= x3):
                    res = 3
                    if res <= Data.type_groups:
                        return res
                elif ((np.sqrt((x - x0)**2 + (atom_pos[1] - y0)**2) <= cutoff) and atom_pos[1] <= np.polyval(f02, x) and x >= x1):
                    res = 4
                    if res <= Data.type_groups:
                        return res
                elif ((np.sqrt((x - x0)**2 + (atom_pos[1] - y0)**2) <= cutoff) and atom_pos[1] <= np.polyval(f02, x) and x < x1):
                    res = 5
                    if res <= Data.type_groups:
                        return res
            # (-90, 0)
            elif self.__theta > -np.pi/2 and self.__theta < 0:
                if (atom_pos[1] >= np.polyval(f02, x) and atom_pos[1] <= np.polyval(f21, x) and atom_pos[1] <= np.polyval(f12, x) and atom_pos[1] >= np.polyval(f01, x)):
                    res = 2
                    if res <= Data.type_groups:
                        return res
                elif (atom_pos[1] <= np.polyval(f01, x) and atom_pos[1] >= np.polyval(f02, x) and atom_pos[1] <= np.polyval(f12, x) and atom_pos[1] >= np.polyval(f31, x)):
                    res = 3
                    if res <= Data.type_groups:
                        return res
                elif ((np.sqrt((x - x0)**2 + (atom_pos[1] - y0)**2) <= cutoff) and atom_pos[1] >= np.polyval(f01, x) and atom_pos[1] <= np.polyval(f21, x)):
                    res = 4
                    if res <= Data.type_groups:
                        return res
                elif ((np.sqrt((x - x0)**2 + (atom_pos[1] - y0)**2) <= cutoff) and atom_pos[1] <= np.polyval(f01, x) and atom_pos[1] >= np.polyval(f31, x)):
                    res = 5
                    if res <= Data.type_groups:
                        return res
            # 0
            elif self.__theta == 0:
                if (atom_pos[1] <= np.polyval(f21, x) and atom_pos[1] >= np.polyval(f01, x) and x >= x1 and x <= x0):
                    res = 2
                    if res <= Data.type_groups:
                        return res
                elif (atom_pos[1] <= np.polyval(f01, x) and atom_pos[1] >= np.polyval(f31, x) and x >= x1 and x <= x0):
                    res = 3
                    if res <= Data.type_groups:
                        return res
                elif ((np.sqrt((x - x0)**2 + (atom_pos[1] - y0)**2) <= cutoff) and atom_pos[1] >= np.polyval(f01, x) and x >= x0):
                    res = 4
                    if res <= Data.type_groups:
                        return res
                elif ((np.sqrt((x - x0)**2 + (atom_pos[1] - y0)**2) <= cutoff) and atom_pos[1] <= np.polyval(f01, x) and x >= x0):
                    res = 5
                    if res <= Data.type_groups:
                        return res
            # (0, 90)
            elif self.__theta > 0 and self.__theta < np.pi/2:
                if (atom_pos[1] <= np.polyval(f02, x) and atom_pos[1] <= np.polyval(f21, x) and atom_pos[1] >= np.polyval(f12, x) and atom_pos[1] >= np.polyval(f01, x)):
                    res = 2
                    if res <= Data.type_groups:
                        return res
                elif (atom_pos[1] <= np.polyval(f01, x) and atom_pos[1] <= np.polyval(f02, x) and atom_pos[1] >= np.polyval(f12, x) and atom_pos[1] >= np.polyval(f31, x)):
                    res = 3
                    if res <= Data.type_groups:
                        return res
                elif ((np.sqrt((x - x0)**2 + (atom_pos[1] - y0)**2) <= cutoff) and atom_pos[1] >= np.polyval(f01, x) and atom_pos[1] >= np.polyval(f02, x)):
                    res = 4
                    if res <= Data.type_groups:
                        return res
                elif ((np.sqrt((x - x0)**2 + (atom_pos[1] - y0)**2) <= cutoff) and atom_pos[1] <= np.polyval(f01, x) and atom_pos[1] >= np.polyval(f02, x)):
                    res = 5
                    if res <= Data.type_groups:
                        return res

            # 90
            elif self.__theta == np.pi/2:
                if (atom_pos[1] >= np.polyval(f12, x) and atom_pos[1] <= np.polyval(f02, x) and x <= x1 and x > x2):
                    res = 2
                    if res <= Data.type_groups:
                        return res
                elif (atom_pos[1] >= np.polyval(f12, x) and atom_pos[1] <= np.polyval(f02, x) and x > x1 and x < x3):
                    res = 3
                    if res <= Data.type_groups:
                        return res
                elif ((np.sqrt((x - x0)**2 + (atom_pos[1] - y0)**2) <= cutoff) and atom_pos[1] >= np.polyval(f02, x) and x <= x0):
                    res = 4
                    if res <= Data.type_groups:
                        return res
                elif ((np.sqrt((x - x0)**2 + (atom_pos[1] - y0)**2) <= cutoff) and atom_pos[1] >= np.polyval(f02, x) and x >= x0):
                    res = 5
                    if res <= Data.type_groups:
                        return res

            # (90, 180)
            elif self.__theta > np.pi/2 and self.__theta < np.pi:
                if (atom_pos[1] <= np.polyval(f02, x) and atom_pos[1] >= np.polyval(f21, x) and atom_pos[1] >= np.polyval(f12, x) and atom_pos[1] <= np.polyval(f01, x)):
                    res = 2
                    if res <= Data.type_groups:
                        return res
                elif (atom_pos[1] >= np.polyval(f01, x) and atom_pos[1] <= np.polyval(f02, x) and atom_pos[1] >= np.polyval(f12, x) and atom_pos[1] <= np.polyval(f31, x)):
                    res = 3
                    if res <= Data.type_groups:
                        return res
                elif ((np.sqrt((x - x0)**2 + (atom_pos[1] - y0)**2) <= cutoff) and atom_pos[1] <= np.polyval(f01, x) and atom_pos[1] >= np.polyval(f02, x)):
                    res = 4
                    if res <= Data.type_groups:
                        return res
                elif ((np.sqrt((x - x0)**2 + (atom_pos[1] - y0)**2) <= cutoff) and atom_pos[1] >= np.polyval(f01, x) and atom_pos[1] >= np.polyval(f02, x)):
                    res = 5
                    if res <= Data.type_groups:
                        return res
            # 180
            elif self.__theta == np.pi:
                if (atom_pos[1] >= np.polyval(f21, x) and atom_pos[1] <= np.polyval(f01, x) and x <= x1 and x >= x0):
                    res = 2
                    if res <= Data.type_groups:
                        return res
                elif (atom_pos[1] >= np.polyval(f01, x) and atom_pos[1] <= np.polyval(f31, x) and x <= x1 and x >= x0):
                    res = 3
                    if res <= Data.type_groups:
                        return res
                elif ((np.sqrt((x - x0)**2 + (atom_pos[1] - y0)**2) <= cutoff) and atom_pos[1] <= np.polyval(f01, x) and x <= x0):
                    res = 4
                    if res <= Data.type_groups:
                        return res
                elif ((np.sqrt((x - x0)**2 + (atom_pos[1] - y0)**2) <= cutoff) and atom_pos[1] >= np.polyval(f01, x) and x <= x0):
                    res = 5
                    if res <= Data.type_groups:
                        return res
            # (180, 270)
            elif self.__theta > np.pi and self.__theta < 3*np.pi/2:
                if (atom_pos[1] >= np.polyval(f21, x) and atom_pos[1] <= np.polyval(f01, x) and atom_pos[1] >= np.polyval(f02, x) and atom_pos[1] <= np.polyval(f12, x)):
                    res = 2
                    if res <= Data.type_groups:
                        return res
                elif (atom_pos[1] <= np.polyval(f31, x) and atom_pos[1] >= np.polyval(f01, x) and atom_pos[1] >= np.polyval(f02, x) and atom_pos[1] <= np.polyval(f12, x)):
                    res = 3
                    if res <= Data.type_groups:
                        return res
                elif ((np.sqrt((x - x0)**2 + (atom_pos[1] - y0)**2) <= cutoff) and atom_pos[1] <= np.polyval(f01, x) and atom_pos[1] <= np.polyval(f02, x)):
                    res = 4
                    if res <= Data.type_groups:
                        return res
                elif ((np.sqrt((x - x0)**2 + (atom_pos[1] - y0)**2) <= cutoff) and atom_pos[1] >= np.polyval(f01, x) and atom_pos[1] <= np.polyval(f02, x)):
                    res = 5
                    if res <= Data.type_groups:
                        return res
            # 270
            elif self.__theta == 3*np.pi/2:
                if (atom_pos[1] >= np.polyval(f02, x) and atom_pos[1] <= np.polyval(f12, x) and x < x1 and x >= x3):
                    res = 2
                    if res <= Data.type_groups:
                        return res
                elif (atom_pos[1] >= np.polyval(f02, x) and atom_pos[1] <= np.polyval(f12, x) and x > x1 and x <= x2):
                    res = 3
                    if res <= Data.type_groups:
                        return res
                elif ((np.sqrt((x - x0)**2 + (atom_pos[1] - y0)**2) <= cutoff) and atom_pos[1] <= np.polyval(f02, x) and x >= x1):
                    res = 4
                    if res <= Data.type_groups:
                        return res
                elif ((np.sqrt((x - x0)**2 + (atom_pos[1] - y0)**2) <= cutoff) and atom_pos[1] <= np.polyval(f02, x) and x < x1):
                    res = 5
                    if res <= Data.type_groups:
                        return res

            #Calculating angle between the atom and x-axis
            if not x - x1:
                if atom_pos[1] > y1:
                    phi = np.pi/2
                elif atom_pos[1] < y1:
                    phi = -np.pi/2
                else:
                    phi = -self.__prev_theta
            else:
                phi = np.arctan((atom_pos[1] - y1)/(x - x1))

            if x < x1:
                phi += np.pi

            ahead_prev = False
            behind_curr = False
            angle_diff = phi - self.__prev_theta
            if angle_diff > np.pi:
                angle_diff = 2*np.pi - angle_diff
            elif angle_diff < -np.pi:
                angle_diff = 2*np.pi + angle_diff
            if angle_diff >= -np.pi/2 and angle_diff <= np.pi/2:
                ahead_prev = True
            angle_diff = np.pi - phi + self.__theta
            if angle_diff > np.pi:
                angle_diff = 2*np.pi - angle_diff
            elif angle_diff < -np.pi:
                angle_diff = 2*np.pi + angle_diff
            if angle_diff >= -np.pi/2 and angle_diff <= np.pi/2:
                behind_curr = True


            if ahead_prev and behind_curr and (np.sqrt((x - x1)**2 + (atom_pos[1] - y1)**2) < cutoff) and (self.__theta - self.__prev_theta != np.pi) and (self.__prev_theta - self.__theta != np.pi):
                if self.__theta > self.__prev_theta and self.__theta < self.__prev_theta + np.pi or self.__theta < self.__prev_theta - np.pi:
                    res = 3
                    if res <= Data.type_groups:
                        return res
                if self.__theta < self.__prev_theta and self.__theta > self.__prev_theta - np.pi or self.__theta > self.__prev_theta + np.pi:
                    res = 2
                    if res <= Data.type_groups:
                        return res




        res = 1
        if res <= Data.type_groups:
            return res
