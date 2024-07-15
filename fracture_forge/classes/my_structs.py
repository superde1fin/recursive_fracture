from lammps import lammps
from classes.Storage import SystemParams, Helper, Data
import glob, os, sys, random
import numpy as np
import ctypes as ct
import regex as re
import heapq
from scipy import interpolate
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

class FracGraph:
    def __init__(self, error = 0.1, simulation_temp = 300, test_mode = False, units = "real"):
        #Testing
        #self.__grid_size = error/np.sqrt(2)
        self.__grid_size = 1

        self.__lmp = self.__read_structure(units = units, test_mode = test_mode)
        

        self.__original_box = np.array(self.__lmp.extract_box()[:2])
        self.__box = self.__detect_surface_regions()
        self.__sides = self.__box[1] - self.__box[0]

        num_divs = np.ceil(self.__sides/self.__grid_size).astype(int)
        self.__energy_matrix = np.empty(num_divs, dtype = Cell)
        for index in np.ndindex(tuple(num_divs)):
            self.__energy_matrix[index] = Cell(position = index)

    def __detect_surface_regions(self):
        mins = np.full((3,), np.inf)
        maxs = np.full((3,), -np.inf)
        my_atoms = np.array(self.__lmp.gather_atoms("x", 1, 3), dtype = ct.c_double).reshape((-1, 3))
        for atom in my_atoms:
            for i in range(3):
                if atom[i] < mins[i]:
                    mins[i] = atom[i]
                if atom[i] > maxs[i]:
                    maxs[i] = atom[i]

        result = np.array((mins, maxs))
        Helper.mpi_print("Atomic bounds:", result)
        return result

    def __read_structure(self, units, test_mode):
        if test_mode:
            if os.path.isdir("logs"):
                os.system("rm -r logs")
            lmp = lammps(cmdargs = ["-log", f"log.lammps"])
        else:
            lmp = lammps(cmdargs = ["-log", "none", "-screen", "none"])
        lmp.command(f"units {units}")
        lmp.command("atom_style charge")
        lmp.command("boundary p p p")
        lmp.command("comm_modify mode single vel yes")
        lmp.command("neighbor 2.0 bin")
        lmp.command("neigh_modify every 1 delay 0")
        if Data.structure_file:
            filename = Data.structure_file
        else:
            filename = glob.glob("glass_*.structure")[-1]
        self.__structure_file = os.path.abspath(filename)
        lmp.command(f"read_data {self.__structure_file}")
        if Data.potfile:
            self.potfile = Data.potfile
        else:
            name_handle = re.search(r"(?<=glass_).+(?=\.structure)", filename).group()
            self.__potfile = os.path.abspath(f"pot_{name_handle}.FF")
        lmp.command(f"include {self.__potfile}")

        lmp.command("run 0")
        return lmp

    def __cmap(self, coords):
        return np.floor(np.asarray(coords - self.__box[0])/self.__grid_size).astype(int)

    def __str__(self):
        return str(list(self.__energy_matrix.flatten()))

    @Helper.linear_func
    def save(self, name = "energy_landscape.csv"):
        with open(name, "w") as f:
            shape = self.__energy_matrix.shape
            f.write(f"""{shape[0]} #X shape
{shape[1]} #Y shape
{shape[2]} #Z shape
{self.__grid_size} #Grid size
""")
            f.write("\n".join(self.__energy_matrix.flatten().astype(str)))

    def load_landscape(self, name = "energy_landscape.csv"):
        x, y, z, self.__grid_size, *flat_landscape = np.loadtxt(name)
        self.__energy_matrix = np.array(flat_landscape).reshape(int(x), int(y), int(z))

    
    def __merge_landscape(self):
        data = comm.gather(self.__energy_matrix, root = 0)
        if rank == 0:
            for matrix in data:
                self.__energy_matrix += matrix
    

    #Getters
    def get_box(self):
        atom_box = tuple(self.__box)
        atom_box[0][1] = Data.old_bounds[0]
        atom_box[1][1] = Data.old_bounds[1]
        return atom_box

    def __len__(self):
        return np.count_nonzero(self.__energy_matrix)

    def __trunc(self, values, dec = 0):
        return np.trunc(np.array(values)*(10**dec))/(10**dec)

    #Main behavior

    def build_test(self, interactions):
        if interactions == "default":
            Data.type_groups = 1
        else:
            Data.type_groups = max(sum(interactions, ()))


        pair_energy = -100
        self.points = (np.array((5, 3, 4)), np.array((10, 12, 17)))
        line_cells = self.__get_line_cells(*self.points)
        new_bond = Bond(cells = line_cells, energy = pair_energy)
        for cell in line_cells:
            cell.add_bond(new_bond)

        #self.__merge_landscape()

    def test_get_line(self):
        return zip(*self.points)

    def get_energy_landscape(self):
        flat_landscape = self.__energy_matrix.flatten()
        return np.vstack((self.__box[0][:, np.newaxis] + (np.indices(self.__energy_matrix.shape).reshape(3, -1) + 0.5) * self.__grid_size, flat_landscape))[:,flat_landscape != 0]

    def get_grid(self, span = None):
        shape = self.__energy_matrix.shape
        if span is None:
            span = np.array([[0, 0, 0], np.array(shape) - 1])
        else:
            span = np.array(span)
        grid_coords = np.indices(shape).reshape(3, -1).T

        x_start = grid_coords[(grid_coords[:, 0] == span[0][0])&np.prod(grid_coords[:, 1:] >= span[0, 1:], axis = 1).astype(bool)&np.prod(grid_coords[:, 1:] <= span[1, 1:], axis = 1).astype(bool)]
        y_start = grid_coords[(grid_coords[:, 1] == span[0][1])&np.prod(grid_coords[:, ::2] >= span[0, ::2], axis = 1).astype(bool)&np.prod(grid_coords[:, ::2] <= span[1, ::2], axis = 1).astype(bool)]
        z_start = grid_coords[(grid_coords[:, 2] == span[0][2])&np.prod(grid_coords[:, :-1] >= span[0, :-1], axis = 1).astype(bool)&np.prod(grid_coords[:, :-1] <= span[1, :-1], axis = 1).astype(bool)]

        # Create lines_start array
        lines_start = np.vstack((x_start, y_start, z_start))
        
        # Create lines_end array
        x_end = np.copy(x_start)
        x_end[:, 0] = span[1][0]
        y_end = np.copy(y_start)
        y_end[:, 1] = span[1][1]
        z_end = np.copy(z_start)
        z_end[:, 2] = span[1][2]
        lines_end = np.vstack((x_end, y_end, z_end))

        grid_lines = np.stack((lines_start, lines_end), axis=-1)

        
        lines = grid_lines*self.__grid_size + self.__box[0][:, np.newaxis]

        return lines

    def __build_potmap(self):
        file = open(self.__potfile, "r")
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

    def __get_cell_center(self, cell_coords):
        return self.__grid_size*(cell_coords + 0.5) + self.__box[0]

    def __get_line_cells(self, pos1, pos2):
        line_cells = set()
        cell1 = self.__cmap(pos1)
        cell2 = self.__cmap(pos2)
        self.__rec_get_line_cells(cell1, cell2, pos1, pos2, line_cells)
        return line_cells

    def __rec_get_line_cells(self, cell1, cell2, pos1, pos2, line_cells):
        line_cells.add(self.__energy_matrix[tuple(cell1)])
        #print(self.__get_cell_center(cell1))
        if np.array_equal(cell1, cell2):
            return

        cell_lower_bounds = cell1*self.__grid_size + self.__box[0]
        cell_center = self.__get_cell_center(cell1)
        dx, dy, dz = pos2 - pos1 #Line vector
        #print("Line vector:", dx, dy, dz)
        #Check whether the line in question intersects with each plane of the pos1 cell
        for vec in np.array([(1, 0, 0), (0, 1, 0), (0, 0, 1), (-1, 0, 0), (0, -1, 0), (0, 0, -1)]):
            point = cell_center + self.__grid_size*vec/2
            A, B, C, D = *vec, np.sum(-point*vec) #Plane definition Ax + By + Cz + D = 0
            numerator = -A*pos1[0] - B*pos1[1] - C*pos1[2] - D
            denominator = A*dx + B*dy + C*dz
            #print("Current cell:", cell1, "Probe vector:", vec, "Numerator:", numerator, "Denominator:", denominator)
            if denominator == 0:
                if numerator == 0:
                    line_cells.add(cell1 + vec) #Line is in the plane in question. Add neighboring cell and proceed forward based on the intersecitons with other planes
                else:
                    pass #No intersection with the plane in question
            else:
                t = numerator/denominator
                #print("Line parameter:", t)
                if t >= 0: #Check that the scan is not backwards
                    intersection = pos1 + np.array([dx, dy, dz])*t
                    #Create comapative upper in lower bounds with the direction of the intersecting plane masked out
                    comp_low = self.__trunc(np.copy(cell_lower_bounds), 3)
                    comp_high = self.__trunc(np.copy(cell_lower_bounds + self.__grid_size), 3)
                    comp_low[np.argmax(np.abs(vec))] = -np.inf
                    comp_high[np.argmax(np.abs(vec))] = np.inf
                    #print("Comparisons:", comp_low, comp_high)
                    #print("Intersection:", intersection)
                    if np.prod(comp_low <= self.__trunc(intersection, 3)) and np.prod(comp_high >= self.__trunc(intersection, 3)):
                        next_cell = cell1 + vec
                        if not self.__energy_matrix[tuple(next_cell)] in line_cells:
                            self.__rec_get_line_cells(next_cell, cell2, intersection, pos2, line_cells)
                        else:
                            #print("Cell already visited")
                            pass



    def build(self, interactions = "default"):
        if interactions == "default":
            Data.type_groups = 1
        else:
            Data.type_groups = max(sum(interactions, ()))

        Helper.mpi_print("Number of type groups:", Data.type_groups)


        forcefield = self.__build_potmap()

        #Gather per-atom information
        positions = np.array(self.__lmp.gather_atoms("x", 1, 3), dtype = ct.c_double).reshape((-1, 3))
        types = np.array(self.__lmp.gather_atoms("type", 0, 1), dtype = ct.c_int)
        neigh_lists = self.__lmp.numpy.get_neighlist(self.__lmp.find_pair_neighlist("table"))
        tags = self.__lmp.extract_atom("id")
        Helper.print(f"Starting neighbor lookup on process {rank}")
        ctr = 0
        for local_id, nlist in neigh_lists:
            tag1 = tags[local_id]
            type1 = types[tag1 - 1]
            for id2 in nlist:
                tag2 = tags[id2]
                type2 = types[tag2 - 1]
                delta = positions[tag1 - 1] - positions[tag2 - 1]
                delta -= np.around(delta/self.__sides)*self.__sides
                dist = np.linalg.norm(delta)
                type_key = tuple(sorted((type1, type2)))
                if forcefield[type_key].x.min() <= dist and dist <= forcefield[type_key].x.max():
                    ctr += 1
                    pair_energy = forcefield[type_key](dist)
                    cell_list = self.__get_line_cells(positions[tag1 -1], positions[tag2 - 1])
                    new_bond = Bond(energy = pair_energy, cells = cell_list)
                    for cell in cell_list:
                        cell.add_bond(new_bond)
                    if ctr == 100:
                        break
            if ctr == 100:
                break
        Helper.print(f"Finished neighbor lookup on process {rank} with {ctr} pairs")

        self.__merge_landscape()


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


class Bond:

    def __init__(self, energy, cells):
        self.__energy = energy
        self.__cells = cells

    @property
    def energy(self):
        return self.__energy

    @property
    def cells(self):
        return self.__cells

    def fracture(self):
        for cell in self.__cells:
            cell.energy -= self.__energy


class Cell:

    def __init__(self, energy = 0, position = (0, 0, 0)):
        self.energy = energy
        self.__bonds = set()
        self.__pos = position

    def add_bond(self, bond):
        self.__bonds.add(bond)
        self.energy += bond.energy

    def fracture(self):
        for bond in self.__bonds:
            bond.fracture()

    @property
    def position(self):
        return self.__pos

    def __str__(self):
        return str(self.energy)

    def __add__(self, other):
        if isinstance(other, Cell):
            self.__bonds.update(other.__bonds)
            self.energy += other.energy
            return self
        else:
            return NotImplemented

    def __bool__(self):
        return self.energy != 0

    def __lt__(self, other):
        if isinstance(other, Cell):
            return self.energy < other.energy
        else:
            return NotImplemented

    def __le__(self, other):
        if isinstance(other, Cell):
            return self.energy <= other.energy
        else:
            return NotImplemented

    def __gt__(self, other):
        if isinstance(other, Cell):
            return self.energy > other.energy
        else:
            return NotImplemented

    def __ge__(self, other):
        if isinstance(other, Cell):
            return self.energy >= other.energy
        else:
            return NotImplemented

    def __eq__(self, other):
        if isinstance(other, Cell):
            return self.energy == other.energy
        else:
            return NotImplemented

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        return hash(self.__pos)

    def __repr__(self):
        return f"|{self.__pos} {self.energy}|"
