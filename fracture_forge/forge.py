import  os, sys, argparse, ast
from classes.Storage import Data, SystemParams, Helper
from classes.my_structs import FracGraph
import numpy as np
import pandas as pd
from mpi4py import MPI

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

@Helper.linear_func
def get_best_path(paths, box):
    full_path = list()
    tail_y = -float("inf")
    for path in paths:
        if path[0][1] > tail_y and path[0][1] > box[1][1]:
            full_path = path[::-1]
            tail_y = path[0][1]

    if full_path:
        eng = 0
        line_length = 0
        prev_node = full_path[0]
        for node in full_path[1:]:
            eng += node[2]
            line_length += np.sqrt((prev_node[0] - node[0])**2 + (prev_node[1] - node[1])**2)
            prev_node = node

        return eng/(2*(box[1][2] - box[0][2])*line_length)
    else:
        raise RuntimeError("Error: No full path was found, something went wrong")


@Helper.linear_func
def get_RGB(eng, min_eng, max_eng):
    norm_eng = (eng - min_eng)/(max_eng - min_eng)
    if norm_eng < 0:
        norm_eng = 0
    if norm_eng > 1:
        norm_eng = 1
    if norm_eng <= 0.5:
        RGB = (2*norm_eng, 1, 0)
    else:
        RGB = (1, 2 * (1 - norm_eng), 0)

    return RGB

@Helper.linear_func
def create_gradient(color1, color2, num_segments):
    return [(color1[0] * (1 - t) + color2[0] * t,
             color1[1] * (1 - t) + color2[1] * t,
             color1[2] * (1 - t) + color2[2] * t)
            for t in np.linspace(0, 1, num_segments)]

@Helper.linear_func
def generate_segments(x, y):
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    return np.concatenate([points[:-1], points[1:]], axis = 1)

@Helper.linear_func
def load_paths(filename):
    file = open(filename, "r")
    custom_globals = {"np" : np}
    file_lines = file.readlines()
    paths = list()
    for line in file_lines:
        path = list()
        for pos in line.split('|'):
            pos_tuple = eval(pos.strip(), {"__builtins__" : None}, custom_globals)
            path.append(pos_tuple)
        paths.append(path)

            
    file.close()
    return paths



@Helper.linear_func
def color_paths(graph, paths = None):
    if not paths is None:
        writing = False
    else:
        paths = graph.get_paths()
        writing = True
        text = ""
        file = open("path_save.csv", "w")

    unique_nodes = pd.unique(pd.DataFrame(paths).values.ravel())
    num_unique = len(unique_nodes)
    energies = np.array([tup[2] for tup in unique_nodes if tup])
    non_zero = [eng for eng in energies if eng != 0]
    if not non_zero:
        raise RuntimeError("All paths have 0 G. Something went wrong")
    min_eng, max_eng = np.percentile(non_zero, 5), np.percentile(non_zero, 95)
    
    box = graph.get_box()
    segments_per_cut = 100
    num_paths = len(paths)
    visited = list()
    ax = plt.gca()
    main_path_offset = 1
    found_at_least_one_full = False
    for npi, path in enumerate(paths):
        full_path = None
        #print(f"Colored {round(100*npi/(num_paths - 1), 2)}% of paths")
        parent_pos = path[0]
        i = 1
        num_nodes = len(path)
        if writing:
            text += "|".join(map(str, path)) + "\n"
        if parent_pos[1] > box[1][1]:
            full_path = path
            found_at_least_one_full = True


        while i < num_nodes:
            line_thickness = 0.5
            line_transparency = 0.5
            node_pos = parent_pos
            parent_pos = path[i]
            if node_pos in visited:
                i += 1
                continue
            if node_pos[0] - parent_pos[0]:
                tan = (node_pos[1] - parent_pos[1])/(node_pos[0] - parent_pos[0])
                line_x = np.linspace(parent_pos[0], node_pos[0], segments_per_cut)
                line_y = parent_pos[1] + tan*(line_x - parent_pos[0])
            else:
                line_x = [node_pos[0]]*segments_per_cut
                line_y = np.linspace(parent_pos[1], node_pos[1], segments_per_cut)


            node_RGB = get_RGB(node_pos[2], min_eng, max_eng)
            if node_pos[2] == 0:
                node_RGB = (0, 0, 0)
                line_thickness = 1
                line_transparency = 1
            parent_RGB = get_RGB(parent_pos[2], min_eng, max_eng)
            colors = create_gradient(parent_RGB, node_RGB, int(segments_per_cut/2))
            colors += [node_RGB]*(segments_per_cut - int(segments_per_cut/2))
            lc = LineCollection(generate_segments(line_x, line_y), colors = colors, linewidth = line_thickness, capstyle = "round", alpha = line_transparency)
            ax.add_collection(lc)

            visited.append(node_pos)
            
            i += 1
        final_path = False

        if full_path:
            line_length = 0
            plt.plot([node[0] for node in full_path], [node[1] for node in full_path], color = (0.5, 0.5, 0.5, 0.2), linewidth = 3)
            #prev_node = (path[0][0], box[1][1])
            prev_node = full_path[1]
            for node in full_path[1:-1]:
                line_length += np.sqrt((prev_node[0] - node[0])**2 + (prev_node[1] - node[1])**2)
                prev_node = node

            node = (path[-1][0], box[0][1])
            line_length += np.sqrt((prev_node[0] - node[0])**2 + (prev_node[1] - node[1])**2)
            #print("Line length:", line_length)
    if not found_at_least_one_full:
        raise RuntimeError("Error: No full path was found, something went wrong")

    if writing:
        file.write(text)
        file.close()



@Helper.linear_func
def visualize(graph, paths = None):
    """
    nodes = graph.flatten()
    points = np.array([node.get_pos() for node in nodes])

    x = points[:, 0]
    y = points[:, 1]
    """

    color_paths(graph, paths)

    box = graph.get_box()
    plt.plot([box[0][0], box[1][0]], [box[0][1], box[0][1]], color = "black", alpha = 0.1)
    plt.plot([box[0][0], box[1][0]], [box[1][1], box[1][1]], color = "black", alpha = 0.1)
    plt.plot([box[0][0], box[0][0]], [box[0][1], box[1][1]], color = "black", alpha = 0.1)
    plt.plot([box[1][0], box[1][0]], [box[0][1], box[1][1]], color = "black", alpha = 0.1)

    #plt.plot(x, y, marker = 'o', linestyle = 'None', color = "black")
    ax = plt.gca()
    ax.set_aspect("equal", adjustable = "box")
    ax.axis("off")
    plt.savefig("energy_landscape.png", dpi = 300)
    plt.show()

def vis_nodes(graph):
    for node in graph.flatten():
        x, y = node.get_pos()
        plt.scatter(x, y, color = "black")
        plt.text(x + 0.1, y + 0.1, str(node.get_id()))

    box = graph.get_box()
    plt.plot([box[0][0], box[1][0]], [box[0][1], box[0][1]], color = "black", alpha = 0.1)
    plt.plot([box[0][0], box[1][0]], [box[1][1], box[1][1]], color = "black", alpha = 0.1)
    plt.plot([box[0][0], box[0][0]], [box[0][1], box[1][1]], color = "black", alpha = 0.1)
    plt.plot([box[1][0], box[1][0]], [box[0][1], box[1][1]], color = "black", alpha = 0.1)
    plt.savefig("node_positions.png")


def parser_call():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--temperature", type = int, default = SystemParams.simulation_temp, help = "Temperature used in the initial velocity command", metavar = '')
    parser.add_argument("-u", "--units", type = str, default = SystemParams.default_units, help = "Units that the potential supports.", metavar = '')
    parser.add_argument("-l", "--load_margin", type = float, default = SystemParams.load_margin, help = "Simulates load factor incremental step (error of experiment).", metavar = '')
    parser.add_argument("-r", "--radius", type = float, default = SystemParams.dr, help = "Probe radius", metavar = "")
    parser.add_argument("-e", "--error", type = int, default = SystemParams.error, help = "Radius within which the nodes of a fracture tree are considered to be equivalent", metavar = "")
    parser.add_argument("-i", "--interactions", action = "store_true", help = "Prompts the user to specify interactions between type groups")
    parser.add_argument("-a", "--arbitrary_grid", action = "store_true", help = "When specified generates grid without regard to atom postions.")
    parser.add_argument("-s", "--structure", default = None, help = "System structure file in lammps format", metavar = "")
    parser.add_argument("-f", "--force_field", default = None, help = "Forcfield defining atom interactions", metavar = "")
    parser.add_argument("-p", "--pivot_type", default = SystemParams.pivot_type, help = "Numerical type corresponding to a atoms around which the fracture nodes will be created", metavar = "", type = int)
    parser.add_argument("-n", "--neighbors", default = SystemParams.neigh_num, help = "Number of nearest neighbors to the pivot atom, used to determine the midpoint of bonds between the pivot atom and its neighbors for fracture node creation", metavar = "")
    parser.add_argument("-w", "--width", default = Data.non_inter_cutoff, help = "Surface width (non-interacting cutoff).", metavar = "", type = float)
    parser.add_argument("-v", "--vary", default = 0, help = "This value when specified changes the default non-interacting cutoff width. Only works with a present path_save.csv file generated after previous calculation.", metavar = "", type = float)
    parser.add_argument("-pr", "--pressure", default = False, help = "Calculate surface energy using pressure difference.", action = "store_true")
    parser.add_argument("-rd", "--random", default = 0, help = "Specifies the number of random paths throug a material to run. If this option is specified no path search will be performed.", metavar = "", type = int)
    parser.add_argument("-nt", "--no_interaction_table", default = SystemParams.nono_table, help = "Path to the no interactions table. Has to include one entry labeled as NoNo with zeros throughout.", metavar = "", type = str)
    parser.add_argument("-m", "--minimize", action = "store_true", help = "When specified lammps minimization is performed after each cut.")
    args = parser.parse_args()

    
    if args.interactions:
        print("\nPlease specify the groups that are supposed to interact according to the employed force field.\nGroup 1 : non-surface, group 2 : top surface, group 3 : bottom_surface, group 4 : top tip, group 5 : bottom tip.\nRegions are only created if the interactions are specified.\nAfter all interactions have been provided press 0.\nFormat : 1 4\n")
        interaction_list = list()
        done = False
        while not done:
            inter_str = input().strip()
            if inter_str == "0":
                done = True
            else:
                try:
                    pair = tuple(map(int, inter_str.split()))
                    assert len(pair) == 2
                    interaction_list.append(pair)
                except:
                    print("Incorrect format. Please input two integers separated by a space")
        args.interactions = interaction_list
    else:
        args.interactions = SystemParams.interactions

    return args




def main():
    args = parser_call()
    Data.structure_file = args.structure
    Data.potfile = args.force_field
    Data.non_inter_cutoff = args.width
    

    if args.pressure:
        Data.use_pressure = True

    graph = FracGraph(error = args.error, start_buffer = args.radius/2, test_mode = False, simulation_temp = args.temperature, connection_radius = args.radius, nono_table = args.no_interaction_table, load_margin = args.load_margin, units = args.units, minimize = args.minimize)
    if not os.path.isfile("path_save.csv"):
        if args.arbitrary_grid:
            graph.build_arbitrary(interactions = args.interactions)
        else:
            graph.build(pivot_atom_type = args.pivot_type, num_neighs = args.neighbors, interactions = args.interactions)
            #graph.build_test(interactions = args.interactions)
        Helper.mpi_print("Number of nodes created:", len(graph))


        #vis_nodes(graph)

        if args.random:
            random_paths = graph.get_random_paths(args.random)
            if rank == 0:
                f = open("random_paths.csv", "w")
                f.write('\n'.join(map(str, random_paths)))
                f.close()
                plt.hist(random_paths, bins = 3)
                plt.savefig("random_paths.png")
        else:
            data_dir = "out_files" 
            res = graph.calculate(data_dir)
        paths = None
    else:
        Helper.mpi_print("------------------------\nPath save file has been located. No calculation will be performed. To initiate new fracture path search delete the path_save.csv file\n------------------------")
        if rank == 0:
            paths = load_paths("path_save.csv")
            if args.vary:
                full_path = None
                found = False
                i = 0
                num_paths = len(paths)
                box = graph.get_box()
                while i < num_paths and not found:
                    if paths[i][0][1] > box[1][1]:
                        full_path = paths[i][-2:0:-1]
                        found = True
                    else:
                        i += 1
                Data.non_inter_cutoff = args.vary
                res = graph.recalculate_path(full_path, interactions = args.interactions)
                print(f"Recalculated G: {0.69478578545*res} for non-interacting width of {args.vary} Angstroms")

        if args.random:
            graph.build(pivot_atom_type = args.pivot_type, num_neighs = args.neighbors, interactions = args.interactions)
            random_paths = graph.get_random_paths(args.random)
            if rank == 0:
                random_paths.append(get_best_path(paths, box = graph.get_box()))
                f = open("random_paths.csv", "w")
                f.write('\n'.join(map(str, random_paths)))
                f.close()
                plt.hist(random_paths, bins = 3)
                plt.savefig("random_paths.png")

            
    if not args.random and rank == 0:
        visualize(graph, paths)





if __name__ == "__main__":

    main()
