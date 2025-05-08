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


def verbosity_type():
    def checker(value):
        ivalue = int(value)
        if ivalue < 1 or ivalue > SystemParams.max_verbosity:
            raise argparse.ArgumentTypeError(f"Verbosity must be between 1 and {SystemParams.max_verbosity}")
        return ivalue
    return checker



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
def load_path(filename):
    file = open(filename, "r")
    custom_globals = {"np" : np}
    file_lines = file.readlines()
    path = list()
    for line in file_lines:
        pos_tuple = eval(line.strip(), {"__builtins__" : None}, custom_globals)
        path.append(pos_tuple)

            
    file.close()
    return path



@Helper.linear_func
def color_path(graph, path = None):
    if not path is None:
        writing = False
    else:
        path = graph.get_path()
        writing = True
        text = ""
        file = open("path_save.csv", "w")


    energies = np.array([tup[2] for tup in path if tup])
    non_zero = [eng for eng in energies if eng != 0]
    if not non_zero:
        raise RuntimeError("All path nodes have 0 energy. Something went wrong")
    min_eng, max_eng = np.percentile(non_zero, 5), np.percentile(non_zero, 95)
    
    box = graph.get_box()
    segments_per_cut = 100
    ax = plt.gca()
    main_path_offset = 1

    parent_pos = path[0]
    i = 1
    num_nodes = len(path)
    if writing:
        text += "\n".join(map(str, path))

    min_x = parent_pos[0]
    max_x = parent_pos[0]


    while i < num_nodes:
        line_thickness = 0.5
        line_transparency = 0.5
        node_pos = parent_pos
        parent_pos = path[i]

        if parent_pos[0] < min_x:
            min_x = parent_pos[0]
        if parent_pos[0] > max_x:
            max_x = parent_pos[0]


        if node_pos[0] - parent_pos[0]:
            tan = (node_pos[1] - parent_pos[1])/(node_pos[0] - parent_pos[0])
            line_x = np.linspace(parent_pos[0], node_pos[0], segments_per_cut)
            line_y = parent_pos[1] + tan*(line_x - parent_pos[0])
        else:
            line_x = [node_pos[0]]*segments_per_cut
            line_y = np.linspace(parent_pos[1], node_pos[1], segments_per_cut)


        node_RGB = get_RGB(node_pos[2], min_eng, max_eng)

        parent_RGB = get_RGB(parent_pos[2], min_eng, max_eng)
        colors = create_gradient(parent_RGB, node_RGB, int(segments_per_cut/2))
        colors += [node_RGB]*(segments_per_cut - int(segments_per_cut/2))
        lc = LineCollection(generate_segments(line_x, line_y), colors = colors, linewidth = line_thickness, capstyle = "round", alpha = line_transparency)
        ax.add_collection(lc)

        
        i += 1

    if writing:
        file.write(text)
        file.close()

    span = max_x - min_x
    return path[-1][0] - span, path[-1][0] + span



@Helper.linear_func
def visualize(graph, path = None):

    min_x, max_x = color_path(graph, path)

    box = graph.get_box()
    plt.plot([min_x, max_x], [box[0][1], box[0][1]], color = "black", alpha = 0.1)
    plt.plot([min_x, max_x], [box[1][1], box[1][1]], color = "black", alpha = 0.1)

    #plt.plot([min_x, min_x], [box[0][1], box[1][1]], color = "black", alpha = 0.1)
    #plt.plot([max_x, max_x], [box[0][1], box[1][1]], color = "black", alpha = 0.1)

    ax = plt.gca()
    ax.set_aspect("equal", adjustable = "box")
    plt.savefig("fracture_path.png", dpi = 300)
    #plt.show()

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
    parser.add_argument("-t", "--temperature", type = float, default = SystemParams.simulation_temp, help = "Temperature used in the initial velocity command", metavar = '')
    parser.add_argument("-u", "--units", type = str, default = SystemParams.default_units, help = "Units that the potential supports.", metavar = '')
    parser.add_argument("-r", "--radius", type = float, default = SystemParams.dr, help = "Probe radius", metavar = "")
    parser.add_argument("-e", "--error", type = int, default = SystemParams.error, help = "Radius within which the nodes of a fracture tree are considered to be equivalent", metavar = "")
    parser.add_argument("-i", "--interactions", action = "store_true", help = "Prompts the user to specify interactions between type groups")
    parser.add_argument("-a", "--arbitrary_grid", type = float, nargs="?", const = 1, default = 0, help = "When specified generates grid without regard to atom postions.", metavar = "")
    parser.add_argument("-s", "--structure", default = None, help = "System structure file in lammps format", metavar = "")
    parser.add_argument("-f", "--force_field", default = None, help = "Forcfield defining atom interactions", metavar = "")
    parser.add_argument("-m", "--minimize", action = "store_true", help = "When specified lammps minimization is performed after each cut.")
    parser.add_argument("-v", "--verbose", type = verbosity_type(), help = "Sets a level of logging information printed to the screen.", metavar = "", default = SystemParams.default_verbosity)
    args = parser.parse_args()

    
    return args




def main():
    args = parser_call()
    Data.structure_file = args.structure
    Data.potfile = args.force_field
    Data.verbosity = args.verbose
    

    graph = FracGraph(error = args.error, start_buffer = args.radius/2, test_mode = Data.verbosity == SystemParams.max_verbosity, simulation_temp = args.temperature, connection_radius = args.radius, units = args.units, minimize = args.minimize)
    if not os.path.isfile("path_save.csv"):
        if args.arbitrary_grid:
            graph.build_arbitrary()
        else:
            #graph.build()
            graph.build_test()
        Helper.mpi_print("Number of nodes created:", len(graph))


        res = graph.calculate()
        path = None
    else:
        Helper.mpi_print("------------------------\nPath save file has been located. No calculation will be performed. To initiate new fracture path search delete the path_save.csv file\n------------------------")
        path = load_path("path_save.csv")

            
    if rank == 0:
        visualize(graph, path)





if __name__ == "__main__":

    main()
