import  os, sys, argparse, ast
from classes.Storage import Data, SystemParams, Helper
from classes.my_structs import FracGraph
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


def draw_arcs(nodes, alpha, R):
    for node in nodes:
        if not node.is_leaf():
            circ_angles = np.linspace(alpha*np.pi/180, np.pi*(1 - alpha/180), 100)
            point = node.get_pos()
            circ_x = point[0] + R*np.cos(circ_angles)
            circ_y = point[1] + R*np.sin(circ_angles)
            plt.plot(circ_x, circ_y, color = "blue")

def draw_lines(nodes):
    for node in nodes:
        parent_pos = node.get_pos()
        for child in node.get_neighbors():
            node_pos = child.get_pos()
            if node_pos[0] - parent_pos[0]:
                tan = (node_pos[1] - parent_pos[1])/(node_pos[0] - parent_pos[0])
                line_x = np.linspace(parent_pos[0], node_pos[0], 100)
                line_y = parent_pos[1] + tan*(line_x - parent_pos[0])
            else:
                line_x = [node_pos[0], node_pos[0]]
                line_y = [parent_pos[1], node_pos[1]]

            plt.plot(line_x, line_y, color = "red")


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

def create_gradient(color1, color2, num_segments):
    return [(color1[0] * (1 - t) + color2[0] * t,
             color1[1] * (1 - t) + color2[1] * t,
             color1[2] * (1 - t) + color2[2] * t)
            for t in np.linspace(0, 1, num_segments)]

def generate_segments(x, y):
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    return np.concatenate([points[:-1], points[1:]], axis = 1)



def color_paths(graph):
    if not os.path.isfile("path_save.csv"):
        paths = graph.get_paths()
        writing = True
        text = ""
        file = open("path_save.csv", "w")
    else:
        writing = False
        file = open("path_save.csv", "r")
        custom_globals = {"np" : np}
        file_lines = file.readlines()
        paths = list()
        for line in file_lines:
            path = list()
            for pos in line.split('|'):
                pos_tuple = eval(pos.strip(), {"__builtins__" : None}, custom_globals)
                path.append(pos_tuple)
            paths.append(path)

                
        #paths = [[eval(pos.strip(), {"__builtins__" : None}, custom_globals) for pos in line.split('|')] for line in file_lines[:-1]]
        file.close()

    unique_nodes = pd.unique(pd.DataFrame(paths).values.ravel())
    print(*unique_nodes, sep = "\n")
    num_unique = len(unique_nodes)
    energies = np.array([tup[2] for tup in unique_nodes if tup])
    min_eng, max_eng = np.percentile(energies, 5), np.percentile(energies, 95)
    
    box = graph.get_box()
    segments_per_cut = 100
    num_paths = len(paths)
    visited = list()
    ax = plt.gca()
    main_path_offset = 1
    for npi, path in enumerate(paths):
        print(f"Colored {round(100*npi/(num_paths - 1), 2)}% of paths")
        parent_pos = path[0]
        i = 1
        num_nodes = len(path)
        if writing:
            text += "|".join(map(str, path)) + "\n"
        if parent_pos[1] > box[1][1]:
            plt.plot([node[0] for node in path], [node[1] for node in path], color = (0.5, 0.5, 0.5, 0.2), linewidth = 3)

        while i < num_nodes:
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
            parent_RGB = get_RGB(parent_pos[2], min_eng, max_eng)
            colors = create_gradient(parent_RGB, node_RGB, int(segments_per_cut/2))
            colors += [node_RGB]*(segments_per_cut - int(segments_per_cut/2))
            #colors = create_gradient(parent_RGB, node_RGB, segments_per_cut)
            lc = LineCollection(generate_segments(line_x, line_y), colors = colors, linewidth = 0.5, capstyle = "round", alpha = 0.5)
            ax.add_collection(lc)
            #plt.plot(line_x, line_y, color = (node_RGB))

            visited.append(node_pos)
            
            i += 1
        final_path = False
    if writing:
        file.write(text)
        file.close()



def visualize(graph, dr, dtheta):
    nodes = graph.flatten()
    points = np.array([node.get_pos() for node in nodes])
    #points = graph.get_node_coords()

    x = points[:, 0]
    y = points[:, 1]

    #draw_arcs(nodes, dtheta, dr)
    #draw_lines(nodes)
    color_paths(graph)

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

def parser_call():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--temperature", type = int, default = SystemParams.simulation_temp, help = "Temperature used in the initial velocity command", metavar = '')
    parser.add_argument("-r", "--radius", type = int, default = SystemParams.dr, help = "Probe radius", metavar = "")
    parser.add_argument("-e", "--error", type = int, default = SystemParams.error, help = "Radius within which the nodes of a fracture tree are considered to be equivalent", metavar = "")
    parser.add_argument("-a", "--angle", type = int, default = SystemParams.dtheta, help = "Angle between the branches of the fracture tree", metavar = "")
    parser.add_argument("-i", "--interactions", action = "store_true", help = "Prompts the user to specify interactions between type groups")
    parser.add_argument("-s", "--structure", default = None, help = "System structure file in lammps format", metavar = "")
    parser.add_argument("-f", "--force_field", default = None, help = "Forcfield defining atom interactions", metavar = "")
    parser.add_argument("-p", "--pivot_type", default = SystemParams.pivot_type, help = "Numerical type corresponding to a atoms around which the fracture nodes will be created", metavar = "")
    parser.add_argument("-n", "--neighbors", default = SystemParams.neigh_num, help = "Number of nearest neighbors to the pivot atom, used to determine the midpoint of bonds between the pivot atom and its neighbors for fracture node creation", metavar = "")
    parser.add_argument("-w", "--width", default = Data.non_inter_cutoff, help = "Surface width", metavar = "", type = int)
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

    graph = FracGraph(error = args.error, start_buffer = args.radius/2, test_mode = True, simulation_temp = args.temperature, connection_radius = args.radius)
    """
    if not os.path.isfile("path_save.csv"):
        graph.build(pivot_atom_type = args.pivot_type, num_neighs = args.neighbors, interactions = args.interactions)
        #graph.build_test(interactions = args.interactions)
        print("Number of nodes created:", len(graph))


        data_dir = "out_files" 
        res = 0.69*graph.calculate(data_dir)
        print("G:", res)
    else:
        Helper.print("------------------------\nPath save file has been located. No calculation will be performed. To initiate new fracture path search delete the path_save.csv file\n------------------------")

    visualize(graph, args.radius, args.angle)
    """





if __name__ == "__main__":
    main()
