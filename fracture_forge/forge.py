import  os, sys, argparse
from classes.Storage import Data, SystemParams, Helper
from classes.my_structs import FracGraph
import numpy as np

import matplotlib.pyplot as plt


def draw_arcs(nodes, plt, alpha, R):
    for node in nodes:
        if not node.is_leaf():
            circ_angles = np.linspace(alpha*np.pi/180, np.pi*(1 - alpha/180), 100)
            point = node.get_pos()
            circ_x = point[0] + R*np.cos(circ_angles)
            circ_y = point[1] + R*np.sin(circ_angles)
            plt.plot(circ_x, circ_y, color = "blue")

def draw_lines(nodes, plt):
    for node in nodes:
        parent_pos = node.get_pos()
        for child in node.get_children():
            node_pos = child.get_pos()
            if node_pos[0] - parent_pos[0]:
                tan = (node_pos[1] - parent_pos[1])/(node_pos[0] - parent_pos[0])
                line_x = np.linspace(parent_pos[0], node_pos[0], 100)
                line_y = parent_pos[1] + tan*(line_x - parent_pos[0])
            else:
                line_x = [node_pos[0], node_pos[0]]
                line_y = [parent_pos[1], node_pos[1]]

            plt.plot(line_x, line_y, color = "red")

def visualize(graph, dr, dtheta):
    nodes = graph.flatten()
    points = np.array([node.get_pos() for node in nodes])

    x = points[:, 0]
    y = points[:, 1]

    #draw_arcs(nodes, plt, dtheta, dr)
    #draw_lines(nodes, plt)

    box = graph.get_box()
    plt.plot([box[0][0], box[1][0]], [box[0][1], box[0][1]], color = "black")
    plt.plot([box[0][0], box[1][0]], [box[1][1], box[1][1]], color = "black")
    plt.plot([box[0][0], box[0][0]], [box[0][1], box[1][1]], color = "black")
    plt.plot([box[1][0], box[1][0]], [box[0][1], box[1][1]], color = "black")

    plt.plot(x, y, marker = 'o', linestyle = 'None', color = "green")
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

    graph = FracGraph(error = args.error, start_buffer = args.radius/2, test_mode = False, simulation_temp = args.temperature, connection_radius = args.radius)
    #tree.build(dtheta = args.angle, dr = args.radius, interactions = args.interactions)
    graph.build_mid_atom(pivot_atom_type = args.pivot_type, num_neighs = args.neighbors, interactions = args.interactions)
    print("Number of nodes created:", len(graph))


    #data_dir = "out_files" 
    #res = 0.69*tree.calculate(data_dir)
    #print("G:", res)

    visualize(graph, args.radius, args.angle)





if __name__ == "__main__":
    main()
