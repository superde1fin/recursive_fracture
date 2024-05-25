import  os, sys
from classes.Storage import Data, SystemParams, Helper
from classes.my_structs import FracTree
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

def visualize(tree, dr, dtheta):
    nodes = tree.flatten()

    points = [node.get_pos() for node in nodes]

    x = np.array(points)[:, 0]
    y = np.array(points)[:, 1]

    #draw_arcs(nodes, plt, dtheta, dr)
    draw_lines(nodes, plt)

    box = tree.get_box()
    plt.plot([box[0][0], box[1][0]], [box[0][1], box[0][1]], color = "black")
    plt.plot([box[0][0], box[1][0]], [box[1][1], box[1][1]], color = "black")
    plt.plot([box[0][0], box[0][0]], [box[0][1], box[1][1]], color = "black")
    plt.plot([box[1][0], box[1][0]], [box[0][1], box[1][1]], color = "black")

    plt.plot(x, y, marker = 'o', linestyle = 'None', color = "green")
    plt.show()


def main():

    dr = 10
    error = 0.1
    dtheta = 30
    interactions = [(1, 2), (1, 3), (1, 4), (1, 5), (2, 4), (3, 5), (4, 5)]
    Data.non_inter_cutoff = 10
    simulation_temp = 300
    tree = FracTree(error = error, start_buffer = dr/2, test_mode = False, simulation_temp = 300)
    tree.build(dtheta = dtheta, dr = dr, interactions = interactions)
    print("Number of nodes created:", len(tree))


    data_dir = "out_files" 
    res = 0.69*tree.calculate(data_dir)
    print("G:", res)

    visualize(tree, dr, dtheta)





if __name__ == "__main__":
    main()
