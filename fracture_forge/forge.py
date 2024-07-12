import  os, sys, argparse, ast
from classes.Storage import Data, SystemParams, Helper
from classes.my_structs import FracGraph
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

@Helper.linear_func
def get_RGB(eng, min_eng, max_eng):
    norm_eng = (eng - min_eng)/(max_eng - min_eng)
    norm_eng = np.where(norm_eng < 0, 0, norm_eng)
    norm_eng = np.where(norm_eng > 1, 1, norm_eng)
    R = np.where(norm_eng <= 0.5, 2 * norm_eng, 1)
    G = np.where(norm_eng <= 0.5, 1, 2 * (1 - norm_eng))
    B = np.zeros_like(norm_eng)

    return np.stack((R, G, B), axis=-1)


@Helper.linear_func
def visualize(graph, show_box = False):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # Generate sample data
    x, y, z, energies = graph.get_energy_landscape()
    min_eng, max_eng = np.percentile(energies, 5), np.percentile(energies, 95)
    colors = get_RGB(energies, min_eng, max_eng)
    #sizes = 100*(energies - min_eng)/(max_eng - min_eng)
    ax.scatter(x, y, z, c = colors, marker = 'o', s = 100, depthshade = False, alpha = 0.8)
    if show_box:
        box = graph.get_box()
        ax.plot([box[0][0], box[1][0]], [box[0][1], box[0][1]], [box[0][2], box[0][2]], 'b-')
        ax.plot([box[0][0], box[1][0]], [box[0][1], box[0][1]], [box[1][2], box[1][2]], 'b-')
        ax.plot([box[0][0], box[1][0]], [box[1][1], box[1][1]], [box[0][2], box[0][2]], 'b-')
        ax.plot([box[0][0], box[1][0]], [box[1][1], box[1][1]], [box[1][2], box[1][2]], 'b-')

        ax.plot([box[0][0], box[0][0]], [box[0][1], box[1][1]], [box[0][2], box[0][2]], 'b-')
        ax.plot([box[1][0], box[1][0]], [box[0][1], box[1][1]], [box[0][2], box[0][2]], 'b-')
        ax.plot([box[0][0], box[0][0]], [box[0][1], box[1][1]], [box[1][2], box[1][2]], 'b-')
        ax.plot([box[1][0], box[1][0]], [box[0][1], box[1][1]], [box[1][2], box[1][2]], 'b-')

        ax.plot([box[0][0], box[0][0]], [box[0][1], box[0][1]], [box[1][2], box[0][2]], 'b-')
        ax.plot([box[0][0], box[0][0]], [box[1][1], box[1][1]], [box[1][2], box[0][2]], 'b-')
        ax.plot([box[1][0], box[1][0]], [box[0][1], box[0][1]], [box[1][2], box[0][2]], 'b-')
        ax.plot([box[1][0], box[1][0]], [box[1][1], box[1][1]], [box[1][2], box[0][2]], 'b-')

    """
    for x, y, z in graph.get_grid(span = [[1, 1, 0], [6, 9, 12]]):
        ax.plot(x, y, z, 'b-')

    x, y, z = graph.test_get_line()
    ax.plot(x, y, z, 'b-')
    """
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()


def parser_call():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--error", type = float, default = SystemParams.error, help = "Radius within which the nodes of a fracture tree are considered to be equivalent", metavar = "")
    parser.add_argument("-i", "--interactions", action = "store_true", help = "Prompts the user to specify interactions between type groups")
    parser.add_argument("-s", "--structure", default = None, help = "System structure file in lammps format", metavar = "")
    parser.add_argument("-f", "--force_field", default = None, help = "Forcfield defining atom interactions", metavar = "")
    parser.add_argument("-l", "--landscape", default = None, help = "Saved fracture energy landscape file", metavar = "")
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

    graph = FracGraph(error = args.error, test_mode = False)
    if not os.path.isfile("path_save.csv"):
        if args.landscape is None:
            #graph.build(pivot_atom_type = args.pivot_type, num_neighs = args.neighbors, interactions = args.interactions)
            graph.build_test(interactions = args.interactions)
            graph.save()
        else:
            Helper.mpi_print("Loading landscape from file")
            graph.load_landscape(os.path.abspath(args.landscape))
        Helper.mpi_print("Number of nodes created:", len(graph))


        data_dir = "out_files" 
        #res = 0.69*graph.calculate(data_dir)
        #print("G:", res)
    else:
        Helper.print("------------------------\nPath save file has been located. No calculation will be performed. To initiate new fracture path search delete the path_save.csv file\n------------------------")

    visualize(graph)





if __name__ == "__main__":
    main()
