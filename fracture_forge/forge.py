from lammps import PyLammps
import ctypes as ct
import  glob, copy, os, sys
from classes.Storage import Data, SystemParams, Helper, Lmpfunc, Fake_lmp
from classes.my_structs import FracTree
import numpy as np
import time, random
import regex as re

import matplotlib.pyplot as plt


def quazi_static(lmp, dr_frac = 0.1, dtheta = 10):
    Helper.print("start")
    dr = dr_frac*min(lmp.eval("lx"), lmp.eval("ly"), lmp.eval("lz"))
    Helper.print(f"dr value: {dr}")
    create_surface(lmp)
    filename = glob.glob("glass_*.structure")[-1]
    name_handle = re.search(r"(?<=glass_).+(?=\.structure)", filename).group()
    potfile = os.path.abspath(f"pot_{name_handle}.FF")
    potfile = modify_potfile(lmp, potfile, interactions = [(1, 2), (1, 3), (1, 4), (1, 5), (2, 4), (3, 5), (4, 5)], groups = Data.type_groups)
    start_coords = (lmp.eval("lx")/2 + lmp.system.xlo, SystemParams.parameters["old_bounds"][0] - 1)
#    start_coords = (lmp.eval("lx")/2 + lmp.system.xlo, lmp.system.ylo + lmp.eval("ly")/2)
    Data.initial_types = lmp.system.ntypes

    starting_dir = "initial"
    if not os.path.isdir(starting_dir):
        Helper.command(f"mkdir {starting_dir}")
    elif os.listdir(starting_dir):
        Helper.command(f"rm -r {starting_dir}/*")
    Helper.chdir(starting_dir)

    #Check so that surface regions don't overlap through a periodix x boundary
    x_side = lmp.eval("lx")
    if dtheta*(np.pi/180) < np.arcsin(2*Data.non_inter_cutoff/x_side):
        angle_span = (2*dtheta*(np.pi/180), np.pi - 2*dtheta*(np.pi/180), int(180/dtheta) - 3)
    else:
        angle_span = (dtheta*(np.pi/180), np.pi - dtheta*(np.pi/180), int(180/dtheta) - 1)

    Helper.lowest_path_energy = float("inf")
    Helper.fracture_memory = {}
    Helper.mem_ctr = 0

    new_lmp = QSR(lmp = lmp, coords = start_coords, dr = dr, span = angle_span, angles = (np.pi/2, None), potfile = potfile, in_glass = False)

    new_lmp.write_data(f"output.{Helper.output_ctr}.structure")
    
    result = 0.69*(new_lmp.eval("pe") - lmp.eval("pe"))/new_lmp.variables["surface_area"].value

    new_lmp.close()
    Helper.mkdir("../QS_results")
    for filename in os.listdir():
        if os.path.isfile(filename):
            Helper.command(f"mv {filename} ../QS_results")

    return result


def QSR(lmp, coords, dr, span, angles, potfile, in_glass):
    prev_theta, theta = angles
    prev_theta = np.pi/2 if not prev_theta else prev_theta
    Helper.command(f"echo '{coords}' >> path.txt")
    Helper.print(f"Recursion step: {Helper.output_ctr}")

    #Output
    Helper.print("Coordinates: ", coords)
    #Helper.print(f"In {os.getcwd().split(os.path.commonprefix([potfile, os.getcwd()]))[-1]}")

    if not in_glass and coords[1] >= SystemParams.parameters["old_bounds"][0]:
        in_glass = True
        lmp.variable(f"surface_area equal 0")

    if in_glass:
        if coords in Helper.fracture_memory:
            Helper.mem_ctr += 1
            return Helper.fracture_memory[coords]

        lmp = copy_lmp(lmp, potfile, (prev_theta, theta), dr, coords)
        if lmp.eval("pe") >= Helper.lowest_path_energy:
            return lmp

        if coords[0] > lmp.system.xhi:
            dist = dr - (coords[0] - lmp.system.xhi)/np.cos(theta)
        elif coords[0] < lmp.system.xlo:
            dist = dr - (lmp.system.xlo - coords[0])/np.sin(np.pi - theta)
        elif coords[1] > SystemParams.parameters["old_bounds"][1]:
            dist = dr - (coords[1] - SystemParams.parameters["old_bounds"][1])/np.sin(theta)
        elif coords[1] - dr*np.sin(theta) < SystemParams.parameters["old_bounds"][0]:
            dist = (coords[1] - SystemParams.parameters["old_bounds"][0])/np.sin(theta)
        else:
            dist = dr

        lmp.variable(f"surface_area equal {lmp.variables['surface_area'].value + dist*lmp.eval('lz')}")

    if coords[0] >= lmp.system.xhi or coords[0] <= lmp.system.xlo or coords[1] >= SystemParams.parameters["old_bounds"][1]:
        lowest_pot = lmp.eval("pe")
        if lowest_pot < Helper.lowest_path_energy:
            Helper.lowest_path_energy = lowest_pot
        return lmp


    lowest_pot = float("inf")
    res_lmp = None
    prev_theta = theta

    for theta in np.linspace(*span):
        if prev_theta is None or np.pi - prev_theta + theta >= np.pi/2 and prev_theta + np.pi - theta >= np.pi/2:
            Helper.print("Theta: ", str(theta))
            Helper.mkdir(f"{theta}")
            Helper.chdir(f"{theta}")
            tmp_lmp =  QSR(lmp, coords = (coords[0] + dr*np.cos(theta), coords[1] + dr*np.sin(theta)), dr = dr, span = span, angles = (prev_theta, theta), potfile = potfile, in_glass = in_glass)
            Helper.chdir("..")
            if re.search(r"output\.\d+\.structure", ' '.join(os.listdir(f"{theta}"))):
                Helper.command(f"mv {theta}/output.*.structure .")
            #Helper.command(f"mv {theta}/positions.*.dump .")
            new_pe = tmp_lmp.eval("pe")
            Helper.print(f"lowest pe: {new_pe}")
            if new_pe < lowest_pot:
                if os.path.isfile(f"{theta}/log.lammps"):
                    Helper.command(f"mv {theta}/log.lammps tmp_log.lammps")
                Helper.command(f"mv {theta}/path.txt tmp_path.txt")
                selected_angle = theta
                lowest_pot = new_pe
                if res_lmp and res_lmp != lmp:
                    res_lmp.close()
                res_lmp = tmp_lmp
            elif tmp_lmp != lmp:
                tmp_lmp.close()

            Helper.rmtree(f"{theta}", ignore_errors = True)


    Helper.fracture_memory[coords] = Fake_lmp(lowest_pot)

    Helper.command("cat tmp_path.txt >> path.txt")
    Helper.command("cat tmp_log.lammps >> log.lammps")
    Helper.command("rm tmp_path.txt")
    Helper.command("rm tmp_log.lammps")
    return res_lmp

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
    #draw_lines(nodes, plt)

    box = tree.get_box()
    plt.plot([box[0][0], box[1][0]], [box[0][1], box[0][1]], color = "black")
    plt.plot([box[0][0], box[1][0]], [box[1][1], box[1][1]], color = "black")
    plt.plot([box[0][0], box[0][0]], [box[0][1], box[1][1]], color = "black")
    plt.plot([box[1][0], box[1][0]], [box[0][1], box[1][1]], color = "black")

    plt.plot(x, y, marker = 'o', linestyle = 'None', color = "green")
    plt.show()


def main():
    dr = 30
    error = 0.1
    dtheta = 30
    interactions = [(1, 2), (1, 3), (1, 4), (1, 5), (2, 4), (3, 5), (4, 5)]
    tree = FracTree(error = error, start_buffer = dr/2)
    tree.build(dtheta = dtheta, dr = dr, interactions = interactions)
    print("Number of nodes created:", len(tree))

    #visualize(tree, dr, dtheta)

    data_dir = "out_files" 
    if os.path.isdir(data_dir):
        os.system(f"rm -r {data_dir}")
    os.mkdir(data_dir)
    res = 0.69*tree.calculate(data_dir)





if __name__ == "__main__":
    main()
