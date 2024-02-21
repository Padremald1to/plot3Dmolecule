# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 19:49:33 2024

@author: Ramón Yáñez
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


coords_dict = {
'ferrocene':(
[[   0.,0.,0.],
 [   1.214056  ,  0.000000  , -1.636090],
          [  0.375164  ,  1.154636  , -1.636090],
          [  0.375164  , -1.154636  , -1.636090],
         [  1.214056  ,  0.000000  ,  1.636090],
          [ -0.982192  ,  0.713604  , -1.636090],
         [ -0.982192  , -0.713604  , -1.636090],
          [  0.375164  , -1.154636  ,  1.636090],
          [  0.375164  ,  1.154636  ,  1.636090],
        [ -0.982192  , -0.713604  ,  1.636090],
         [ -0.982192  ,  0.713604  ,  1.636090],
        [  2.302230  ,  0.000000  , -1.621820],
         [  0.711428  ,  2.189550  , -1.621820],
         [  0.711428  , -2.189550  , -1.621820],
         [ -1.862543  ,  1.353217  , -1.621820],
        [ -1.862543  , -1.353217  , -1.621820],
         [  2.302230  ,  0.000000  ,  1.621820],
         [  0.711428  , -2.189550  ,  1.621820],
         [  0.711428  ,  2.189550  ,  1.621820],
       [ -1.862543  , -1.353217  ,  1.62182],
       [ -1.862543  ,  1.353217  ,  1.621820]],
[26,6,6,6,6,6,6,6,6,6,6,1,1,1,1,1,1,1,1,1,1]),
}

covalent_radii={1:0.31,2:0.6,3:1.28,4:0.96,5:0.84,6:0.79,7:0.71,8:0.66,9:0.57,10:0.58,11:1.66,12:1.41,13:1.21,14:1.11,15:1.07,16:1.05,17:1.02,18:1.06,19:2.03,20:1.76,21:1.70,22:1.60,23:1.53,\
        24:1.39,25:1.61,26:1.52,27:1.50,28:1.24,29:1.32,30:1.22,31:1.22,32:1.20,33:1.19,34:1.20,35:1.20,36:1.16,37:2.20,38:1.95,39:1.90,40:1.75,41:1.64,42:1.54,43:1.47,44:1.46,45:1.42,\
        46:1.39,47:1.45,48:1.44,74:1.51,75:1.44,77:1.41,78:1.36,79:1.36,179:1.36
}

atno_to_element={1:"H",2:"He",3:"Li",4:"Be",5:"B",6:"C",7:"N",8:"O",9:"F",10:"Ne",11:"Na",12:"Mg",13:"Al",14:"Si",15:"P",16:"S",17:"Cl",18:"Ar",19:"K",20:"Ca",21:"Sc",22:"Ti",23:"V",\
        24:"Cr",25:"Mn",26:"Fe",27:"Co",28:"Ni",29:"Cu",30:"Zn",31:"Ga",32:"Ge",33:"As",34:"Se",35:"Br",36:"Kr",37:"Rb",38:"Sr",39:"Y",40:"Zr",41:"Nb",42:"Mo",43:"Tc",44:"Ru",45:"Rh",\
        46:"Pd",47:"Ag",48:"Cd",49:"In",50:"Sn",51:"Sb",52:"Te",53:"I",54:"Xe",55:"Cs",56:"Ba",47:"La",58:"Ce",59:"Pr",60:"Nd",61:"Pm",62:"Sm",63:"Eu",64:"Gd",65:"Tb",66:"Dy",67:"Ho",68:"Er",69:"Tm",70:"Yb",\
        71:"Lu",72:"Hf",73:"Ta",74:"W",75:"Re",77:"Ir",78:"Pt",79:"Au",179:"Au",100:"XX"
}

atom_colours={1:(255,255,255),2:(217,255,255),3:(204,128,255),4:(194,255,0),5:(255,181,181),6:(144,144,144),7:(48,80,248),8:(255,13,13),9:(144,224,80),10:(179,227,245),\
        11:(171,92,242),12:(138,255,0),13:(191,166,166),14:(240,200,160),15:(255,128,0),16:(255,255,48),17:(31,240,31),18:(128,209,227),19:(143,64,212),20:(61,255,0),\
        21:(230,230,230),22:(191,194,199),23:(166,166,171),24:(138,153,199),25:(156,122,199),\
        26:(224,102,51),27:(240,144,160),28:(80,208,80),29:(200,128,51),30:(125,128,176),\
        31:(194,143,143),32:(102,143,143),33:(189,128,227),34:(255,161,0),35:(166,41,41),\
        36:(92,184,209),37:(112,46,176),38:(0,255,0),39:(148,255,255),40:(148,224,224),\
        41:(115,194,201),42:(84,181,181),43:(59,158,158),44:(36,143,143),45:(10,125,140),\
        46:(0,105,133),47:(192,192,192),48:(255,217,143),\
        49:(166,117,115),50:(102,128,128),51:(158,99,181),52:(212,122,0),53:(148,0,148),\
        54:(66,158,176),55:(87,23,143),56:(0,201,0),57:(112,212,255),\
        58:(255,255,199),59:(217,255,199),60:(199,255,199),61:(163,255,199),\
        62:(143,255,199),63:(97,255,199),64:(69,255,199),65:(48,255,199),66:(31,255,199),\
        67:(0,255,156),68:(0,230,117),69:(0,212,82),70:(0,191,56),71:(0,171,36),\
        72:(77,194,255),73:(77,166,255),74:(33,148,214),75:(38,125,171),76:(38,102,150),\
        77:(23,84,135),78:(208,208,224),79:(255,209,35),\
        179:(255,209,35)
}


def set_axes_limits(ax, coordinates):
    """
    Set the axis limits based on the coordinates.

    Args:
    ax (Axes3D): The 3D axis object.
    coordinates (list of tuples): The coordinates of the atoms.
    """
    xcoords, ycoords, zcoords = zip(*coordinates)
    max_range = np.array([max(xcoords) - min(xcoords), max(ycoords) - min(ycoords), max(zcoords) - min(zcoords)]).max() / 2.0
    mean_x, mean_y, mean_z = np.mean(xcoords), np.mean(ycoords), np.mean(zcoords)
    ax.set_xlim(mean_x - max_range, mean_x + max_range)
    ax.set_ylim(mean_y - max_range, mean_y + max_range)
    ax.set_zlim(mean_z - max_range, mean_z + max_range)

def plot_atoms(ax, coordinates, atoms, atom_colours, covalent_radii):
    """
    Plot the atoms in the 3D plot.

    Args:
    ax (Axes3D): The 3D axis object.
    coordinates (list of tuples): The coordinates of the atoms.
    atoms (list): The atomic numbers of the atoms.
    atom_colours (dict): Mapping of atomic numbers to RGB colors.
    covalent_radii (dict): Mapping of atomic numbers to covalent radii.
    """
    for i, coord in enumerate(coordinates):
        if atoms[i] == 1:
            atom_colour = 'k'
        else:
            atom_colour = [x / 255 for x in atom_colours[atoms[i]]]
        size = covalent_radii[atoms[i]] * 100
        ax.scatter(coord[0], coord[1], coord[2], color=atom_colour, s=size)

def plot_bonds(ax, coordinates, atoms, covalent_radii):
    """
    Plot the bonds between atoms in the 3D plot.

    Args:
    ax (Axes3D): The 3D axis object.
    coordinates (list of tuples): The coordinates of the atoms.
    atoms (list): The atomic numbers of the atoms.
    covalent_radii (dict): Mapping of atomic numbers to covalent radii.
    """

    for i in range(len(coordinates)):
        for j in range(i, len(coordinates)):
            if i < j:
                vec_ab = np.array(coordinates[j]) - np.array(coordinates[i])
                dist_ab = np.linalg.norm(vec_ab)
                sum_of_radii = (covalent_radii[atoms[i]] + covalent_radii[atoms[j]]) / 0.6
                if dist_ab < 1.2 * sum_of_radii:
                    ax.plot([coordinates[j][0], coordinates[i][0]], [coordinates[j][1], coordinates[i][1]], [coordinates[j][2], coordinates[i][2]], color="green", lw=1)

def plot_molecule(molecule, coordinates, atoms, atom_colours, covalent_radii):
    """
    Plot the molecule in a 3D plot.

    Args:
    molecule (str): The name of the molecule.
    coordinates (list of lists): The coordinates of the atoms.
    atoms (list): The atomic numbers of the atoms.
    atom_colours (dict): Mapping of atomic numbers to RGB colors.
    covalent_radii (dict): Mapping of atomic numbers to covalent radii.
    """
    

    fig = plt.figure(figsize = (16,12))
   
    ax = fig.add_subplot(111, projection='3d')

    set_axes_limits(ax, coordinates)
    plot_atoms(ax, coordinates, atoms, atom_colours, covalent_radii)
    plot_bonds(ax, coordinates, atoms, covalent_radii)

    ax.set_title(molecule)
    plt.show()

# Example usage

molecule = "ferrocene"
coords = np.array(coords_dict[molecule][0])
result = coords / 0.6
coordinates = result.tolist()
atoms = coords_dict[molecule][1]

plot_molecule(molecule, coordinates, atoms, atom_colours, covalent_radii)