#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False
from libc.float cimport DBL_MAX

import numpy as np
cimport numpy as cnp

cnp.import_array()

def _enforce_connectivity_labels_with_mask(Py_ssize_t[:, :, ::1] segments,
                                     Py_ssize_t[:, :, ::1] mask,
                                     Py_ssize_t[:, :, ::1] artefacts,
                                     Py_ssize_t[:, :, ::1] min_size_map,
                                     Py_ssize_t[:, :, ::1] max_size_map,
                                     Py_ssize_t max_size,
                                     Py_ssize_t[:] n_sp_vect,
                                     Py_ssize_t start_label=0):
    """ Helper function to remove small disconnected regions from the labels
    Parameters
    ----------
    segments : 3D array of int, shape (Z, Y, X)
        The label field/superpixels found by SLIC.
    min_size_map : int
        The minimum size of the segment
    max_size_map : int
        The maximum size of the segment. This is done for performance reasons,
        to pre-allocate a sufficiently large array for the breadth first search
    start_label : int
        The label indexing start value.
    mask : 3D array of int, shape (Z, Y, X), optional
        The mask defining different objects in the image.
    region_to_object : 1D array of int, optional
        Array mapping region labels to object IDs.

    Returns
    -------
    connected_segments : 3D array of int, shape (Z, Y, X)
        A label field with connected labels starting at label=1
    """

    # get image dimensions
    cdef Py_ssize_t depth, height, width
    depth = segments.shape[0]
    height = segments.shape[1]
    width = segments.shape[2]

    # neighborhood arrays
    cdef Py_ssize_t[::1] ddx = np.array((1, -1, 0, 0, 0, 0), dtype=np.intp)
    cdef Py_ssize_t[::1] ddy = np.array((0, 0, 1, -1, 0, 0), dtype=np.intp)
    cdef Py_ssize_t[::1] ddz = np.array((0, 0, 0, 0, 1, -1), dtype=np.intp)

    # new object with connected segments initialized to mask_label
    cdef Py_ssize_t mask_label = start_label - 1

    cdef Py_ssize_t[:, :, ::1] connected_segments = np.full_like(segments, mask_label, dtype=np.intp)

    cdef Py_ssize_t current_new_label = start_label
    cdef Py_ssize_t label = start_label

    # variables for the breadth first search
    cdef Py_ssize_t current_segment_size = 1
    cdef Py_ssize_t bfs_visited = 0
    cdef Py_ssize_t adjacent
    cdef Py_ssize_t n_sp_by_object

    cdef Py_ssize_t zz, yy, xx
    cdef Py_ssize_t z, y, x  # Définition explicite de z, y et x en tant que Py_ssize_t
    cdef Py_ssize_t zi, yi, xi  # Définition explicite de z, y et x en tant que Py_ssize_t

    # Variable pour stocker l'ID d'objet (déclaration ici pour l'utilisation dans les boucles)
    cdef Py_ssize_t object_id, is_artefact, min_size

    # Créer un tableau de coordonnées avec une taille maximale préallouée
    cdef Py_ssize_t[:, ::1] coord_list = np.empty((max_size, 3), dtype=np.intp)

    cdef Py_ssize_t count, tmp, 

    count = 0
    tmp = -1
    with nogil:  # Moved nogil back after explicit definition of x, y, z
        while (count < depth*height*width):
            tmp = tmp + 1
            #with gil:  # Récupérer le GIL pour imprimer
            #    print(f"HELLO count {count} {min_size} {tmp}")
            for z in range(depth):
                for y in range(height):
                    for x in range(width):

                        if segments[z, y, x] == mask_label:
                            continue

                        if connected_segments[z, y, x] > mask_label:
                            continue
                    
                        #if (count>0):
                        #    with gil:  # Récupérer le GIL pour imprimer
                        #        print(f"HELLO xx: {x}, yy: {y}, is_art : {is_artefact}, {current_new_label}")
        



                        # Obtenir l'ID d'objet correspondant à la région actuelle
                        #if region_to_object is not None:
                        #    #object_id = region_to_object[segments[z, y, x]]
                        object_id = mask[z, y, x]
                        min_size = min_size_map[z, y, x]
                        max_size = min_size_map[z, y, x]
                        n_sp_by_object = 2 #by default more than one sp in an object
                        #else:
                        #    object_id = -1  # Si aucun masque n'est fourni

                        
                        #if (count>0):
                        #    with gil:  # Récupérer le GIL pour imprimer
                        #        print(f"{min_size}")
        

                        # find the component size
                        is_artefact = artefacts[z, y, x]
                        adjacent = mask_label
                        label = segments[z, y, x]
                        connected_segments[z, y, x] = current_new_label
                        current_segment_size = 1
                        bfs_visited = 0
                        coord_list[bfs_visited, 0] = z
                        coord_list[bfs_visited, 1] = y
                        coord_list[bfs_visited, 2] = x

                        #if (x==51 and y==37):
                        #    with gil:  # Récupérer le GIL pour imprimer
                        #        print(f"HELLO xx: {x}, yy: {y}, is_art : {is_artefact}")
        

                        # perform a breadth first search to find
                        # the size of the connected component
                        while bfs_visited < current_segment_size < max_size:
                            for i in range(6):
                                zz = coord_list[bfs_visited, 0] + ddz[i]
                                yy = coord_list[bfs_visited, 1] + ddy[i]
                                xx = coord_list[bfs_visited, 2] + ddx[i]
                                if 0 <= yy < height and 0 <= zz < depth and 0 <= xx < width:

                                    #if (label==9 and x==51 and y==37):
                                    #    with gil:  # Récupérer le GIL pour imprimer
                                    #        print(f"xx: {xx}, yy: {yy}, is_art : {is_artefact}")
        
                                    # Vérification de l'appartenance à la même région
                                    if (segments[zz, yy, xx] == label and
                                        connected_segments[zz, yy, xx] == mask_label): # and mask[zz, yy, xx] == object_id):
                                            connected_segments[zz, yy, xx] = current_new_label
                                            coord_list[current_segment_size, 0] = zz
                                            coord_list[current_segment_size, 1] = yy
                                            coord_list[current_segment_size, 2] = xx
                                            #On check si dans cette petite région on a que des pixels artefacts.
                                            #Si oui on peut faire ce qu'on veut.
                                            #Si non, on a normalement qu'un seul pixel objet
                                            if (artefacts[zz,yy,xx]==0):
                                                is_artefact = 0
                                                #object_id = region_to_object[segments[zz,yy,xx]]
                                                object_id = mask[zz,yy,xx]
                                                min_size = min_size_map[zz,yy,xx]
                                                max_size = max_size_map[zz,yy,xx]
                                                n_sp_by_object = n_sp_vect[segments[zz, yy, xx]]
                                            current_segment_size += 1
                                            if current_segment_size >= max_size:
                                                break

                                    #elif (connected_segments[zz, yy, xx] > mask_label and
                                    #    connected_segments[zz, yy, xx] != current_new_label):
                                    #    if ((artefacts[zz, yy, xx]) or (mask[zz, yy, xx] == object_id)):
                                    #        adjacent = connected_segments[zz, yy, xx]
                            bfs_visited += 1

                        #if only too small connected regions
                        if (tmp >= 2):
                            min_size /= tmp

                        #if only one sp by object whose size is inferior to the threshold, keep it
                        if (n_sp_by_object == 1):
                            min_size = current_segment_size

                        #En théorie impossible d'avoir des pixels déconnectés qui n'ont pas de voisin de la même région
                        #car petites régions déjà virées. Et que des objets connexes.
                        #if (count>0):
                        #    with gil:  # Récupérer le GIL pour imprimer
                        #        print(f"BLOU is {is_artefact}, object {object_id}, size {current_segment_size}, min {min_size}, {coord_list[0,1]}, {coord_list[0,2]}")


                        # Vérification de la taille de la composante
                        if current_segment_size < min_size:

                            # Réinitialisation de adjacent pour éviter une attribution incorrecte
                            adjacent = mask_label
                            # Recherche des voisins valides
                            for i in range(current_segment_size):
                                zi = coord_list[i, 0]
                                yi = coord_list[i, 1]
                                xi = coord_list[i, 2]
                                if (artefacts[zi,yi,xi]==0):
                                    is_artefact = 0
                                    object_id = mask[zi,yi,xi]
                                #if (count>0):
                                #    with gil:  # Récupérer le GIL pour imprimer
                                #        print(f"xi: {xi}, yi: {yi}, is {is_artefact}, {object_id}")


                            for i in range(current_segment_size):
                                zi = coord_list[i, 0]
                                yi = coord_list[i, 1]
                                xi = coord_list[i, 2]
                                
                                for j in range(6):
                                    zz = zi + ddz[j]
                                    yy = yi + ddy[j]
                                    xx = xi + ddx[j]
                                    #if (count>0):
                                    #    #with gil:  # Récupérer le GIL pour imprimer
                                    #    #    print(f"xx: {xx}, yy: {yy}, {object_id}, {is_artefact}, {mask[zz, yy, xx]}, {artefacts[zz,yy,xx]}")
                                    #     #   print(f"connected : {connected_segments[zz, yy, xx]}, mask_label: {mask_label}, current_new={current_new_label}")
                                    if 0 <= xx < width and 0 <= yy < height and 0 <= zz < depth:
                                        if ((connected_segments[zz, yy, xx] > mask_label) and
                                            (connected_segments[zz, yy, xx] != current_new_label) and
                                            ((is_artefact) or (artefacts[zz,yy,xx]==0 and mask[zz, yy, xx] == object_id))):
                                            adjacent = connected_segments[zz, yy, xx]
                                            break
                                if adjacent != mask_label:
                                    break

                            # Si un voisin valide a été trouvé, fusionner
                            if adjacent != mask_label:
                                #if (label==97):
                                #    with gil:  # Récupérer le GIL pour imprimer
                                #        print(f"adj {adjacent}")
                                for i in range(current_segment_size):
                                    zi = coord_list[i, 0]
                                    yi = coord_list[i, 1]
                                    xi = coord_list[i, 2]
                                    connected_segments[zi, yi, xi] = adjacent
                                count += current_segment_size

                            else: #petite région qui n'a trouvé personne de labellisé dans le masque encore
                                for i in range(current_segment_size):
                                    zi = coord_list[i, 0]
                                    yi = coord_list[i, 1]
                                    xi = coord_list[i, 2]
                                    #with gil:  # Récupérer le GIL pour imprimer
                                    #    print(f"not found xi: {xi}, yy: {yi}")
                                    connected_segments[zi, yi, xi] = mask_label

                        else:
                            count += current_segment_size
                            current_new_label += 1

    return np.asarray(connected_segments)