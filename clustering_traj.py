"""
This script takes a trajectory and based on a minimal RMSD classify the structures in clusters.

The RMSD implementation using the Kabsch algorithm to superpose the molecules is taken from: https://github.com/charnley/rmsd
A very good description of the problem of superposition can be found at http://cnx.org/contents/HV-RsdwL@23/Molecular-Distance-Measures
A very good tutorial on hierachical clustering with scipy can be seen at https://joernhees.de/blog/2015/08/26/scipy-hierarchical-clustering-and-dendrogram-tutorial/
This script performs agglomerative clustering as suggested in https://stackoverflow.com/questions/31085393/hierarchical-clustering-a-pairwise-distance-matrix-of-precomputed-distances

Author: Henrique Musseli Cezar
Date: NOV/2017
"""

import os
import sys
import argparse
import numpy as np
import rmsd
import pybel
import openbabel
import scipy.cluster.hierarchy as hcl
from scipy.spatial.distance import squareform
from sklearn import manifold
import matplotlib as mpl
import matplotlib.pyplot as plt
import multiprocessing
import itertools


# https://stackoverflow.com/a/25830011/3254658
def initialize_adists(_adists):
  global adists
  adists = _adists


def initialize_mollist(_mollist):
  global mollist
  mollist = _mollist


def initialize_mollist_distmat(_mollist, _distmat, _clusters):
  global mollist, distmat, clusters
  mollist = _mollist
  distmat = _distmat
  clusters = _clusters


def get_atom_dist(r1, r2):
  return np.linalg.norm(np.asarray(r1)-np.asarray(r2))


def get_adists_mol(idxmol):
  adist = []

  for idx1, atom1 in enumerate(mollist[idxmol].atoms):
    for idx2, atom2 in enumerate(mollist[idxmol].atoms):
      if idx1 <= idx2:
        continue

      adist.append(get_atom_dist(atom1.coords, atom2.coords))

  return sorted(adist)


def get_adists_mol_sp(idxmol):
  adist = {}

  for idx1, atom1 in enumerate(mollist[idxmol].atoms):
    for idx2, atom2 in enumerate(mollist[idxmol].atoms):
      if idx1 <= idx2:
        continue

      if atom1.atomicnum < atom2.atomicnum:
        spsp = int(str(atom1.atomicnum)+str(atom2.atomicnum))
      else:
        spsp = int(str(atom2.atomicnum)+str(atom1.atomicnum))

      if spsp not in adist:
        adist[spsp] = [get_atom_dist(atom1.coords, atom2.coords)]
      else:
        adist[spsp].append(get_atom_dist(atom1.coords, atom2.coords))

  radists = []
  for spsp in sorted(adist):
    radists += sorted(adist[spsp])

  return radists


def build_distance_matrix(mollist, nprocs):
  # calculate the atom distances
  pp = multiprocessing.Pool(processes = nprocs, initializer = initialize_mollist, initargs = (mollist,))
  adists = pp.map(get_adists_mol, range(len(mollist)), 50)
  pp.close()

  # build the distance matrix in parallel
  p = multiprocessing.Pool(processes = nprocs, initializer = initialize_adists, initargs = (adists,))
  ldistmat = p.map(compute_distmat_line, range(len(adists)), 50)
  p.close()

  return np.asarray([x for n in ldistmat if len(n) > 0 for x in n])


def compute_distmat_line(idx1):
  # initialize distance matrix
  distmat = []

  for idx2 in range(idx1+1, len(adists)):
    # get the distance and store
    distmat.append(np.sum(np.square(np.asarray(adists[idx1])-np.asarray(adists[idx2]))))

  return distmat


def save_clusterfile(cidx, outbasename, outfmt):
  # create object to output the configurations
  outfile = pybel.Outputfile(outfmt, outbasename+"_"+str(cidx)+"."+outfmt)

  # creates mask with True only for the members of cluster number cidx
  mask = np.array([1 if i==cidx else 0 for i in clusters], dtype=bool)

  # gets the member with smallest sum of distances from the submatrix
  idx = np.argmin(sum(distmat[:,mask][mask,:]))

  # get list with the members of this cluster only and store medoid
  sublist=[num for (num, cluster) in enumerate(clusters) if cluster==cidx]
  medoid = sublist[idx]

  # print the medoid coordinates
  outfile.write(mollist[medoid])

  # print the coordinates of the other NPs
  for idx in sublist:
    outfile.write(mollist[idx])

  # closes the file for the cidx cluster
  outfile.close()


def save_clusters_config(mollist, clusters, distmat, outbasename, outfmt, nprocs):
  # complete distance matrix
  sqdistmat = squareform(distmat)
  
  p = multiprocessing.Pool(processes = nprocs, initializer = initialize_mollist_distmat, initargs = (mollist, sqdistmat, clusters))
  p.starmap_async(save_clusterfile, zip(range(1,max(clusters)+1), itertools.repeat(outbasename), itertools.repeat(outfmt)))


def check_positive(value):
  ivalue = int(value)
  if ivalue <= 0:
       raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
  return ivalue


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Run a clustering analysis on a trajectory based on the minimal RMSD obtained with a Kabsch superposition.')
  parser.add_argument('trajectory_file', help='path to the trajectory containing the conformations to be classified')
  parser.add_argument('min_rmsd', help='value of RMSD used to classify structures as similar')
  parser.add_argument('-np', '--nprocesses', metavar='NPROCS', type=check_positive, default=2, help='defines the number of processes used to compute the distance matrix and multidimensional representation (default = 2)')
  parser.add_argument('-p', '--plot', action='store_true', help='enable the dendrogram plot saving the figures in pdf format (filenames use the same basename of the -oc option)')
  parser.add_argument('-pmds', '--plot-mds', action='store_true', help='plot the multidimensional scaling too')
  parser.add_argument('-m', '--method', metavar='METHOD', default='average', help="method used for clustering (see valid methods at https://docs.scipy.org/doc/scipy-0.19.1/reference/generated/scipy.cluster.hierarchy.linkage.html) (default: average)")
  parser.add_argument('-cc', '--clusters-configurations', metavar='EXTENSION', help='save superposed configurations for each cluster in EXTENSION format (basename based on -oc option)')
  parser.add_argument('-oc', '--outputclusters', default='clusters.dat', metavar='FILE', help='file to store the clusters (default: clusters.dat)')

  io_group = parser.add_mutually_exclusive_group()
  io_group.add_argument('-i', '--input', type=argparse.FileType('rb'), metavar='FILE', help='file containing input distance matrix in condensed form')
  io_group.add_argument('-od', '--outputdistmat', metavar='FILE', help='file to store distance matrix in condensed form (default: distmat.dat)')

  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)

  args = parser.parse_args()

  # check input consistency manually since I did not find a good way to use FileType and have the behavior that I wanted
  if args.method not in ["single","complete","average","weighted","centroid","median","ward"]:
    print("The method you selected with -m (%s) is not valid." % args.method)
    sys.exit(1)

  if args.clusters_configurations:
    if args.clusters_configurations not in ["acr", "adf", "adfout", "alc", "arc", "bgf", "box", "bs", "c3d1", "c3d2", "cac", "caccrt", "cache", "cacint", "can", "car", "ccc", "cdx", "cdxml", "cht", "cif", "ck", "cml", "cmlr", "com", "copy", "crk2d", "crk3d", "csr", "cssr", "ct", "cub", "cube", "dmol", "dx", "ent", "fa", "fasta", "fch", "fchk", "fck", "feat", "fh", "fix", "fpt", "fract", "fs", "fsa", "g03", "g92", "g94", "g98", "gal", "gam", "gamin", "gamout", "gau", "gjc", "gjf", "gpr", "gr96", "gukin", "gukout", "gzmat", "hin", "inchi", "inp", "ins", "jin", "jout", "mcdl", "mcif", "mdl", "ml2", "mmcif", "mmd", "mmod", "mol", "mol2", "molden", "molreport", "moo", "mop", "mopcrt", "mopin", "mopout", "mpc", "mpd", "mpqc", "mpqcin", "msi", "msms", "nw", "nwo", "outmol", "pc", "pcm", "pdb", "png", "pov", "pqr", "pqs", "prep", "qcin", "qcout", "report", "res", "rsmi", "rxn", "sd", "sdf", "smi", "smiles", "sy2", "t41", "tdd", "test", "therm", "tmol", "txt", "txyz", "unixyz", "vmol", "xed", "xml", "xyz", "yob", "zin"]:
      print("The format you selected to save the clustered superposed configurations (%s) is not valid." % args.clusters_configurations)
      sys.exit(1)

  if not args.input:
    if not args.outputdistmat:
      args.outputdistmat = "distmat.dat"

    if os.path.exists(args.outputdistmat):
      exit("File %s already exists, specify a new filename with the -od command option. If you are trying to read the distance matrix from a file, use the -i option" % args.outputdistmat)
    else:
      args.outputdistmat = open(args.outputdistmat,'wb')

  if os.path.exists(args.outputclusters):
    exit("File %s already exists, specify a new filename with the -oc command option" % args.outputclusters)
  else:
    args.outputclusters = open(args.outputclusters,'wb')

  # check if distance matrix will be read from input or calculated
  # if a file is specified, read it (TODO: check if the matrix makes sense)
  if args.input:
    print('\nReading condensed distance matrix from %s\n' % args.input.name)
    distmat = np.loadtxt(args.input)
  # build a distance matrix already in the condensed form
  else:
    print('\nReading trajectory\n')
    # create list with all the mol objects
    mollist = list(pybel.readfile(os.path.splitext(args.trajectory_file)[1][1:], args.trajectory_file))
    print('Calculating distance matrix\n')
    distmat = build_distance_matrix(mollist, args.nprocesses)
    print('Saving condensed distance matrix to %s\n' % args.outputdistmat.name)
    np.savetxt(args.outputdistmat, distmat, fmt='%.18f')
    args.outputdistmat.close()

  # linkage
  print("Starting clustering using '%s' method to join the clusters\n" % args.method)
  Z = hcl.linkage(distmat, args.method)

  # build the clusters and print them to file
  clusters = hcl.fcluster(Z, float(args.min_rmsd), criterion='distance')
  print("Saving clustering classification to %s\n" % args.outputclusters.name)
  np.savetxt(args.outputclusters, clusters, fmt='%d')
  args.outputclusters.close()

  # get the elements closest to the centroid (see https://stackoverflow.com/a/39870085/3254658)
  if args.clusters_configurations:
    if not mollist:
      mollist = list(pybel.readfile(os.path.splitext(args.trajectory_file)[1][1:], args.trajectory_file))
    print("Writing superposed configurations per cluster to files %s\n" % (os.path.splitext(args.outputclusters.name)[0]+"_confs"+"_*"+"."+args.clusters_configurations))
    save_clusters_config(mollist, clusters, distmat, os.path.splitext(args.outputclusters.name)[0]+"_confs", args.clusters_configurations, args.nprocesses)

  if args.plot:
    print("Plotting the data\n")
    # plot evolution with o cluster in trajectory
    plt.figure(figsize=(25, 10))
    plt.plot(range(1,len(clusters)+1), clusters, "o-", markersize=4)
    plt.xlabel('Sample Index')
    plt.ylabel('Cluster classification')
    plt.savefig(os.path.splitext(args.outputclusters.name)[0]+"_evo.pdf", bbox_inches='tight')

    # plot the dendrogram
    plt.figure(figsize=(25, 10))
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Sample Index')
    plt.ylabel('RMSD')
    hcl.dendrogram(
      Z,
      leaf_rotation=90.,  # rotates the x axis labels
      leaf_font_size=8.,  # font size for the x axis labels
    )
    plt.axhline(float(args.min_rmsd),linestyle='--')
    plt.savefig(os.path.splitext(args.outputclusters.name)[0]+"_dendrogram.pdf", bbox_inches='tight')

    # finds the 2D representation of the distance matrix (multidimensional scaling) and plot it
    # plt.figure()
    # mds = manifold.MDS(n_components=2, dissimilarity="precomputed", random_state=666, n_init=6, max_iter=300, eps=1e-3, n_jobs=args.nprocesses)
    # coords = mds.fit_transform(squareform(distmat))
    # plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labelleft='off')
    # plt.scatter(coords[:, 0], coords[:, 1], marker = 'o', c=clusters, cmap=plt.cm.nipy_spectral)
    # plt.savefig(os.path.splitext(args.outputclusters.name)[0]+".pdf", bbox_inches='tight')


  # print the cluster sizes
  print("A total of %d cluster(s) was(were) found.\n" % max(clusters))

  print("A total of %d structures were read from the trajectory. The cluster sizes are:" % len(clusters))
  print("Cluster\tSize")
  labels, sizes = np.unique(clusters, return_counts=True)
  for label, size in zip(labels,sizes):
    print("%d\t%d" % (label,size))

  print()