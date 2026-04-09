# Cryo-ET-Data-Preparation-for-OrganelleSeg
Contains a variety of tools for data prep and pre-processing before passing into training or prediction of OrganelleSeg

## TODO
- Develop resolution/bin script to alter tomograms (if/how much noise to add?)

# File Descriptions
## cryoET_class_mapping.json and cryoET_containment_map.json
These files detail the class integer labels. For example, plasma membrane -> (1), nucleus -> (2), mitochondria -> (4), etc.
In particular, the containment map file shows which classes are included in which other classes. For example, the nuclear envelope class (9) is included in the nucleus class (2). As we add more classes and sub-classes to our annotation capabilities, we'll simply need to update these JSON files.

## remapping_and_collating_labels.py
This CLI file takes several 3D .mrc or .rec annotation files that were labeled using the author's labeling system. The user needs to figure out what the author's labeling system was. This script will ask the user to relabel the classes. It will then collate all the masks into a single 3D .mrc file (with labeling that matches the user's system). This 3D .mrc file is ready for 2D cropping.

## crop_tomo.py
This CLI file takes two files: the 3D reconstructed tomogram (.mrc or .rec) and the 3D annotation file (.mrc or .rec). As per user specifications, the script will slice the 3D structures along the z axis (skipping every set number of slices). The 2D slice will then be divided into a number of crops. The number of crops from each z-slice will depend on the desired pixel dimensions of the user. Crops with no annotations will be discarded. 2D crops from the original tomogram and from the annotations will be output in the specified directory as .tif files.

## annotations_check.py
This CLI file takes a 2D annotations .tif file and prints the unique values contained in it. This would simply be useful to make sure that the labels the user is expecting are actually found in the annotation crop.

# Dataset Descriptions
Since every dataset has a unique folder hierarchy and structure, not all the tools in this repo will be used and some may be used in different orders. Here, I explain the structure of each dataset and how to use the tools in preparation for OrganelleSeg.

## EMPIAR 10988
This dataset contains VPP/ and DEF/ directories. Each of these contains the following directories: frames/, labels/, metadata/, particle_lists/, tomograms/.
The tomograms/ directory contains 3D .rec tomograms.
The labels/ directory contains corresponding 3D .mrc files that label the various classes found within the specified tomogram.
The base names of the label files are the same as the base name for the tomogram they label (for example, TS_0001_membranes.mrc and TS_0001_organelles.mrc are two of the files that label the TS_0001.rec tomogram). The cyto_ribosomes.mrc, cytosol.mrc, FAS.mrc, and membranes.mrc files are all binary files (0 for background and 1 for the label) for cytosolic ribosomes, cytosol, fatty acid synthase, and membranes (respectively). The organelles.mrc file contains voxels that range in value from 0 to 13. Each of these values maps to a specific label:\
0       exterior\
1       cytoplas\
2       mitochondria\
3       vesicle\
4       tube\
5       ER\
6       nuclear envelope\
7       nucleus\
8       vacuole\
9       lipid droplet\
10      golgi\
11      vesicular body\
13      not identified compartment.\
Additionally, the particle_lists/ directory contains cyto_ribosomes and FAS coordinate CSV files.

To prepare this dataset, I used the remapping_and_collating_labels.py file and input all 5 label .mrc files. It output a 3D master_compiled_labels.mrc file with labels in accordance with our own labeling scheme (see JSON files). I then used this master label file along with the initial tomogram in the crop_tomo.py script. That script sliced the 3D files into 2D z-slices and cropped them. It named the crops in a corresponding manner and placed them in images/, labels/, and previews/ directories. 
