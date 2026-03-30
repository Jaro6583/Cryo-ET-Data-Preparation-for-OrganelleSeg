# Cryo-ET-Data-Preparation-for-OrganelleSeg
Contains a variety of tools for data prep and pre-processing before passing into training or prediction of OrganelleSeg

# File Descriptions
## cryoET_class_mapping.json and cryoET_containment_map.json
These files detail the class integer labels. For example, plasma membrane -> (1), nucleus -> (2), mitochondria -> (4), etc.
In particular, the containment map file shows which classes are included in which other classes. For example, the nuclear envelope class (9) is included in the nucleus class (2). As we add more classes and sub-classes to our annotation capabilities, we'll simply need to update these JSON files.

## remapping_and_collating_labels.py
This CLI file takes several 3D .mrc or .rec annotation files that were labeled using the author's labeling system. The user needs to figure out what the author's labeling system was. This script will ask the user to relabel the classes. It will then collate all the masks into a single 3D .mrc file (with labeling that matches the user's system). This 3D .mrc file is ready for 2D cropping.

## crop_tomo.py
This CLI file takes two files: the 3D reconstructed tomogram (.mrc or .rec) and the 3D annotation file (.mrc or .rec). As per user specifications, the script will slice the 3D structures along the z axis (skipping every set number of slices). The 2D slice will then be divided into a number of crops. The number of crops from each z-slice will depend on the desired pixel dimensions of the user. Crops with no annotations will be discarded. 
