"""
This script is responsible for:
- LOADING the scan that the user as inputted into the CLI (as a .nii.gz MRI file)
- READING the scans metadata (scan resolutions, voxel size, tracking mask of the patient (brain mask), how many directions the patient was scanned in...etc)
- PROCESSING the scans metadata into valid matrices of white matter, gray matter, where each scan direction was from (our dataset has 32 direction DTI scans) 
The MRI scan file inputted must be one of the pre-processed scans processed by I, Hugo Piper, for maximum compatibilty. This is because I precomputed
a brain mask from a T1 scan into the scans metadata as to keep purity of ACO-related scripts within this program, and to promote accuracy due to DTI scans being lower resolution.
"""

import os                       #For loading scans from the users pc
import nibabel                  #For loading the scans images
from dataclasses import field   #
import numpy as np              #For 

#Primary variables related to loading and processing scans
FORMAT_NAME = "ACO-ready"   #A name I put into the headers of the preprocessed scans so the program reconises valid scans.
FORMAT_VERSION = 1  #The version determines if a scan comes with or without T1 scan tissue maps (V2 adds them, files without T1 scan data stay V1). This is a compatibility feature 
INTENT_NAME = f"{FORMAT_NAME} v{FORMAT_VERSION}" #16 chars in a nifti file that can be added and will be displayed in the 3D viewer.
VOLUMES = ["fa", "tracking_mask", "brain_mask", "b0", "Dxx", "Dxy", "Dyy", "Dxz", "Dyz", "Dzz"] #The layout IN ORDER of the different data in the NIFTI (MRI) file. 
TISSUE_VOLUMES = ["wm", "gm", "csf"] #Incase a t1 file is added for accuracy, determines the quantity of WhiteMatter, GrayMatter, CerebralSpinalFluid
VOLUMES_V2 = VOLUMES + TISSUE_VOLUMES #Adds the DTI (nerve fiber) scan data and combines it with the T1 (brain tissue) scan data
TENSOR_SLICE = slice(4, 10) #Tells load() where the different tensor slices are (dxx, dxy, dyy, dxz, dyz, dzz)
TISSUE_SLICE = slice(10, 13)#Tells load() where the tissue slices are from the T1 scan (wm, gm, csf)
DESCRIP = "ACO-ready v1: FA,track mask,brain mask,b0,Dxx,Dxy,Dyy,Dxz,Dyz,Dzz"   #Layout of nifti (MRI) file without T1 metadata
DESCRIP_V2 = "ACO-ready v2: FA,track,brain,b0,Dxx,Dxy,Dyy,Dxz,Dyz,Dzz,WM,GM,CSF"#Layout of a nifti (MRI) file with T1 metadata
_ECODE_COMMENT = 6 #NIFTI files allow comments to be added in the metadata under code 6, I however don't think I'll encorprorate this in the final version.

class loadAndReadScanMetadata:

    #All the data the ACO script needs is held here. Extracted from scan metadata.
    fa: np.ndarray                  #Fractional Anistrophy (how freely water flows): (X,Y,Z) in float32 format.
    tracking_mask: np.ndarray       #Where the ants can walk within the brain: (X,Y,Z) boolean
    brain_mask: np.ndarray          #Where the brain is and isn't: (X,Y,Z) boolean
    b0: np.ndarray                  #b0 scan is a default scan with no diffusion weighting (basline scan): (X,Y,Z) float32, 0 to 1
    tensor_lt: np.ndarray           #Diffusion tensor at every voxel represented as a Lower Triangle (lt)
                                    #It represents how freely water flows in each direction as a 3x3 matrix. (X,Y,Z,6) float32: Dxx, Dxy, Dxz, Dyx, Dyy, Dyz, Dzx, Dzy, dzz
                                    #Eigenvector with largest eigenvalue is the direction water moves best. This is how we obtain FA and mean diffusivity.
    affine: np.ndarray              #A 4x4 matrix that maps voxel indices (i,j,k) to world coordinates in mm (x, y, z). Scale[i], roatation[j], translation[k], and axis flips.
    meta: dict = field(default_factory=dict) #Dictionary holding metadata
    path: str = ""                  #Path of the scan the user entered.
    zooms: tuple = None             #Voxel size (mm) from the scan header; if none in head, we get it from the affine.
    tissue: np.ndarray = None       #Tissue data from an optional T1 scan (X,Y,Z,3) float32 WM, GM, CSF from a T1. None = not available
    

    
    def load(path_user_entered: str):
        "Reads the file from the path that the user put into the CLI. Invalid file/data error handling happens here"

        if not os.path.exists(path_user_entered):
            raise FileNotFoundError(f"\nFile not found at path {path_user_entered}.\nPlease enter a valid path and try again.")
        print("valid path.")
        try:
            #Where the image is actually loaded from the users PC
            img = nibabel.load(path_user_entered)
        except Exception as exc:
            raise ValueError(f"{os.path.basename(path_user_entered)} is not a readable or valid NIFTI file.\nError code:\n{exc}")


userpath = str(input("Please enter a valid path:\n>>>"))
loadAndReadScanMetadata.load(userpath)

