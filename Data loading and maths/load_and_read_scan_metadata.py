"""
This script is responsible for:
- LOADING the scan that the user as inputted into the CLI (as a .nii.gz MRI file)
- READING the scans metadata (scan resolutions, voxel size, tracking mask of the patient (brain mask), how many directions the patient was scanned in...etc)
- PROCESSING the scans metadata into valid matrices of white matter, gray matter, where each scan direction was from (our dataset has 32 direction DTI scans) 
The MRI scan file inputted must be one of the pre-processed scans processed by I, Hugo Piper, for maximum compatibilty. This is because I precomputed
a brain mask from a T1 scan into the scans metadata as to keep purity of ACO-related scripts within this program, and to promote accuracy due to DTI scans being lower resolution.
"""

import os

def load(path_user_entered: str):
    "Reads the file from the path that the user put into the CLI. Invalid file/data error handling happens here"

    if not os.path.exists(path_user_entered):
        raise FileNotFoundError(f"\nFile not found at path {path_user_entered}.\nPlease enter a valid path and try again.")
    else:
        print("valid path")

userpath = str(input("Please enter a valid path:\n>>>"))
load(userpath)

