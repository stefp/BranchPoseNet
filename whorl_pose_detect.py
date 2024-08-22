####################################################################################################################################
# Import libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import laspy
from PIL import Image
import re
from scipy.spatial import cKDTree
import concurrent.futures

import ultralytics
from ultralytics import YOLO

# import my own functions
from tools import rotate_point_cloud,slice_tree_center_thick_slices,plot_to_image,plot_section_as_image_with_alpha
from tools import convert_to_real_world,process_file,calculate_angle_at_p2,calculate_distance,pose_detection_tree
from tools import convert_sections_to_images,process_point_cloud,get_image_size,list_las_laz_files


####################################################################################################################################
# directory to where YOLO temp predictions are stored
dir_root="C:/Users/stpu/YOLOv5-whorlDetector/test_forest" # path to root folder where .las/.laz files are stored
# alpha for plotting
alpha= 0.5
# pose model
my_model="C:/Users/stpu/YOLOv5-whorlDetector/pose_estimation/whorl_pose_n_640/weights/best.pt"
# minimum distance between whorls (used to remove duplicates in postprocessing)
min_dist_whorls=0.25 # in m

# define files to process
files_to_process= list_las_laz_files(dir_root)

####################################################################################################################################
# Iterate through all *.las/*.laz files in root folder
for filename in files_to_process:
    print("Starting pose estimation for all trees in: "+ filename)

    ####################################################################################################################################
    # define inner directories 
    dir_output=dir_root+"/results" # path to where to store the final results
    dir_pred=dir_root+"/pred_temp" # temp path to where to store the intermediate prediction
    
    # create directories
    if not os.path.exists(dir_pred):
       os.makedirs(dir_pred)
    if not os.path.exists(dir_output):
       os.makedirs(dir_output)
        
    ####################################################################################################################################
    # Read in the forest segmented data
    print("Reading in lidar data........................................................................................................")
    one_forest=dir_root+"/"+filename
    las = laspy.read(one_forest)
    
    # convert to a numpy array 
    las_np = np.vstack((las.x, las.y, las.z, las.preds_instance_segmentation, las.preds_semantic_segmentation)).transpose()
    
    # split tree/non-tree
    trees= las_np[las_np[:,4]==1]
    non_tree= las_np[las_np[:,4]==0]
    
    #print("tree points: "+ str(round(len(trees)/len(las_np)*100))+"%")
    #print("Non-tree points: "+ str(round(len(non_tree)/len(las_np)*100))+"%")

    # Now, modifying the loop to use parallel execution
    unique_treeIDs = np.unique(trees[:,3])
    results = []
    
    ####################################################################################################################################
    # Parallel processing of all trees in a forest
    print("Parallel processing..........................................................................................................")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        # Submitting tasks to the executor
        future_to_treeID = {executor.submit(pose_detection_tree, treeID, trees, non_tree,dir_root, my_model,min_dist_whorls=0.25): treeID for treeID in unique_treeIDs}
        
        # Collecting results as they complete
        for future in concurrent.futures.as_completed(future_to_treeID):
            treeID = future_to_treeID[future]
            try:
                data = future.result()
                if data is not None:
                    results.append(data)
            except Exception as exc:
                print(f'TreeID {treeID} generated an exception: {exc}')
    
    
    ####################################################################################################################################
    # Post-processing the results
    # Creating a dictionary mapping treeID to its result
    results_dict = {result['treeID']: result['result'] for result in results if result is not None}
    
    # Merging the 'whorl_pc' DataFrames from each result
    whorls_pc_dfs = [result['whorl_pc'] for result in results if result is not None and 'whorl_pc' in result]
    whorld_df = pd.concat(whorls_pc_dfs, ignore_index=True)
    
    # correct z
    whorld_df.loc[whorld_df['branch_opening_angle'] == 0, 'z'] = whorld_df['world_pz2']
    
    
    ####################################################################################################################################
    # Write outputs
    whorld_df.to_csv(dir_output+'/'+os.path.splitext(os.path.basename(filename))[0]+'_whorls_pc.csv', index=False) 

    ####################################################################################################################################
    # Cleanup temp files
    import shutil
    shutil.rmtree(dir_pred)
