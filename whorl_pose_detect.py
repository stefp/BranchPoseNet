####################################################################################################################################
# Import libraries
import os, glob
import pandas as pd
import numpy as np
import io
import laspy
from PIL import Image
import re
from scipy.spatial import cKDTree
import concurrent.futures
import matplotlib
matplotlib.use('Agg')  # Use Agg backend for non-GUI rendering
import matplotlib.pyplot as plt


import ultralytics
from ultralytics import YOLO

# import my own functions
from tools import rotate_point_cloud,slice_tree_center_thick_slices,plot_to_image,plot_section_as_image_with_alpha
from tools import convert_to_real_world,process_file,calculate_angle_at_p2,calculate_distance,pose_detection_tree
from tools import convert_sections_to_images,process_point_cloud,get_image_size


####################################################################################################################################
# directory to where YOLO temp predictions are stored
dir_root="data" # path to root folder where .las/.laz files are stored

# alpha for plotting
alpha= 0.5

# minimum distance between whorls (used to remove duplicates)
min_internodal_d= 0.3 # in m (it can be set to something like 0.01 to allow the full output to post-process later)

# pose model
my_model="whorl_pose_nano_1000px/weights/best.pt"

# label for the column with tree instance unique identifiers
tree_id_label='treeID' 

# label for the column with semantic labels (non required)
semantic_label='semantic' 

####################################################################################################################################
# files_to_process= list_las_laz_files(dir_root)
# Find all .las and .laz files in the root folder
las_files = glob.glob(os.path.join(dir_root, '*.las'))
laz_files = glob.glob(os.path.join(dir_root, '*.laz'))

# Combine the lists of files
all_files = las_files + laz_files
all_files

# Iterate through all *.las/*.laz files in root folder
for filename in all_files:
    print("Starting pose estimation for all trees in: "+ filename)

    ####################################################################################################################################
    # define inner directories 
    dir_output = os.path.join(dir_root, "results")  # path to where to store the final results
    dir_pred = os.path.join(dir_root, "pred_temp")  # temp path to where to store the intermediate prediction
    dir_temp_imgs = os.path.join(dir_pred, "orig_imgs") # Further subdirectories within 'dir_pred'

    # Ensure directories exist without throwing an error if they already do
    os.makedirs(dir_pred, exist_ok=True)
    os.makedirs(dir_output, exist_ok=True)
    os.makedirs(dir_temp_imgs, exist_ok=True)
    
    ####################################################################################################################################
    # Read in the forest segmented data
    print("Reading in lidar data........................................................................................................")
    # Read in the forest segmented data
    las = laspy.read(filename)
    
    # convert to a numpy array 
    las_np = np.vstack((las.x, las.y, las.z,  getattr(las, tree_id_label),  getattr(las, semantic_label))).transpose()
    
    # split tree/non-tree (assumes that the semantic label corresponds to 0 if it's not a tree and 1 if it is a tree)
    trees= las_np[getattr(las, semantic_label)!=0]
    non_tree= las_np[getattr(las, semantic_label)==0]

    # Now, modifying the loop to use parallel execution
    unique_treeIDs = np.unique(trees[:,3])
    results = []
    
    ####################################################################################################################################
    # Whorl detection in parallel for all trees in the las file
    print("Whorl detection..............................................................................................................")
    # Parallel individual tree whorl pose detection
    unique_treeIDs = np.unique(getattr(las, tree_id_label))
    results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        # Submitting tasks to the executor
        future_to_treeID = {
            executor.submit(
                pose_detection_tree, 
                treeID=treeID,         # Pass treeID as a keyword argument
                trees=trees, 
                non_tree=non_tree, 
                dir_root=dir_root, 
                my_model=my_model, 
                dir_pred=dir_pred, 
                min_dist_whorls=min_internodal_d  # Pass min_dist_whorls as a keyword argument
            ): treeID 
            for treeID in unique_treeIDs
        }
        
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
    
    # remove negative whorls
    whorld_df= whorld_df[whorld_df['world_pz2']>=0]
    
    # write out the output
    whorld_df.to_csv(dir_output+'/'+os.path.splitext(os.path.basename(filename))[0]+'_'+str(round(min_internodal_d*100))+'cmThresh_whorls_pc_HKL2model.csv', index=False)  
  
    
    ####################################################################################################################################
    # Write outputs
    whorld_df.to_csv(dir_output+'/'+os.path.splitext(os.path.basename(filename))[0]+'_whorls_pc.csv', index=False) 

    ####################################################################################################################################
    # Cleanup temp files
    import shutil
    shutil.rmtree(dir_pred)
