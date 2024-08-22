# Useful Functions for whorl pose estimation

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

import ultralytics
from ultralytics import YOLO

####################################################################################################################################
# Define custom functions
def rotate_point_cloud(point_cloud, angle_degrees, center_point):
    """
    Rotate the point cloud around a center point by a given angle in degrees.
    """
    theta = np.radians(angle_degrees)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    rotated_points = np.dot(point_cloud[:, :2] - center_point, R.T) + center_point
    return np.hstack((rotated_points, point_cloud[:, 2].reshape(-1, 1)))

def slice_tree_center_thick_slices(point_cloud, slice_thickness=10):
    """
    Take thick slices in the X and Y directions, centered around the tree's center.
    """
    tree_center = point_cloud[point_cloud[:,2].argmax(), :2]
    x_slice_mask = (point_cloud[:,0] >= tree_center[0] - slice_thickness/2) & \
                   (point_cloud[:,0] <= tree_center[0] + slice_thickness/2)
    y_slice_mask = (point_cloud[:,1] >= tree_center[1] - slice_thickness/2) & \
                   (point_cloud[:,1] <= tree_center[1] + slice_thickness/2)
    x_slice = point_cloud[x_slice_mask]
    y_slice = point_cloud[y_slice_mask]
    return x_slice, y_slice

def plot_to_image(figure, dpi):
    """
    Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it as a numpy array, setting the resolution with a high DPI.
    """
    buf = io.BytesIO()
    figure.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=dpi)
    plt.close(figure)
    buf.seek(0)
    image = Image.open(buf)
    return np.array(image)

def plot_section_as_image_with_alpha(slice_data, z_low, z_high, alpha=0.3, output_size=(1000, 1000), dpi=100):
    """
    Create a figure and plot the slice_data with alpha transparency.
    Dynamically adjusts plot limits based on the data and resizes the output image to a square format.
    """
    if slice_data.size == 0:
        return None  # Return None if there are no data points to plot.

    # Determine aspect ratio and figure size
    buffer = 0 # Add a buffer around data extents
    x_min, x_max = np.min(slice_data[:, 0]) - buffer, np.max(slice_data[:, 0]) + buffer
    y_min, y_max = np.min(slice_data[:, 2]) - buffer, np.max(slice_data[:, 2]) + buffer

    # Dynamically adjust xlim and ylim to include all points and maintain real tree dimensions
    x_range = x_max - x_min
    y_range = z_high - z_low  # This should be close to section_height if properly sliced

    # Determine the scale factor to use for x and y to maintain aspect ratio
    if x_range > y_range:
        scale_factor = x_range / y_range
        fig_width, fig_height = 10 * scale_factor, 10
    else:
        scale_factor = y_range / x_range
        fig_width, fig_height = 10, 10 * scale_factor

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.scatter(slice_data[:, 0], slice_data[:, 2], s=3, color='black', alpha=alpha, edgecolors='none')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('auto')  # 'auto' allows free aspect ratio that adjusts to specified limits

    ax.axis('off')
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    return plot_to_image(fig, dpi)
    
def convert_sections_to_images(point_cloud, section_height, slice_thickness, tree_center, bottom_height, output_dir, base_filename):
    max_height =  np.max(point_cloud[:, 2])-bottom_height
    num_sections = int(np.ceil(max_height / section_height))

    # Normalize point cloud
    point_cloud[:, 2]=point_cloud[:, 2]-bottom_height

    for i in range(num_sections):
        # compute the lower end of the interval
        z_low=np.float64(0)
        if i>0:
            z_low = np.float64(i)*section_height

        # get upper end of the interval
        z_high = (i +1) * section_height

        # slice pointcloud
        section_mask = (point_cloud[:, 2] >= z_low) & (point_cloud[:, 2] < z_high)
        section_points = point_cloud[section_mask]

        # skip if there are no points
        if not section_points.size:
            continue

        # if the tree does not fill the frame then modify the lower and upper limits
        if (np.max(section_points[:, 2])-np.min(section_points[:, 2]))<(section_height-0.5):
            z_high = np.max(section_points[:, 2])
            z_low= z_high-section_height
            # slice pointcloud
            section_mask = (point_cloud[:, 2] >= z_low) & (point_cloud[:, 2] < z_high)
            section_points = point_cloud[section_mask]

        
        x_section_slice, y_section_slice = slice_tree_center_thick_slices(section_points, slice_thickness)
        rotated_section_points = rotate_point_cloud(section_points, 45, tree_center)
        x45_section_slice, y45_section_slice = slice_tree_center_thick_slices(rotated_section_points, slice_thickness)

        plot_limits = [tree_center[0] - slice_thickness/2, tree_center[0] + slice_thickness/2,
                       tree_center[1] - slice_thickness/2, tree_center[1] + slice_thickness/2]

        for slice_data, slice_name in zip([x_section_slice, y_section_slice, x45_section_slice, y45_section_slice],
                                          ['x', 'y', 'x45', 'y45']):
            img_array = plot_section_as_image_with_alpha(slice_data, z_low, z_high, dpi=100)
            
            # Metadata for filename
            min_x_or_y = np.min(slice_data[:, 0 if slice_name in ['x', 'x45'] else 1])
            max_x_or_y = np.max(slice_data[:, 0 if slice_name in ['x', 'x45'] else 1])
            filename = f"{output_dir}/{base_filename}_{slice_name}_section_{i}_min{min_x_or_y:.2f}_max{max_x_or_y:.2f}_zmin{z_low:.2f}_zmax{z_high:.2f}.png"
            
            Image.fromarray(img_array).save(filename)


def process_point_cloud(point_cloud, output_directory, bottom_height, base_filename):
    tree_center = point_cloud[point_cloud[:,2].argmax(), :2]
    convert_sections_to_images(point_cloud, 10, 10, tree_center,bottom_height, output_directory, base_filename)

# Function to get image size
def get_image_size(image_path):
    with Image.open(image_path) as img:
        return img.size  # (width, height)

# Function to convert normalized coordinates to real-world coordinates
def convert_to_real_world(px, py, img_width, img_height, x_min, x_max, z_min, z_max):
    real_x = px * img_width
    real_y = py * img_height
    world_x = real_x / img_width * (x_max - x_min) + x_min
    world_z = real_y / img_height * (z_max - z_min) + z_min
    return world_x, world_z

# Function to process a single file
def process_file(text_file_path, img_width, img_height, x_min, x_max, z_min, z_max):
    real_world_data = []
    with open(text_file_path, 'r') as file:
        for line in file.readlines():
            parts = line.strip().split()

            # get coordinates and confidence for the three keypoints where
            # p1: left branch; p2: whorl center; p3: right branch
            px1 = float(parts[5])
            py1 = float(parts[6])
            confidence_p1 = float(parts[7])
            px2 = float(parts[8])
            py2 = float(parts[9])
            confidence_p2 = float(parts[10])
            px3 = float(parts[11])
            py3 = float(parts[12])
            confidence_p3 = float(parts[13])


            world_px1, world_pz1 = convert_to_real_world(px1,py1, img_width, img_height, x_min, x_max, z_min, z_max)
            world_px2, world_pz2 = convert_to_real_world(px2, py2, img_width, img_height, x_min, x_max, z_min, z_max)
            world_px3, world_pz3 = convert_to_real_world(px3, py3, img_width, img_height, x_min, x_max, z_min, z_max)

            real_world_data.append((confidence_p1, world_px1, world_pz2, confidence_p2, world_px2, world_pz2,confidence_p3, world_px3, world_pz3))
    return real_world_data

# Function to calculate the angle at p2 formed by p1 and p3
def calculate_angle_at_p2(px1, pz1, px2, pz2, px3, pz3):
    # Construct vectors from p2 to p1 and p2 to p3
    vector_p2_p1 = np.array([px1 - px2, pz1 - pz2])
    vector_p2_p3 = np.array([px3 - px2, pz3 - pz2])
    # Calculate the dot product and norms of the vectors
    dot_product = np.dot(vector_p2_p1, vector_p2_p3)
    norm_p2_p1 = np.linalg.norm(vector_p2_p1)
    norm_p2_p3 = np.linalg.norm(vector_p2_p3)
    # Calculate the angle in radians and then convert to degrees
    angle = np.arccos(np.clip(dot_product / (norm_p2_p1 * norm_p2_p3), -1.0, 1.0))
    angle_degrees = np.degrees(angle)
    # Check for reflex angle (greater than 180 degrees)
    if np.cross(vector_p2_p1, vector_p2_p3) < 0:  # using cross product to determine the orientation
        angle_degrees = 360 - angle_degrees
    return angle_degrees

# Function to calculate Euclidean distance
def calculate_distance(px1, pz1, px2, pz2):
    return np.sqrt((px1 - px2) ** 2 + (pz1 - pz2) ** 2)

# function to process each tree and obtain the detected whorls
def pose_detection_tree(treeID, trees,non_tree, dir_root, my_model, dir_pred, min_dist_whorls=0.24):
    # skip if treeID ==255
    if treeID == 0:
        return None  # Skip if treeID is 0

    # Folder path containing  the text files
    dir_temp_imgs=dir_pred+"/orig_imgs"
    if not os.path.exists(dir_temp_imgs):
       os.makedirs(dir_temp_imgs)
    dir_orig_imgs_tree=dir_temp_imgs+"/"+str(round(treeID))
    if not os.path.exists(dir_orig_imgs_tree):
       os.makedirs(dir_orig_imgs_tree)
        
    dir_pred_out=dir_pred+"/preds"
    if not os.path.exists(dir_pred_out):
       os.makedirs(dir_pred_out)
        
    dir_labels=dir_pred_out+"/"+str(round(treeID))+"/labels"

    # select the one tree
    one_tree_np= trees[trees[:,3]==treeID]
    
    # compute tree top (will be used later)
    top_point= one_tree_np[one_tree_np[:,2]==np.max(one_tree_np[:,2])] 
    top_height=top_point[0,2]
    
    # compute tree bottom (will be used later)
    bottom_point= one_tree_np[one_tree_np[:,2]==np.min(one_tree_np[:,2])] 
    bottom_height=bottom_point[0,2]
    
    if len(non_tree)<0:
        # Separate the x, y, and z coordinates
        xy_coordinates = non_tree[:, :2]  # Extract x and y
        #print(xy_coordinates)
        #print(type(xy_coordinates))
        z_values = non_tree[:, 2]  # Extract z
    
        # Create a KDTree for efficient spatial search
        KDtree = cKDTree(xy_coordinates)
        
        # Given point coordinates (x, y)
        x, y = bottom_point[0,0], bottom_point[0,1]
        #print(type(bottom_point[0,0]))
        #print([x, y])
        #print(KDtree)
        # Find the nearest point
        _, index = KDtree.query([x, y], k=1)
        
        # Extract the z value of the nearest point
        bottom_height = z_values[index]
    

    ## Create images from point clouds
    process_point_cloud(one_tree_np, dir_orig_imgs_tree,bottom_height, 'img_')



    ####################################################################################################################################
    ## Predict on new data
    model=YOLO(my_model)
    model.predict(source=dir_orig_imgs_tree,conf=0.3, imgsz=1000, save=True, save_txt=True, project=dir_pred_out, name=str(round(treeID)))  # no arguments needed, dataset and settings remembered


    
    ####################################################################################################################################
    ## Parse output to produce data for the whole tree
    
    # List all text files in the folder
    text_files = [f for f in os.listdir(dir_labels) if f.endswith('.txt')]
    #print(dir_labels)
    # Process each text file
    all_data = []
    for text_file in text_files:
        # Extract the base filename to find the corresponding image file
        base_filename = text_file.replace('.txt', '')
        image_filename = base_filename + '.png'  # Assuming image extension is .png
        image_path = os.path.join(dir_orig_imgs_tree, image_filename)
        text_file_path = os.path.join(dir_labels, text_file)
        
        # Get image size
        img_width, img_height = get_image_size(image_path)
        
        # Extract metadata from the file name
        pattern = r"min(-?\d+\.\d+)_max(-?\d+\.\d+)_zmin(-?\d+\.\d+)_zmax(-?\d+\.\d+)"
    
        match = re.search(pattern, text_file)
        if match:
            x_min, x_max, z_min, z_max = map(float, match.groups())
            #x_min, x_max, z_min, z_max = map(float, re.findall(r"min(-?\d+\.\d+)_max(-?\d+\.\d+)_zmin(-?\d+\.\d+)_zmax(-?\d+\.\d+)", text_file)[0])
            #print(x_min)

        else:
            x_min, x_max, z_min, z_max = map(float, re.findall(r"min(\d+\.\d+)_max(\d+\.\d+)_zmin(\d+\.\d+)_zmax(\d+\.\d+)", text_file)[0])
        
        # Process the file
        file_data = process_file(text_file_path, img_width, img_height, x_min, x_max, z_min, z_max)
        
        # Add filename string to each row and extend the all_data list
        slice_direction = base_filename.split('__')[1].split('_section')[0]
        all_data.extend([(confidence_p1, world_px1, world_pz2, confidence_p2, world_px2, world_pz2,confidence_p3, world_px3, world_pz3, slice_direction) for confidence_p1, world_px1, world_pz2, confidence_p2, world_px2, world_pz2,confidence_p3, world_px3, world_pz3 in file_data])
    
    # Create a DataFrame with all data
    df_all = pd.DataFrame(all_data, columns=['confidence_p1', 'world_px1', 'world_pz1', 'confidence_p2', 'world_px2', 'world_pz2','confidence_p3', 'world_px3', 'world_pz3','slice_direction'])
    df_all['treeID']=treeID
    df_all_sorted = df_all.sort_values(by='world_pz2')
    
    # Applying the functions to the DataFrame
    df_all_sorted['branch_opening_angle'] = df_all_sorted.apply(lambda row: calculate_angle_at_p2(row['world_px1'], row['world_pz1'],row['world_px2'], row['world_pz2'],row['world_px3'], row['world_pz3']), axis=1)
    #df_all_sorted['branch_opening_angle']=180-df_all_sorted['branch_opening_angle']
    df_all_sorted['branch_length_p1_p2'] = df_all_sorted.apply(lambda row: calculate_distance(row['world_px1'], row['world_pz1'], row['world_px3'], row['world_pz3']), axis=1)
    df_all_sorted['branch_length_p3_p2'] = df_all_sorted.apply(lambda row: calculate_distance(row['world_px3'], row['world_pz3'], row['world_px2'], row['world_pz2']), axis=1)
    #df_all_sorted['branch_length_p3_p1'] = df_all_sorted.apply(lambda row: calculate_distance(row['world_px3'], row['world_pz3'], row['world_px1'], row['world_pz1']), axis=1)

    # add tree top and tree bottom points to the sorted dataframe
    df_all_sorted.loc[len(df_all_sorted)] = [0, 0,0,1,top_point[0,0], top_height,0,0,0,0,0,0,0,0 ]
    #df_all_sorted.loc[len(df_all_sorted)] = [0, 0,0,1,bottom_point[0,0], bottom_height,0,0,0,0,0,0,0,0 ]
    
    # re-sort the dataframe
    df_all_sorted = df_all_sorted.sort_values(by='world_pz2')
    
    
    
    ####################################################################################################################################
    # Cleanup
    # subset to select only most confident predictions within each "min_dist_whorls" cm interval
    # Now, let's iterate through each row and select the one with the largest probability
    # if consecutive rows are closer than 0.05 in Z values.
    selected_rows = []
    current_row = df_all_sorted.iloc[0]
    
    for index, next_row in df_all_sorted.iterrows():
        if (next_row['world_pz2'] - current_row['world_pz2']) < min_dist_whorls:
            # If the Z values are closer than 0.05, check the confidence
            if next_row['confidence_p2'] > current_row['confidence_p2']:
                current_row = next_row
        else:
            # If they are not closer, add the current row to the selected rows
            selected_rows.append(current_row)
            current_row = next_row
    
    # Make sure to add the last row after the loop
    selected_rows.append(current_row)
    
    # Create a DataFrame with the selected rows
    df_selected = pd.DataFrame(selected_rows)
    
    #df_selected
    
    # Calculating the maximum branch length
    df_selected['max_branch_length'] = df_selected[['branch_length_p1_p2', 'branch_length_p3_p2']].max(axis=1)
    
    # Calculating the average branch length
    df_selected['average_branch_length'] = df_selected[['branch_length_p1_p2', 'branch_length_p3_p2']].mean(axis=1)
    #df_selected['crown_diam'] = df_selected[['branch_length_p3_p1', 'branch_length_p3_p2']].max(axis=1)

    
    # Now, replace values greater than 10 with 0 in both columns
    df_selected['max_branch_length'] = df_selected['max_branch_length'].apply(lambda x: 0 if x > 10 else x)
    df_selected['average_branch_length'] = df_selected['average_branch_length'].apply(lambda x: 0 if x > 10 else x)
    #df_selected['crown_diam'] = df_selected['crown_diam'].apply(lambda x: 0 if x > 10 else x)

    
    ####################################################################################################################################
    ## Create pointcloud result
    whorls_pc= df_selected[['world_px2','world_pz2','confidence_p2','branch_opening_angle','max_branch_length','average_branch_length']]
    whorls_pc['x']=top_point[0,0]
    whorls_pc['y']=top_point[0,1]
    
    # de-normalize z
    whorls_pc['z']=whorls_pc['world_pz2']+bottom_height
    # add tree ID
    #ID = re.findall(r'\d+', os.path.splitext(os.path.basename(treeID))[0])
    ID = int(treeID)
    whorls_pc['treeID']=ID    # This should include any processing and return the necessary results

    # For demonstration, returning a simple dictionary. This should be replaced with actual processing results
    return {'treeID': treeID, 'result': df_selected, 'whorl_pc':whorls_pc}  # Replace with actual result
