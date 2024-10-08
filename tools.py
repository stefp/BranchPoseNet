# Useful Functions for Whorl Pose Estimation

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
from ultralytics import YOLO

####################################################################################################################################
# Define custom functions

def rotate_point_cloud(point_cloud, angle_degrees, center_point):
    """
    Rotate the point cloud around a center point by a given angle in degrees.

    Parameters:
    - point_cloud: numpy array of shape (n, 3) representing the point cloud (x, y, z coordinates).
    - angle_degrees: float, the angle by which to rotate the point cloud (in degrees).
    - center_point: numpy array of shape (2,) representing the center point (x, y) around which to rotate.

    Returns:
    - numpy array of shape (n, 3) representing the rotated point cloud.
    """
    theta = np.radians(angle_degrees)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    rotated_points = np.dot(point_cloud[:, :2] - center_point, R.T) + center_point
    return np.hstack((rotated_points, point_cloud[:, 2].reshape(-1, 1)))

def slice_tree_center_thick_slices(point_cloud, slice_thickness=10):
    """
    Take thick slices in the X and Y directions, centered around the tree's center.

    Parameters:
    - point_cloud: numpy array of shape (n, 3) representing the point cloud (x, y, z coordinates).
    - slice_thickness: float, the thickness of the slices to be taken (in meters).

    Returns:
    - tuple of two numpy arrays, representing the X and Y direction slices respectively.
    """
    tree_center = point_cloud[point_cloud[:, 2].argmax(), :2]
    x_slice_mask = (point_cloud[:, 0] >= tree_center[0] - slice_thickness/2) & \
                   (point_cloud[:, 0] <= tree_center[0] + slice_thickness/2)
    y_slice_mask = (point_cloud[:, 1] >= tree_center[1] - slice_thickness/2) & \
                   (point_cloud[:, 1] <= tree_center[1] + slice_thickness/2)
    x_slice = point_cloud[x_slice_mask]
    y_slice = point_cloud[y_slice_mask]
    return x_slice, y_slice

def plot_to_image(figure, dpi):
    """
    Converts a Matplotlib plot specified by 'figure' to a PNG image and returns it as a numpy array.

    Parameters:
    - figure: Matplotlib figure object.
    - dpi: int, the resolution in dots per inch.

    Returns:
    - numpy array representing the image.
    """
    buf = io.BytesIO()
    figure.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=dpi)
    plt.close(figure)
    buf.seek(0)
    image = Image.open(buf)
    return np.array(image)

def plot_section_as_image_with_alpha(slice_data, z_low, z_high, alpha=0.3, output_size=(1000, 1000), dpi=100):
    """
    Plot the slice data with alpha transparency and return it as an image array.

    Parameters:
    - slice_data: numpy array of shape (n, 3) representing the slice data (x, y, z coordinates).
    - z_low: float, the lower bound of the Z coordinate.
    - z_high: float, the upper bound of the Z coordinate.
    - alpha: float, the alpha transparency value for the points.
    - output_size: tuple of ints, the size of the output image (width, height).
    - dpi: int, the resolution in dots per inch.

    Returns:
    - numpy array representing the image, or None if no data points are provided.
    """
    if slice_data.size == 0:
        return None

    buffer = 0
    x_min, x_max = np.min(slice_data[:, 0]) - buffer, np.max(slice_data[:, 0]) + buffer
    y_min, y_max = np.min(slice_data[:, 2]) - buffer, np.max(slice_data[:, 2]) + buffer

    x_range = x_max - x_min
    y_range = z_high - z_low

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
    ax.set_aspect('auto')

    ax.axis('off')
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    return plot_to_image(fig, dpi)

def convert_sections_to_images(point_cloud, section_height, slice_thickness, tree_center, bottom_height, output_dir, base_filename):
    """
    Convert sections of the point cloud into images and save them.

    Parameters:
    - point_cloud: numpy array of shape (n, 3) representing the point cloud (x, y, z coordinates).
    - section_height: float, the height of each section to be sliced.
    - slice_thickness: float, the thickness of the slices.
    - tree_center: numpy array of shape (2,), the center point (x, y) of the tree.
    - bottom_height: float, the height of the tree bottom.
    - output_dir: str, the directory where images will be saved.
    - base_filename: str, the base name for saved images.
    """
    max_height = np.max(point_cloud[:, 2]) - bottom_height
    num_sections = int(np.ceil(max_height / section_height))

    point_cloud[:, 2] = point_cloud[:, 2] - bottom_height

    for i in range(num_sections):
        z_low = np.float64(i) * section_height if i > 0 else np.float64(0)
        z_high = (i + 1) * section_height

        section_mask = (point_cloud[:, 2] >= z_low) & (point_cloud[:, 2] < z_high)
        section_points = point_cloud[section_mask]

        if not section_points.size:
            continue

        if (np.max(section_points[:, 2]) - np.min(section_points[:, 2])) < (section_height - 0.5):
            z_high = np.max(section_points[:, 2])
            z_low = z_high - section_height
            section_mask = (point_cloud[:, 2] >= z_low) & (point_cloud[:, 2] < z_high)
            section_points = point_cloud[section_mask]

        x_section_slice, y_section_slice = slice_tree_center_thick_slices(section_points, slice_thickness)
        rotated_section_points = rotate_point_cloud(section_points, 45, tree_center)
        x45_section_slice, y45_section_slice = slice_tree_center_thick_slices(rotated_section_points, slice_thickness)

        for slice_data, slice_name in zip([x_section_slice, y_section_slice, x45_section_slice, y45_section_slice],
                                          ['x', 'y', 'x45', 'y45']):
            img_array = plot_section_as_image_with_alpha(slice_data, z_low, z_high, dpi=100)
            if img_array is not None:
                min_x_or_y = np.min(slice_data[:, 0 if slice_name in ['x', 'x45'] else 1])
                max_x_or_y = np.max(slice_data[:, 0 if slice_name in ['x', 'x45'] else 1])
                filename = f"{output_dir}/{base_filename}_{slice_name}_section_{i}_min{min_x_or_y:.2f}_max{max_x_or_y:.2f}_zmin{z_low:.2f}_zmax{z_high:.2f}.png"
                Image.fromarray(img_array).save(filename)

def process_point_cloud(point_cloud, output_directory, bottom_height, base_filename):
    """
    Process the point cloud to generate and save sectional images.

    Parameters:
    - point_cloud: numpy array of shape (n, 3) representing the point cloud (x, y, z coordinates).
    - output_directory: str, the directory where images will be saved.
    - bottom_height: float, the height of the tree bottom.
    - base_filename: str, the base name for saved images.
    """
    tree_center = point_cloud[point_cloud[:, 2].argmax(), :2]
    convert_sections_to_images(point_cloud, 10, 10, tree_center, bottom_height, output_directory, base_filename)

def get_image_size(image_path):
    """
    Get the dimensions of an image.

    Parameters:
    - image_path: str, the path to the image file.

    Returns:
    - tuple of ints representing the image size (width, height).
    """
    with Image.open(image_path) as img:
        return img.size

def convert_to_real_world(px, py, img_width, img_height, x_min, x_max, z_min, z_max):
    """
    Convert normalized image coordinates to real-world coordinates.

    Parameters:
    - px, py: floats, the normalized coordinates in the image.
    - img_width, img_height: ints, the width and height of the image.
    - x_min, x_max, z_min, z_max: floats, the real-world bounds for the X and Z coordinates.

    Returns:
    - tuple of floats representing the real-world coordinates (world_x, world_z).
    """
    real_x = px * img_width
    real_y = py * img_height
    world_x = real_x / img_width * (x_max - x_min) + x_min
    world_z = real_y / img_height * (z_max - z_min) + z_min
    return world_x, world_z

def process_file(text_file_path, img_width, img_height, x_min, x_max, z_min, z_max):
    """
    Process a text file containing point data and convert coordinates to real-world values.

    Parameters:
    - text_file_path: str, path to the text file containing the point data.
    - img_width, img_height: ints, the dimensions of the corresponding image.
    - x_min, x_max, z_min, z_max: floats, the real-world bounds for the X and Z coordinates.

    Returns:
    - list of tuples containing the converted real-world coordinates and confidence values.
    """
    real_world_data = []
    with open(text_file_path, 'r') as file:
        for line in file.readlines():
            parts = line.strip().split()

            px1, py1, confidence_p1 = float(parts[5]), float(parts[6]), float(parts[7])
            px2, py2, confidence_p2 = float(parts[8]), float(parts[9]), float(parts[10])
            px3, py3, confidence_p3 = float(parts[11]), float(parts[12]), float(parts[13])

            world_px1, world_pz1 = convert_to_real_world(px1, py1, img_width, img_height, x_min, x_max, z_min, z_max)
            world_px2, world_pz2 = convert_to_real_world(px2, py2, img_width, img_height, x_min, x_max, z_min, z_max)
            world_px3, world_pz3 = convert_to_real_world(px3, py3, img_width, img_height, x_min, x_max, z_min, z_max)

            real_world_data.append((confidence_p1, world_px1, world_pz2, confidence_p2, world_px2, world_pz2, confidence_p3, world_px3, world_pz3))
    return real_world_data

def calculate_angle_at_p2(px1, pz1, px2, pz2, px3, pz3):
    """
    Calculate the angle at point p2 formed by the vectors from p2 to p1 and from p2 to p3.

    Parameters:
    - px1, pz1: floats, coordinates of point p1.
    - px2, pz2: floats, coordinates of point p2 (the vertex of the angle).
    - px3, pz3: floats, coordinates of point p3.

    Returns:
    - float representing the angle at p2 in degrees.
    """
    vector_p2_p1 = np.array([px1 - px2, pz1 - pz2])
    vector_p2_p3 = np.array([px3 - px2, pz3 - pz2])

    dot_product = np.dot(vector_p2_p1, vector_p2_p3)
    norm_p2_p1 = np.linalg.norm(vector_p2_p1)
    norm_p2_p3 = np.linalg.norm(vector_p2_p3)

    angle = np.arccos(np.clip(dot_product / (norm_p2_p1 * norm_p2_p3), -1.0, 1.0))
    angle_degrees = np.degrees(angle)

    if np.cross(vector_p2_p1, vector_p2_p3) < 0:
        angle_degrees = 360 - angle_degrees

    return angle_degrees

def calculate_distance(px1, pz1, px2, pz2):
    """
    Calculate the Euclidean distance between two points.

    Parameters:
    - px1, pz1: floats, coordinates of the first point.
    - px2, pz2: floats, coordinates of the second point.

    Returns:
    - float representing the distance between the two points.
    """
    return np.sqrt((px1 - px2) ** 2 + (pz1 - pz2) ** 2)


def pose_detection_tree(treeID, trees,non_tree, dir_root, my_model, dir_pred, min_dist_whorls=0.24):
    """
    Detect the whorls of a single tree using the YOLOv8 model and post-process the results.

    Parameters:
    - treeID: int, the unique identifier of the tree.
    - trees: numpy array, the point cloud data of all trees.
    - non_tree: numpy array, the point cloud data of non-tree objects.
    - dir_root: str, the root directory for saving results.
    - my_model: str, the path to the YOLO model.
    - dir_pred: str, the directory for saving intermediate predictions.
    - min_dist_whorls: float, the minimum distance between whorls to remove duplicates (in meters).

    Returns:
    - dict containing the processed tree results and point cloud data of detected whorls.
    """
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
