# 🌲 Whorl pose detector 📈
Repo containing the code to run pose estimation to detect whorls in individual tree point clouds

This is a continuation of the work done by [Puliti et al.(2023)](https://academic.oup.com/forestry/article/96/1/37/6628789) on whorl detection for coniferous trees. In this second version we replaced the bounding box detector model with a pose estimation model aimed at detecting the geometry of each whorl (see image below), with branches on the left side of the tree (🔵), the whorl in the center in correspondence of the tree trunk (🟠), and branches on the right side of the tree (🟢). 

![whorl_pose](https://github.com/user-attachments/assets/05fb09f3-4a65-4676-81d1-43bc2f6f90d9)

## 🛠️ Setting it up
To install, first clone the repo
```
git clone https://github.com/stefp/whorl_pose_detector
```

Create and activate a new conda environment
```
conda create -n whorl_pose_detector python
conda activate whorl_pose_detector
```

Then install the required libraries
```
cd whorl_pose_detector
pip install -r requirements.txt
```


## 🚀 How to run it for prediction on new data
In the 'data' folder You can find a demo dataset showing how the data should look like. The *.las or *.laz file should include a treeID and semantic column. These instance and semantic labels can be obtained for example using [ForAINet](https://github.com/bxiang233/ForAINet) or [SegmentAnyTree](https://github.com/SmartForest-no/SegmentAnyTree). 

For predicting on your own data just point the --dir_root to the path to the folder where the point cloud data is stored. 

### 🖥️ To run using command line (CLI): 
```
python whorl_pose_detect_CLI.py --dir_root data --my_model whorl_pose_nano_1000px/weights/best.pt --alpha 0.5 --min_internodal_d 0.3 --tree_id_label treeID --semantic_label semantic  
```

### 🎮 Demo version
The 'demo_predict_whorl_pose' notebook aims to provide a more detailed understanding of the different steps of the method, including some nice plots 

