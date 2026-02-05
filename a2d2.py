"""Working with the configuration file
The configuration JSON file plays an important role in processing the A2D2 dataset."""


import json
import pprint

with open ('cams_lidars.json', 'r') as f:
    config = json.load(f)


pprint.pprint(config)


config.keys()
"""Each sensor in 'lidars', 'cameras', and 'vehicle' has an associated 'view'. A view is a sensor coordinate system, defined by an origin, an x-axis, and a y-axis. These are specified in cartesian coordinates (in m) relative to an external coordinate system. Unless otherwise stated the external coordinate system is the car's frame of reference.

'vehicle' contains a 'view' object specifying the frame of reference of the car. It also contains an 'ego-dimensions' object, which specifies the extension of the vehicle in the frame of reference of the car.

The 'lidars' object contains objects specifying the extrinsic calibration parameters for each LiDAR sensor. Our car has five LiDAR sensors: 'front_left', 'front_center', 'front_right', 'rear_right', and 'rear_left'. Each LiDAR has a 'view' defining its pose in the frame of reference of the car.

The 'cameras' object contains camera objects which specify their calibration parameters. The car has six cameras: 'front_left', 'front_center', 'front_right', 'side_right', 'rear_center' and 'side_left'. Each camera object contains:

'view'- pose of the camera relative to the external coordinate system (frame of reference of the car)
'Lens'- type of lens used. It can take two values: 'Fisheye' or 'Telecam'
'CamMatrix' - the intrinsic camera matrix of undistorted camera images
'CamMatrixOriginal' - the intrinsic camera matrix of original (distorted) camera images
'Distortion' - distortion parameters of original (distorted) camera images
'Resolution' - resolution (columns, rows) of camera images (same for original and undistorted images)
'tstamp_delay'- specifies a known delay in microseconds between actual camera frame times (default: 0)"""

#Display the contents of 'vehicle':

config['vehicle'].keys()
#Likewise for LiDAR sensors:

config['lidars'].keys()
#Here we see the names of the LiDAR sensors mounted on the car. For example, the configuration parameters for the front_left LiDAR sensor can be accessed using

config['lidars']['front_left']
#The camera sensors mounted on the car can be obtained using

config['cameras'].keys()
#Configuration parameters for a particular camera can be accessed using e.g.

config['cameras']['front_left']
"""
Working with view objects
We have seen that the vehicle and each sensor in the configuration file have a 'view' object. A view specifies the pose of a sensor relative to an external coordinate system, here the frame of reference of the car. In the following we use the term 'global' interchangeably with 'frame of reference of the car'.

A view associated with a sensor can be accessed as follows:
"""
view = config['cameras']['front_left']['view']
import numpy as np
import numpy.linalg as la
Define a small constant to avoid errors due to small vectors.

EPSILON = 1.0e-10 # norm should not be small
The following functions get the axes and origin of a view.

def get_axes_of_a_view(view):
    x_axis = view['x-axis']
    y_axis = view['y-axis']
     
    x_axis_norm = la.norm(x_axis)
    y_axis_norm = la.norm(y_axis)
    
    if (x_axis_norm < EPSILON or y_axis_norm < EPSILON):
        raise ValueError("Norm of input vector(s) too small.")
        
    # normalize the axes
    x_axis = x_axis / x_axis_norm
    y_axis = y_axis / y_axis_norm
    
    # make a new y-axis which lies in the original x-y plane, but is orthogonal to x-axis
    y_axis = y_axis - x_axis * np.dot(y_axis, x_axis)
 
    # create orthogonal z-axis
    z_axis = np.cross(x_axis, y_axis)
    
    # calculate and check y-axis and z-axis norms
    y_axis_norm = la.norm(y_axis)
    z_axis_norm = la.norm(z_axis)
    
    if (y_axis_norm < EPSILON) or (z_axis_norm < EPSILON):
        raise ValueError("Norm of view axis vector(s) too small.")
        
    # make x/y/z-axes orthonormal
    y_axis = y_axis / y_axis_norm
    z_axis = z_axis / z_axis_norm
    
    return x_axis, y_axis, z_axis
def get_origin_of_a_view(view):
    return view['origin']
#A homogeneous transformation matrix from view point to global coordinates (inverse "extrinsic" matrix) can be obtained as follows. Note that this matrix contains the axes and the origin in its columns.

def get_transform_to_global(view):
    # get axes
    x_axis, y_axis, z_axis = get_axes_of_a_view(view)
    
    # get origin 
    origin = get_origin_of_a_view(view)
    transform_to_global = np.eye(4)
    
    # rotation
    transform_to_global[0:3, 0] = x_axis
    transform_to_global[0:3, 1] = y_axis
    transform_to_global[0:3, 2] = z_axis
    
    # origin
    transform_to_global[0:3, 3] = origin
    
    return transform_to_global
#For the view defined above

transform_to_global = get_transform_to_global(view)
print (transform_to_global)
[[ 9.96714314e-01 -8.09890350e-02  1.16333982e-03  1.71104606e+00]
 [ 8.09967396e-02  9.96661051e-01 -1.03090934e-02  5.80000039e-01]
 [-3.24531964e-04  1.03694477e-02  9.99946183e-01  9.43144935e-01]
 [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00]]
#Homogeneous transformation matrix from global coordinates to view point coordinates ("extrinsic" matrix)

def get_transform_from_global(view):
   # get transform to global
   transform_to_global = get_transform_to_global(view)
   trans = np.eye(4)
   rot = np.transpose(transform_to_global[0:3, 0:3])
   trans[0:3, 0:3] = rot
   trans[0:3, 3] = np.dot(rot, -transform_to_global[0:3, 3])
    
   return trans
#For the view defined above

transform_from_global = get_transform_from_global(view)
print(transform_from_global)
"""
[[ 9.96714314e-01  8.09967396e-02 -3.24531964e-04 -1.75209613e+00]
 [-8.09890350e-02  9.96661051e-01  1.03694477e-02 -4.49267371e-01]
 [ 1.16333982e-03 -1.03090934e-02  9.99946183e-01 -9.39105431e-01]
 [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00]]
The transform_to_global and transform_from_global matrices should be the inverse of one another. Check that muliplying them results in an identity matrix (subject to numerical precision):
"""
print(np.matmul(transform_from_global, transform_to_global))
"""[[ 1.00000000e+00 -4.05809291e-18 -7.51833703e-21  0.00000000e+00]
 [-4.05809291e-18  1.00000000e+00 -5.22683251e-19  5.55111512e-17]
 [-7.51833703e-21 -5.22683251e-19  1.00000000e+00  0.00000000e+00]
 [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00]]
The global-to-view rotation matrix can be obtained using
"""
def get_rot_from_global(view):
    # get transform to global
    transform_to_global = get_transform_to_global(view)
    # get rotation
    rot =  np.transpose(transform_to_global[0:3, 0:3])
    
    return rot
#For the view defined above

rot_from_global = get_rot_from_global(view)
print(rot_from_global)
"""[[ 9.96714314e-01  8.09967396e-02 -3.24531964e-04]
 [-8.09890350e-02  9.96661051e-01  1.03694477e-02]
 [ 1.16333982e-03 -1.03090934e-02  9.99946183e-01]]
The rotation matrix from this view point to the global coordinate system can be obtained using
"""
def get_rot_to_global(view):
    # get transform to global
    transform_to_global = get_transform_to_global(view)
    # get rotation
    rot = transform_to_global[0:3, 0:3]
    
    return rot
#For the view defined above

rot_to_global = get_rot_to_global(view)
print(rot_to_global)
"""[[ 9.96714314e-01 -8.09890350e-02  1.16333982e-03]
 [ 8.09967396e-02  9.96661051e-01 -1.03090934e-02]
 [-3.24531964e-04  1.03694477e-02  9.99946183e-01]]
Let us see how we can calculate a rotation matrix from a source view to a target view
"""
def rot_from_to(src, target):
    rot = np.dot(get_rot_from_global(target), get_rot_to_global(src))
    
    return rot
#A rotation matrix from front left camera to front right camera

src_view = config['cameras']['front_left']['view']
target_view = config['cameras']['front_right']['view']
rot = rot_from_to(src_view, target_view)
print(rot)
"""[[ 0.99614958 -0.0876356  -0.00245312]
 [ 0.08757611  0.99598808 -0.01838914]
 [ 0.00405482  0.01810349  0.9998279 ]]
A rotation matrix in the opposite direction (front right camera -> front left camera)
"""
rot = rot_from_to(target_view, src_view)
print(rot)
"""[[ 0.99614958  0.08757611  0.00405482]
 [-0.0876356   0.99598808  0.01810349]
 [-0.00245312 -0.01838914  0.9998279 ]]
In the same manner, we can also calculate a transformation matrix from a source view to a target view. This will give us a 4x4 homogeneous transformation matrix describing the total transformation (rotation and shift) from the source view coordinate system into the target view coordinate system.
"""
def transform_from_to(src, target):
    transform = np.dot(get_transform_from_global(target), \
                       get_transform_to_global(src))
    
    return transform
#A transformation matrix from front left camera to front right camera

src_view = config['cameras']['front_left']['view']
target_view = config['cameras']['front_right']['view']
trans = transform_from_to(src_view, target_view)
print(trans)
"""[[ 0.99614958 -0.0876356  -0.00245312 -0.00769387]
 [ 0.08757611  0.99598808 -0.01838914  1.1599368 ]
 [ 0.00405482  0.01810349  0.9998279   0.00935439]
 [ 0.          0.          0.          1.        ]]
A transformation matrix in the opposite direction (front right camera -> front left camera)
"""
transt = transform_from_to(target_view, src_view)
print (transt)
"""[[ 0.99614958  0.08757611  0.00405482 -0.09395644]
 [-0.0876356   0.99598808  0.01810349 -1.15612684]
 [-0.00245312 -0.01838914  0.9998279   0.01195858]
 [ 0.          0.          0.          1.        ]]
Check if the product of the two opposite transformations results in a near identity matrix.
"""
print(np.matmul(trans, transt))
"""[[ 1.00000000e+00  9.38784819e-18 -2.59997757e-19 -4.15466272e-16]
 [ 9.38784819e-18  1.00000000e+00 -1.78819878e-18  0.00000000e+00]
 [-2.59997757e-19 -1.78819878e-18  1.00000000e+00 -5.20417043e-18]
 [ 0.00000000e+00  0.00000000e+00  0.00000000e+00  1.00000000e+00]]
We have seen that by using views we can transform coordinates from one sensor to another, or from a sensor to global global coordinates (and vice versa). In the following section, we read point clouds corresponding to all cameras. The point clouds are in camera view coordinates. In order to get a coherent view of the point clouds, we need to transform them into global coordinates.

Working with LiDAR data
First, read a LiDAR point cloud corresponding to the front center camera. The LiDAR data is saved in compressed numpy format, which can be read as follows:
"""
from os.path import join
import glob

root_path = './camera_lidar_semantic_bboxes/'
# get the list of files in lidar directory
file_names = sorted(glob.glob(join(root_path, '*/lidar/cam_front_center/*.npz')))

# select the lidar point cloud
file_name_lidar = file_names[5]

# read the lidar data
lidar_front_center = np.load(file_name_lidar)
#Let us explore the LiDAR data using the LiDAR points within the field of view of the front center camera. List keys:

print(list(lidar_front_center.keys()))
['azimuth', 'row', 'lidar_id', 'depth', 'reflectance', 'col', 'points', 'timestamp', 'distance']
#Get 3D point measurements

points = lidar_front_center['points']
#Get reflectance measurements

reflectance = lidar_front_center['reflectance']
#Get timestamps

timestamps = lidar_front_center['timestamp']
#Get coordinates of LiDAR points in image space

rows = lidar_front_center['row']
cols = lidar_front_center['col']
#Get distance and depth values

distance = lidar_front_center['distance']
depth = lidar_front_center['depth']
#Since the car is equipped with five LiDAR sensors, you can get the LiDAR sensor ID of each point using

lidar_ids = lidar_front_center['lidar_id']
#One way of visualizing point clouds is to use the Open3D library. The library supports beyond visualization other functionalities useful for point cloud processing. For more information on the library please refer to http://www.open3d.org/docs/release/.

import open3d as o3
#To visualize the LiDAR point clouds, we need to create an Open3D point cloud from the 3D points and reflectance values. The following function generates colors based on the reflectance values.

# Create array of RGB colour values from the given array of reflectance values
def colours_from_reflectances(reflectances):
    return np.stack([reflectances, reflectances, reflectances], axis=1)
    
#Now we can create Open3D point clouds for visualization

def create_open3d_pc(lidar, cam_image=None):
    # create open3d point cloud
    pcd = o3.geometry.PointCloud()
    
    # assign point coordinates
    pcd.points = o3.utility.Vector3dVector(lidar['points'])
    
    # assign colours
    if cam_image is None:
        median_reflectance = np.median(lidar['reflectance'])
        colours = colours_from_reflectances(lidar['reflectance']) / (median_reflectance * 5)
        
        # clip colours for visualisation on a white background
        colours = np.clip(colours, 0, 0.75)
    else:
        rows = (lidar['row'] + 0.5).astype(np.int)
        cols = (lidar['col'] + 0.5).astype(np.int)
        colours = cam_image[rows, cols, :] / 255.0
        
    pcd.colors = o3.utility.Vector3dVector(colours)
    
    return pcd
#Generate Open3D point cloud for the LiDAR data associated with the front center camera

pcd_front_center = create_open3d_pc(lidar_front_center)
#Visualize the point cloud

o3.visualization.draw_geometries([pcd_front_center])

#Let us transform LiDAR points from the camera view to the global view.

#First, read the view for the front center camera from the configuration file:

src_view_front_center = config['cameras']['front_center']['view']
#The vehicle view is the global view

vehicle_view = target_view = config['vehicle']['view']
#The following function maps LiDAR data from one view to another. Note the use of the function 'transform_from_to'. LiDAR data is provided in a camera reference frame.

def project_lidar_from_to(lidar, src_view, target_view):
    lidar = dict(lidar)
    trans = transform_from_to(src_view, target_view)
    points = lidar['points']
    points_hom = np.ones((points.shape[0], 4))
    points_hom[:, 0:3] = points
    points_trans = (np.dot(trans, points_hom.T)).T 
    lidar['points'] = points_trans[:,0:3]
    
    return lidar
#Now project the LiDAR points to the global frame (the vehicle frame of reference)

lidar_front_center = project_lidar_from_to(lidar_front_center,\
                                          src_view_front_center, \
                                          vehicle_view)
#Create open3d point cloud for visualizing the transformed points

pcd_front_center = create_open3d_pc(lidar_front_center)
#Visualise:

o3.visualization.draw_geometries([pcd_front_center])


#Working with images
#Import the necessary packages for reading, saving and showing images.

import cv2
%matplotlib inline
import matplotlib.pylab as pt
#Let us load the image corresponding to the above point cloud.

def extract_image_file_name_from_lidar_file_name(file_name_lidar):
    file_name_image = file_name_lidar.split('/')
    file_name_image = file_name_image[-1].split('.')[0]
    file_name_image = file_name_image.split('_')
    file_name_image = file_name_image[0] + '_' + \
                        'camera_' + \
                        file_name_image[2] + '_' + \
                        file_name_image[3] + '.png'

    return file_name_image
seq_name = file_name_lidar.split('/')[2]
file_name_image = extract_image_file_name_from_lidar_file_name(file_name_lidar)
file_name_image = join(root_path, seq_name, 'camera/cam_front_center/', file_name_image)
image_front_center = cv2.imread(file_name_image)
#Display image

image_front_center = cv2.cvtColor(image_front_center, cv2.COLOR_BGR2RGB)
pt.fig = pt.figure(figsize=(15, 15))

# display image from front center camera
pt.imshow(image_front_center)
pt.axis('off')
pt.title('front center')
#Text(0.5, 1.0, 'front center')

#In order to map point clouds onto images, or in order to color point clouds using colors drived from images, we need to perform distortion correction.

def undistort_image(image, cam_name):
    if cam_name in ['front_left', 'front_center', \
                    'front_right', 'side_left', \
                    'side_right', 'rear_center']:
        # get parameters from config file
        intr_mat_undist = \
                  np.asarray(config['cameras'][cam_name]['CamMatrix'])
        intr_mat_dist = \
                  np.asarray(config['cameras'][cam_name]['CamMatrixOriginal'])
        dist_parms = \
                  np.asarray(config['cameras'][cam_name]['Distortion'])
        lens = config['cameras'][cam_name]['Lens']
        
        if (lens == 'Fisheye'):
            return cv2.fisheye.undistortImage(image, intr_mat_dist,\
                                      D=dist_parms, Knew=intr_mat_undist)
        elif (lens == 'Telecam'):
            return cv2.undistort(image, intr_mat_dist, \
                      distCoeffs=dist_parms, newCameraMatrix=intr_mat_undist)
        else:
            return image
    else:
        return image
undist_image_front_center = undistort_image(image_front_center, 'front_center')
pt.fig = pt.figure(figsize=(15, 15))
pt.imshow(undist_image_front_center)
pt.axis('off')
pt.title('front center')
#Text(0.5, 1.0, 'front center')

#Each image has a timestamp and a LiDAR point cloud associated with it. The timestamp information is saved for each image in JSON format. Let us open the file for the front center camera.

file_name_image_info = file_name_image.replace(".png", ".json")

def read_image_info(file_name):
    with open(file_name, 'r') as f:
        image_info = json.load(f)
        
    return image_info

image_info_front_center = read_image_info(file_name_image_info)  
#Display the information for the front center camera

pprint.pprint(image_info_front_center)
{'cam_name': 'front_center',
 'cam_tstamp': 1537876333900931,
 'lidar_ids': {'0': 'front_left',
               '1': 'rear_right',
               '2': 'front_right',
               '3': 'front_center',
               '4': 'rear_left'}}

"""
We can see that the camera info contains the camera name, the time stamp in TAI (international atomic time) and a dictionary associating the LiDAR IDs with names of the LiDARs.

The LiDAR points are already mapped onto the undistorted images. The rows and columns of the corresponding pixels are saved in the lidar data.

Let us list the keys once again:
"""
lidar_front_center.keys()
dict_keys(['azimuth', 'row', 'lidar_id', 'depth', 'reflectance', 'col', 'points', 'timestamp', 'distance'])
Print the timestamp of each point in the LiDAR measurement.

pprint.pprint(lidar_front_center['timestamp'])
"""
array([1537876333860863, 1537876333860962, 1537876333860973, ...,
       1537876333847857, 1537876333847892, 1537876333847947])
Here we also see the timestamps of each measurement point in TAI. The camera is lagging behind the LiDAR points, i.e. the LiDAR measurements are taken before the corresponding image is captured. (timestamp_lidar-timestamp_camera)/(1000000) gives us the time difference between the measurement times of lidar data and the corresponding camera frame in seconds.
"""
def plot_lidar_id_vs_delat_t(image_info, lidar):
    timestamps_lidar = lidar['timestamp']
    timestamp_camera = image_info['cam_tstamp']
    time_diff_in_sec = (timestamps_lidar - timestamp_camera) / (1e6)
    lidar_ids = lidar ['lidar_id']
    pt.fig = pt.figure(figsize=(15, 5))
    pt.plot(time_diff_in_sec, lidar_ids, 'go', ms=2)
    pt.grid(True)
    ticks = np.arange(len(image_info['lidar_ids'].keys()))
    ticks_name = []
    for key in ticks:
        ticks_name.append(image_info['lidar_ids'][str(key)])
    pt.yticks(ticks, tuple(ticks_name))
    pt.ylabel('LiDAR sensor')
    pt.xlabel('delta t in sec')
    pt.title(image_info['cam_name'])
    pt.show()
#If we plot the lidar_ids versus the time difference for the front center camera we obtain

plot_lidar_id_vs_delat_t(image_info_front_center, lidar_front_center)

#Now we use col and row to map the LiDAR data onto images. The first function we use converts HSV to RGB. Please refere to the wikipedia article https://en.wikipedia.org/wiki/HSL_and_HSV for more information.

def hsv_to_rgb(h, s, v):
    if s == 0.0:
        return v, v, v
    
    i = int(h * 6.0)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i = i % 6
    
    if i == 0:
        return v, t, p
    if i == 1:
        return q, v, p
    if i == 2:
        return p, v, t
    if i == 3:
        return p, q, v
    if i == 4:
        return t, p, v
    if i == 5:
        return v, p, q
#The following function visualizes the mapping

def map_lidar_points_onto_image(image_orig, lidar, pixel_size=3, pixel_opacity=1):
    image = np.copy(image_orig)
    
    # get rows and cols
    rows = (lidar['row'] + 0.5).astype(np.int)
    cols = (lidar['col'] + 0.5).astype(np.int)
  
    # lowest distance values to be accounted for in colour code
    MIN_DISTANCE = np.min(lidar['distance'])
    # largest distance values to be accounted for in colour code
    MAX_DISTANCE = np.max(lidar['distance'])

    # get distances
    distances = lidar['distance']  
    # determine point colours from distance
    colours = (distances - MIN_DISTANCE) / (MAX_DISTANCE - MIN_DISTANCE)
    colours = np.asarray([np.asarray(hsv_to_rgb(0.75 * c, \
                        np.sqrt(pixel_opacity), 1.0)) for c in colours])
    pixel_rowoffs = np.indices([pixel_size, pixel_size])[0] - pixel_size // 2
    pixel_coloffs = np.indices([pixel_size, pixel_size])[1] - pixel_size // 2
    canvas_rows = image.shape[0]
    canvas_cols = image.shape[1]
    for i in range(len(rows)):
        pixel_rows = np.clip(rows[i] + pixel_rowoffs, 0, canvas_rows - 1)
        pixel_cols = np.clip(cols[i] + pixel_coloffs, 0, canvas_cols - 1)
        image[pixel_rows, pixel_cols, :] = \
                (1. - pixel_opacity) * \
                np.multiply(image[pixel_rows, pixel_cols, :], \
                colours[i]) + pixel_opacity * 255 * colours[i]
    return image.astype(np.uint8)
#Visualise the mapping of the LiDAR point clouds onto the front center image

image = map_lidar_points_onto_image(undist_image_front_center, lidar_front_center)
#Show the image

pt.fig = pt.figure(figsize=(20, 20))
pt.imshow(image)
pt.axis('off')
"""
(-0.5, 1919.5, 1207.5, -0.5)

The same result can be obtained by mapping the LiDAR point clouds using intrinsic camera parameters.

Let us finally open the semantic segmentation label corresponding to the above image
"""
def extract_semantic_file_name_from_image_file_name(file_name_image):
    file_name_semantic_label = file_name_image.split('/')
    file_name_semantic_label = file_name_semantic_label[-1].split('.')[0]
    file_name_semantic_label = file_name_semantic_label.split('_')
    file_name_semantic_label = file_name_semantic_label[0] + '_' + \
                  'label_' + \
                  file_name_semantic_label[2] + '_' + \
                  file_name_semantic_label[3] + '.png'
    
    return file_name_semantic_label
seq_name = file_name_lidar.split('/')[2]
file_name_semantic_label = extract_semantic_file_name_from_image_file_name(file_name_image)
file_name_semantic_label = join(root_path, seq_name, 'label/cam_front_center/', file_name_semantic_label)
semantic_image_front_center = cv2.imread(file_name_semantic_label)#
Display the semantic segmentation label

semantic_image_front_center = cv2.cvtColor(semantic_image_front_center, cv2.COLOR_BGR2RGB)
pt.fig = pt.figure(figsize=(15, 15))
pt.imshow(semantic_image_front_center)
pt.axis('off')
pt.title('label front center')
#Text(0.5, 1.0, 'label front center')

"""
We can use the semantic segmentation label to colour lidar points. This creates a 3D semantic label for a given frame.

First we need to undistort the semantic segmentation label.
"""
semantic_image_front_center_undistorted = undistort_image(semantic_image_front_center, 'front_center')
pt.fig = pt.figure(figsize=(15, 15))
pt.imshow(semantic_image_front_center_undistorted)
pt.axis('off')
pt.title('label front center')
Text(0.5, 1.0, 'label front center')

pcd_lidar_colored = create_open3d_pc(lidar_front_center, semantic_image_front_center_undistorted)
#Visualize the coloured lidar points

o3.visualization.draw_geometries([pcd_lidar_colored])
"""
Working with 3D bounding boxes
Before we can start working with 3D bounding boxes, we need some utility functions. The first utility function we need is the conversion from axis-angle representation into rotation matrices.
"""
def skew_sym_matrix(u):
    return np.array([[    0, -u[2],  u[1]], 
                     [ u[2],     0, -u[0]], 
                     [-u[1],  u[0],    0]])

def axis_angle_to_rotation_mat(axis, angle):
    return np.cos(angle) * np.eye(3) + \
        np.sin(angle) * skew_sym_matrix(axis) + \
        (1 - np.cos(angle)) * np.outer(axis, axis)
#Read the bounding boxes corresponding to the frame. We can read the bounding boxes as follows

import json
def read_bounding_boxes(file_name_bboxes):
    # open the file
    with open (file_name_bboxes, 'r') as f:
        bboxes = json.load(f)
        
    boxes = [] # a list for containing bounding boxes  
    print(bboxes.keys())
    
    for bbox in bboxes.keys():
        bbox_read = {} # a dictionary for a given bounding box
        bbox_read['class'] = bboxes[bbox]['class']
        bbox_read['truncation']= bboxes[bbox]['truncation']
        bbox_read['occlusion']= bboxes[bbox]['occlusion']
        bbox_read['alpha']= bboxes[bbox]['alpha']
        bbox_read['top'] = bboxes[bbox]['2d_bbox'][0]
        bbox_read['left'] = bboxes[bbox]['2d_bbox'][1]
        bbox_read['bottom'] = bboxes[bbox]['2d_bbox'][2]
        bbox_read['right']= bboxes[bbox]['2d_bbox'][3]
        bbox_read['center'] =  np.array(bboxes[bbox]['center'])
        bbox_read['size'] =  np.array(bboxes[bbox]['size'])
        angle = bboxes[bbox]['rot_angle']
        axis = np.array(bboxes[bbox]['axis'])
        bbox_read['rotation'] = axis_angle_to_rotation_mat(axis, angle) 
        boxes.append(bbox_read)

    return boxes 
#Let us read bounding boxes corresponding to the above image

def extract_bboxes_file_name_from_image_file_name(file_name_image):
    file_name_bboxes = file_name_image.split('/')
    file_name_bboxes = file_name_bboxes[-1].split('.')[0]
    file_name_bboxes = file_name_bboxes.split('_')
    file_name_bboxes = file_name_bboxes[0] + '_' + \
                  'label3D_' + \
                  file_name_bboxes[2] + '_' + \
                  file_name_bboxes[3] + '.json'
    
    return file_name_bboxes
seq_name = file_name_lidar.split('/')[2]
file_name_bboxes = extract_bboxes_file_name_from_image_file_name(file_name_image)
file_name_bboxes = join(root_path, seq_name, 'label3D/cam_front_center/', file_name_bboxes)
print (file_name_bboxes)
boxes = read_bounding_boxes(file_name_bboxes)
#./camera_lidar_semantic_bboxes/20180925_135056/label3D/cam_front_center/20180925135056_label3D_frontcenter_000003556.json
#dict_keys(['box_0', 'box_1', 'box_2'])
pprint.pprint(boxes)
[{'alpha': 0.0,
  'bottom': 1411.882,
  'center': array([17.13065, -3.3704 , -1.06485]),
  'class': 'Car',
  'left': 712.8365,
  'occlusion': 1.0,
  'right': 896.8972,
  'rotation': array([[ 9.99564440e-01,  2.95115171e-02,  1.03800706e-08],
       [-2.95115171e-02,  9.99564440e-01, -1.06910588e-08],
       [-1.06910588e-08,  1.03800706e-08,  1.00000000e+00]]),
  'size': array([3.53, 1.48, 1.53]),
  'top': 1204.545,
  'truncation': 0.0},
 {'alpha': 0.0,
  'bottom': 1773.064,
  'center': array([10.7088 , -3.38945, -0.8694 ]),
  'class': 'Car',
  'left': 667.6932,
  'occlusion': 1.0,
  'right': 1045.329,
  'rotation': array([[ 0.99985267, -0.01716511,  0.        ],
       [ 0.01716511,  0.99985267,  0.        ],
       [ 0.        ,  0.        ,  1.        ]]),
  'size': array([3.37, 1.82, 1.91]),
  'top': 1299.654,
  'truncation': 0.0},
 {'alpha': 0.0,
  'bottom': 2520.479,
  'center': array([ 6.27   , -3.33225, -0.95805]),
  'class': 'Car',
  'left': 705.9764,
  'occlusion': 1.0,
  'right': 1408.097,
  'rotation': array([[ 9.99618840e-01,  2.76075115e-02,  1.03902480e-08],
       [-2.76075115e-02,  9.99618840e-01, -1.06811681e-08],
       [-1.06811681e-08,  1.03902480e-08,  1.00000000e+00]]),
  'size': array([3.53, 1.7 , 1.72]),
  'top': 1496.03,
  'truncation': 1.0}]
#We can generate the vertices of a bounding box in a particular order. The following function generates the vertices in the following order: [bottom_rear_left, bottom_front_left, bottom_front_right, bottom_rear_right, top_rear_left, top_front_left, top_front_right, top_rear_right]

def get_points(bbox):
    half_size = bbox['size'] / 2.
    
    if half_size[0] > 0:
        # calculate unrotated corner point offsets relative to center
        brl = np.asarray([-half_size[0], +half_size[1], -half_size[2]])
        bfl = np.asarray([+half_size[0], +half_size[1], -half_size[2]])
        bfr = np.asarray([+half_size[0], -half_size[1], -half_size[2]])
        brr = np.asarray([-half_size[0], -half_size[1], -half_size[2]])
        trl = np.asarray([-half_size[0], +half_size[1], +half_size[2]])
        tfl = np.asarray([+half_size[0], +half_size[1], +half_size[2]])
        tfr = np.asarray([+half_size[0], -half_size[1], +half_size[2]])
        trr = np.asarray([-half_size[0], -half_size[1], +half_size[2]])
     
        # rotate points
        points = np.asarray([brl, bfl, bfr, brr, trl, tfl, tfr, trr])
        points = np.dot(points, bbox['rotation'].T)
        
        # add center position
        points = points + bbox['center']
  
    return points
#Let us get the vertices of the first bounding box in the list

points = get_points(boxes[0])
#Print the vertices
"""
print(points)
[[15.38825728 -2.57863448 -1.82984997]
 [18.91671975 -2.68281013 -1.82985001]
 [18.87304271 -4.16216551 -1.82985003]
 [15.34458023 -4.05798985 -1.82984999]
 [15.38825729 -2.57863449 -0.29984997]
 [18.91671977 -2.68281015 -0.29985001]
 [18.87304272 -4.16216552 -0.29985003]
 [15.34458025 -4.05798987 -0.29984999]]
Let us visualize the bounding boxes in the LiDAR space
"""
# Create or update open3d wire frame geometry for the given bounding boxes
def _get_bboxes_wire_frames(bboxes, linesets=None, color=None):

    num_boxes = len(bboxes)
        
    # initialize linesets, if not given
    if linesets is None:
        linesets = [o3.geometry.LineSet() for _ in range(num_boxes)]

    # set default color
    if color is None:
        #color = [1, 0, 0]
        color = [0, 0, 1]

    assert len(linesets) == num_boxes, "Number of linesets must equal number of bounding boxes"

    # point indices defining bounding box edges
    lines = [[0, 1], [1, 2], [2, 3], [3, 0],
             [0, 4], [1, 5], [2, 6], [3, 7],
             [4, 5], [5, 6], [6, 7], [7, 4], 
             [5, 2], [1, 6]]

    # loop over all bounding boxes
    for i in range(num_boxes):
        # get bounding box corner points
        points = get_points(bboxes[i])
        # update corresponding Open3d line set
        colors = [color for _ in range(len(lines))]
        line_set = linesets[i]
        line_set.points = o3.utility.Vector3dVector(points)
        line_set.lines = o3.utility.Vector2iVector(lines)
        line_set.colors = o3.utility.Vector3dVector(colors)

    return linesets
#Load LiDAR data again

lidar_front_center = np.load(file_name_lidar)
#reate Open3D point cloud

pcd_front_center = create_open3d_pc(lidar_front_center)
#Draw LiDAR points with bounding boxes

entities_to_draw = []
entities_to_draw.append(pcd_front_center)

for bbox in boxes:
    linesets = _get_bboxes_wire_frames([bbox], color=(255,0,0))
    entities_to_draw.append(linesets[0])
    
o3.visualization.draw_geometries(entities_to_draw)

#To visualize both the bounding boxes and 3D semantic segmentation

pcd_lidar_colored = create_open3d_pc(lidar_front_center, semantic_image_front_center_undistorted)
entities_to_draw = []
entities_to_draw.append(pcd_lidar_colored)

for bbox in boxes:
    linesets = _get_bboxes_wire_frames([bbox], color=(255, 0, 0))
    entities_to_draw.append(linesets[0])
    
o3.visualization.draw_geometries(entities_to_draw)

