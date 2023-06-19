import os
import numpy as np
from scipy.optimize import fsolve
import math
import cv2
import h5py
import torch


# equation to solve to remove radial distortion from image
def radial_dist_equations(p, distortion, vector):
    x, y = p

    return (x + distortion * (x ** 3 + y ** 2) - vector[0][0],
            y + distortion * (x ** 2 + y ** 3) - vector[1][0])


# transforms quaternion to rotation matrix
def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2]])


# transforms rotation matrix to quaternion
def rotmat2qvec(Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz):
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    if qvec[0] < 0:
        qvec *= -1
    return qvec


# searchs corresponding file for camera and from there returns K, radial distortion coefficient and
# path to image
def get_intrinsics(workspace_path, img_name):
    # subdirectories = [file for file in os.listdir(workspace_path + '/dataset') if
    #                   os.path.isdir(os.path.join(workspace_path + '/dataset', file))]
    # if 'depth_images' in subdirectories:
    #     subdirectories.remove('depth_images')
    # for folder in subdirectories:
    #     path = workspace_path + '/dataset/' + folder + '/images'
    #     files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    #     if img_name in files:
    #         subfolder_path = workspace_path + '/dataset/' + folder
    #         break
    # # K, distortion = get_calibration_matrix(subfolder_path + '/data/markers_placed_JPEG.xml')
    # path = subfolder_path + '/images/' + img_name
    K = np.array([[1641.92, 0, 1080],
                  [0, 1641.92, 720],
                  [0, 0, 1]])
    distortion = 0
    return K, distortion


# return dict with all images in subdirectories including their extrinsics
def get_gps_poses(dir):
    subdirectories = [file for file in os.listdir(dir) if os.path.isdir(os.path.join(dir, file))]
    if 'depth_images' in subdirectories:
        subdirectories.remove('depth_images')
    poses = {}

    used_images = []
    for folder in subdirectories:
        img_path = dir + '/' + folder + '/images'
        used_images.extend([f for f in os.listdir(img_path) if os.path.isfile(os.path.join(img_path, f))])
        file_name = dir + '/' + folder + '/data/image_poses.txt'
        with open(file_name) as f:
            lines = f.readlines()
        for i in range(1, len(lines)):
            data = lines[i].split(',')
            timestamp = "".join(list(filter(str.isdigit, data[0]))).zfill(6) + '.png'
            R = qvec2rotmat([float(data[7]), float(data[4]), float(data[5]), float(data[6])]).tolist()
            coords = [[float(data[1])], [float(data[2])], [float(data[3])]]
            coords = np.append(coords, R)
            poses.update({timestamp: coords})
    poses = dict(sorted(poses.items()))
    used_images = sorted(used_images)

    with open(os.path.dirname(dir) + '/image_poses_full.txt', 'w') as f:
        f.write('timestamp, p_x, p_y, p_z, q_x, q_y, q_z, q_w\n')
        for img_name in used_images:
            try:
                coords = poses[img_name]
                q_new = rotmat2qvec(-coords[3], -coords[4], coords[5],
                                    -coords[6], -coords[7], coords[8],
                                    -coords[9], -coords[10], coords[11])
                f.write(img_name + ',' + str(coords[0]) + ',' + str(coords[1]) + ',' + str(coords[2]) + ',' +
                        str(q_new[3]) + ',' + str(q_new[0]) + ',' + str(q_new[1]) + ',' + str(q_new[2]) + '\n')
            except:
                pass

    return poses


# undistort pixel location and get depth in depth image
def get_depth(workspace_path, pixel_coords, name_image, K, distortion):
    depth_img_name = workspace_path + '/depth_images/' + name_image + '.h5'
    f = h5py.File(depth_img_name, 'r')
    mykeys = list(f.keys())
    dset = f[mykeys[0]]
    data = np.array(dset[:, :, :])

    depths, coords_to_remove = [], []
    for coords in pixel_coords:
        distorted_coords = np.matmul(np.linalg.inv(K), np.array([[coords[0]], [coords[1]], [1.0]]))
        x, y = fsolve(radial_dist_equations, (1, 1), args=(distortion, distorted_coords))
        undistorted_coords = np.matmul(K, np.array([[x], [y], [1.0]]))
        try:
            depths.append(data[0][int(undistorted_coords[1][0])][int(undistorted_coords[0][0])])
        except:
            coords_to_remove.append(coords)
    for coords in coords_to_remove:
        pixel_coords.remove(coords)

    return depths


# transform pixel coordinates to world coordinates
def get_world_coords(workspace_path, pixel_coords, depth, name_image, K, distortion):
    # computation of camera intrinsics and extrinsics
    with open(os.path.dirname(workspace_path) + '/image_poses_full.txt') as f:
        lines = f.readlines()
    for line in lines:
        data = line.split(',')
        if data[0] == name_image:
            R = qvec2rotmat([float(data[7]), float(data[4]), float(data[5]), float(data[6][:-2])])
            a = np.array([[float(data[1])], [float(data[2])], [float(data[3])]])

    # compute point (a_plane) and normal vector (nvec) of plane with distance depth to camera center (a)
    ndist = np.matmul(np.linalg.inv(K), np.array([[K[0,2]], [K[1,2]], [1.0]]))

    x, y = fsolve(radial_dist_equations, (1, 1), args=(distortion, ndist))
    nvec = np.matmul(np.linalg.inv(R), np.array([[x], [y], [1.0]]))
    nvec = np.array([[nvec[0][0]], [nvec[1][0]], [nvec[2][0]]]) / math.sqrt(
        nvec[0][0] ** 2 + nvec[1][0] ** 2 + nvec[2][0] ** 2)

    world_points = []
    for i in range(len(pixel_coords)):
        img_coords = np.array([[pixel_coords[i][0]], [pixel_coords[i][1]], [1.0]])
        a_plane = np.add(a, depth[i] * nvec)

        # take camera intrinsics into account
        tdist = np.matmul(np.linalg.inv(K), img_coords)

        # take radial distortion into account
        x, y = fsolve(radial_dist_equations, (1, 1), args=(distortion, tdist))
        tvec = np.matmul(np.linalg.inv(R), np.array([[x], [y], [1.0]]))

        # compute intersection of ray through specified pixel (ray = a + lambda_param * tvec) with plane that
        # is depth away from camera (point: a_plane, normal vector: nvec)
        lambda_param = (np.subtract(a_plane, a)[0][0] * nvec[0][0] + np.subtract(a_plane, a)[1][0] * nvec[1][0] +
                        np.subtract(a_plane, a)[2][0] * nvec[2][0]) / (
                               tvec[0][0] * nvec[0][0] + tvec[1][0] * nvec[1][0] + tvec[2][0] * nvec[2][0])
        world_point = np.add(a, lambda_param * tvec)
        world_points.append(world_point)

    return world_points


# transforms world point to pixel coordinates
def get_image_coords(workspace_path, world_points, name_image, K, distortion):
    # computation of camera extrinsics
    with open(os.path.dirname(workspace_path) + '/image_poses_full.txt') as f:
        lines = f.readlines()
    for line in lines:
        data = line.split(',')
        if data[0] == name_image:
            R = qvec2rotmat([float(data[7]), float(data[4]), float(data[5]), float(data[6][:-2])])
            a = np.array([[float(data[1])], [float(data[2])], [float(data[3])]])

    Tr = - np.matmul(R, a)

    # computations of camera intrinsics and extrinsics
    T_world_to_cam = [[R[0][0], R[0][1], R[0][2], Tr[0][0]],
                      [R[1][0], R[1][1], R[1][2], Tr[1][0]],
                      [R[2][0], R[2][1], R[2][2], Tr[2][0]],
                      [0.0, 0.0, 0.0, 1.0]]

    img_points = []
    for world_point in world_points:
        # transform from world to camera frame and normalize
        camera_coords = np.matmul(T_world_to_cam,
                                  np.array([[world_point[0][0]], [world_point[1][0]], [world_point[2][0]], [1.0]]))
        camera_coords = camera_coords / camera_coords[2][0]

        # take radial distortion into account
        camera_coords = np.array(
            [[camera_coords[0][0] + distortion * (camera_coords[0][0] ** 3 + camera_coords[1][0] ** 2)],
             [camera_coords[1][0] + distortion * (camera_coords[0][0] ** 2 + camera_coords[1][0] ** 3)],
             [1.0]])

        # take camera intrinsics into account
        img_point = np.matmul(K, camera_coords)
        img_points.append([img_point[0][0], img_point[1][0]])

    return img_points


# remove transformed points that were mapped out of image
def remove_out_of_scope_points(pixel_target, pixel_coords, height, width):
    pixel_target_filtered, pixel_coords_filtered = [], []
    for i in range(len(pixel_target)):
        if pixel_target[i][0] >= 0 and pixel_target[i][0] < width:
            if pixel_target[i][1] >= 0 and pixel_target[i][1] < height:
                pixel_target_filtered.append(pixel_target[i])
                pixel_coords_filtered.append(pixel_coords[i])
    return pixel_target_filtered, pixel_coords_filtered


# transform from reference image to target image
def get_correspondence(path_to_ws, img_name_ref, img_name_target, pixel_coords, H, W):
    K_ref, distortion_ref = get_intrinsics(path_to_ws, img_name_ref)
    K_target, distortion_target = get_intrinsics(path_to_ws, img_name_target)
    depth = get_depth(path_to_ws, pixel_coords, img_name_ref, K_ref, distortion_ref)
    world_points = get_world_coords(path_to_ws, pixel_coords, depth, img_name_ref, K_ref, distortion_ref)
    pixel_target = get_image_coords(path_to_ws, world_points, img_name_target, K_target, distortion_target)

    pixel_target, pixel_coords = remove_out_of_scope_points(pixel_target, pixel_coords, H, W)
    return np.array(pixel_target), np.array(pixel_coords)