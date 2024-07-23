"""
  Sample Run:
  python test/.py
"""
import os, sys
sys.path.append(os.getcwd())

import os.path as osp
import glob
import numpy as np
np.set_printoptions   (precision= 2, suppress= True)

import imageio
import cv2
from collections import Counter
from matplotlib import pyplot as plt


def get_calib_from_file(calib_file):
    with open(calib_file) as f:
        lines = f.readlines()

    obj = lines[2].strip().split(' ')[1:]
    P2 = np.array(obj, dtype=np.float32)
    obj = lines[3].strip().split(' ')[1:]
    P3 = np.array(obj, dtype=np.float32)
    obj = lines[4].strip().split(' ')[1:]
    R0 = np.array(obj, dtype=np.float32)
    obj = lines[5].strip().split(' ')[1:]
    Tr_velo_to_cam = np.array(obj, dtype=np.float32)

    return {'P2': P2.reshape(3, 4),
            'P3': P3.reshape(3, 4),
            'R0': R0.reshape(3, 3),
            'Tr_velo2cam': Tr_velo_to_cam.reshape(3, 4)}


def load_velodyne_points(filename):
    """Load 3D point cloud from KITTI file format
    (adapted from https://github.com/hunse/kitti)
    """
    points = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)
    points[:, 3] = 1.0  # homogeneous
    return points

def sub2ind(matrixSize, rowSub, colSub):
    """Convert row, col matrix subscripts to linear indices
    """
    m, n = matrixSize
    return rowSub * (n-1) + colSub - 1

def generate_depth_map(calib_filename, velo_filename, cam=2, vel_depth=False):
    """Generate a depth map from velodyne data
    https://github.com/nianticlabs/monodepth2/blob/master/kitti_utils.py
    """
    # # load calibration files
    # cam2cam = read_calib_file(os.path.join(calib_dir, 'calib_cam_to_cam.txt'))
    # velo2cam = read_calib_file(os.path.join(calib_dir, 'calib_velo_to_cam.txt'))
    # velo2cam = np.hstack((velo2cam['R'].reshape(3, 3), velo2cam['T'][..., np.newaxis]))
    # velo2cam = np.vstack((velo2cam, np.array([0, 0, 0, 1.0])))
    #
    # # get image shape
    # im_shape = cam2cam["S_rect_02"][::-1].astype(np.int32)
    #
    # # compute projection matrix velodyne->image plane
    # R_cam2rect = np.eye(4)
    # R_cam2rect[:3, :3] = cam2cam['R_rect_00'].reshape(3, 3)
    # P_rect = cam2cam['P_rect_0'+str(cam)].reshape(3, 4)
    # P_velo2im = np.dot(np.dot(P_rect, R_cam2rect), velo2cam)

    # width, height from file command in linux
    # Here we write as height, width
    im_shape = [1024, 1224]

    calib = get_calib_from_file(calib_filename)
    P2 = np.eye(4)
    P2[:3, :] = calib['P2']
    R0 = np.eye(4)
    R0[:3, :3]  = calib['R0']
    Tr_velo2cam = np.eye(4)
    Tr_velo2cam[:3, :] = calib['Tr_velo2cam']

    # x = P2 * R0_rect * Tr_velo_to_cam * y
    # https://github.com/abhi1kumar/groomed_nms/blob/main/data/kitti_split1/devkit/readme.txt#L96
    P_velo2im = P2 @ R0 @ Tr_velo2cam

    # load velodyne points and remove all behind image plane (approximation)
    # each row of the velodyne data is forward, left, up, reflectance
    velo = load_velodyne_points(velo_filename)
    velo = velo[velo[:, 0] >= 0, :]

    # project the points to the camera
    velo_pts_im = np.dot(P_velo2im, velo.T).T
    velo_pts_im[:, :2] = velo_pts_im[:, :2] / velo_pts_im[:, 2][..., np.newaxis]

    if vel_depth:
        velo_pts_im[:, 2] = velo[:, 0]

    # check if in bounds
    # use minus 1 to get the exact same value as KITTI matlab code
    velo_pts_im[:, 0] = np.round(velo_pts_im[:, 0]) - 1
    velo_pts_im[:, 1] = np.round(velo_pts_im[:, 1]) - 1
    val_inds = (velo_pts_im[:, 0] >= 0) & (velo_pts_im[:, 1] >= 0)
    val_inds = val_inds & (velo_pts_im[:, 0] < im_shape[1]) & (velo_pts_im[:, 1] < im_shape[0])
    velo_pts_im = velo_pts_im[val_inds, :]

    # project to image
    depth = np.zeros((im_shape[:2]))
    depth[velo_pts_im[:, 1].astype(np.int64), velo_pts_im[:, 0].astype(np.int64)] = velo_pts_im[:, 2]

    # find the duplicate points and choose the closest depth
    inds = sub2ind(depth.shape, velo_pts_im[:, 1], velo_pts_im[:, 0])
    dupe_inds = [item for item, count in Counter(inds).items() if count > 1]
    for dd in dupe_inds:
        pts = np.where(inds == dd)[0]
        x_loc = int(velo_pts_im[pts[0], 0])
        y_loc = int(velo_pts_im[pts[0], 1])
        depth[y_loc, x_loc] = velo_pts_im[pts, 2].min()
    depth[depth < 0] = 0

    return depth

if __name__ == '__main__':
    base_folder = "data/coda/"
    vmin= 0
    vmax= 50
    cmap= 'magma_r'

    for split in ["testing", "training"]:
        calib_folder = osp.join(base_folder, split, "calib")
        print(calib_folder)
        calib_files  = sorted(glob.glob(calib_folder + "/*.txt"))

        num_files  = len(calib_files)
        output_folder = calib_folder.replace("calib", "depth_image")
        os.makedirs(output_folder, exist_ok= True)

        print("{} files found".format(num_files))

        for i, calib_file in enumerate(calib_files):
            velo_file = calib_file.replace("calib", "velodyne").replace(".txt", ".bin")
            gt_depth = generate_depth_map(calib_file, velo_file) # h x w

            output_file = osp.join(output_folder, osp.basename(calib_file).replace(".txt", ".png"))
            cv2.imwrite(output_file, gt_depth)

            # plt.imshow(gt_depth, vmin= vmin, vmax= vmax, cmap= cmap)
            # plt.show()

            if (i + 1) % 500 == 0 or  (i + 1) == num_files:
                print("{} images done.".format(i+1))