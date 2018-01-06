from CameraCalibration import load_camera_mtx
import glob
import cv2
import numpy as np

def gen_perpective_transform():
    img_height = 720
    img_width = 1280

    r_offset = 600
    l_offset = 600
    b_offset = -20
    t_offset = -500

    src = np.float32([[592,450],[682,450],[1116,720],[200,720]])
    dst = np.float32([[(l_offset, t_offset)], [((img_width-r_offset),t_offset)], [((img_width-r_offset),(img_height-b_offset))], [(l_offset,(img_height-b_offset))]])
    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)
    return M, Minv


def undistort_img(img, mtx, dist):
    undistorted = cv2.undistort(img, mtx, dist)
    return undistorted

def warp(img):
    M, Minv = gen_perpective_transform()
    img_size = (img.shape[1], img.shape[0])
    
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    return warped

def unwarp(img):
    M, Minv = gen_perpective_transform()
    img_size = (img.shape[1], img.shape[0])
    
    unwarped = cv2.warpPerspective(img, Minv, img_size, flags=cv2.INTER_LINEAR)
    return unwarped
    
def undistort_warp(img):
    mtx, dist = load_camera_mtx()
    undistorted = undistort_img(img, mtx, dist)
#    print(mtx)
#    print(dist)

    warped = warp(undistorted)
    return warped

#fnames = glob.glob('test_images/s*.jpg')
#for idx, fname in enumerate(fnames):
#img = cv2.imread('test_images/straight_lines1.jpg')
##    print(fname)
#fname = 'test_images/str_undist.jpg'
#mtx, dist = load_camera_mtx()
#img = undistort_img(img, mtx, dist)
#cv2.imwrite(fname, img)
    