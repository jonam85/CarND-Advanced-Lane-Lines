import cv2
import numpy as np
import pickle
import os
import glob

def calibrate_camera(nx, ny, fnames):
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.
    
    
    objp = np.zeros((ny*nx,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)
    
    # Make a list of calibration images
    
    
    for idx, fname in enumerate(fnames):
        
        img = cv2.imread(fname)
    
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
            
        # If found, draw corners
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
        
            # Draw and display the corners
            cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
        
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    
    return mtx,dist  

def load_camera_mtx(nx = 9,ny = 6, pfile="wide_dist_pickle.p", fnames = glob.glob('camera_cal/calibration*.jpg')):
        
    if os.path.exists("wide_dist_pickle.p"):
        dist_pickle = pickle.load( open( "wide_dist_pickle.p", "rb" ) )
        mtx = dist_pickle["mtx"]
        dist = dist_pickle["dist"]
    
    # otherwise: calculate calibration data and save to pickle.
    else:
        mtx, dist = calibrate_camera(nx,ny, fnames)
        with open("wide_dist_pickle.p", "wb") as f:
            pickle.dump(
                    {
                    'mtx': mtx,
                    'dist': dist,
                }, f)
    
    return mtx, dist
