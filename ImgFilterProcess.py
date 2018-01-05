import cv2
import numpy as np


def abs_sobel_thresh(img, orient='x', thresh_min=25, thresh_max=255):
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    # Here I'm using inclusive (>=, <=) thresholds, but exclusive is ok too
    binary_output[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    return binary_output

def mag_sobel_thresh(img, thresh_min=25, thresh_max=255):
    
    # Convert to grayscale //**
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    abs_sobelx = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    abs_sobely = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    gradmag = np.sqrt(abs_sobelx**2 + abs_sobely**2)
    scaled_gradmag = np.uint8(255*gradmag/np.max(gradmag))
    binary_output = np.zeros_like(scaled_gradmag)
    binary_output[(scaled_gradmag >= thresh_min) & (scaled_gradmag <= thresh_max)] = 1
    return binary_output
   
def dir_sobel_thresh(img, thresh_min=0, thresh_max=1.5):
    
    # Convert to grayscale //**
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    abs_sobelx = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    abs_sobely = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    graddir = np.arctan2(abs_sobely, abs_sobelx)
    binary_output =  np.zeros_like(graddir)
    binary_output[(graddir >= thresh_min) & (graddir <= thresh_max)] = 1
    return  binary_output

def red_thres(img, thresh_min = 25, thresh_max = 255):
    red = img[:,:,1]
    binary_output = np.zeros_like(red)
    binary_output[(red >= thresh_min) & (red <= thresh_max)]  = 1
    return binary_output

def blue_thres(img, thresh_min = 25, thresh_max = 255):
    blue = img[:,:,2]
    binary_output = np.zeros_like(blue)
    binary_output[(blue >= thresh_min) & (blue <= thresh_max)]  = 1
    return binary_output

def green_thres(img, thresh_min = 25, thresh_max = 255):
    green = img[:,:,1]
    binary_output = np.zeros_like(green)
    binary_output[(green >= thresh_min) & (green <= thresh_max)]  = 1
    return binary_output

def s_thres(img, thresh_min = 25, thresh_max = 255):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel >= thresh_min) & (s_channel <= thresh_max)] = 1
    return binary_output

def l_thres(img, thresh_min = 25, thresh_max = 255):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    l_channel = hls[:,:,1]
    binary_output = np.zeros_like(l_channel)
    binary_output[(l_channel >= thresh_min) & (l_channel <= thresh_max)] = 1
    return binary_output

def lab_b_thres(img, thresh_min = 25, thresh_max = 255):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l_channel = hls[:,:,2]
    binary_output = np.zeros_like(l_channel)
    binary_output[(l_channel >= thresh_min) & (l_channel <= thresh_max)] = 1
    return binary_output

def h_thres(img, thresh_min = 25, thresh_max = 255):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    h_channel = hls[:,:,0]
    binary_output = np.zeros_like(h_channel)
    binary_output[(h_channel >= thresh_min) & (h_channel <= thresh_max)] = 1
    return binary_output

def v_thres(img, thresh_min = 25, thresh_max = 255):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    v_channel = hsv[:,:,2]
    binary_output = np.zeros_like(v_channel)
    binary_output[(v_channel >= thresh_min) & (v_channel <= thresh_max)] = 1
    return binary_output

def gray_thres(img, thresh_min = 25, thresh_max = 255):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    binary_output = np.zeros_like(gray)
    binary_output[(gray >= thresh_min) & (gray <= thresh_max)] = 1
    return binary_output

def single_channel_thres(img, thresh_min = 25, thresh_max = 255):
    binary_output = np.zeros_like(img)
    binary_output[(img >= thresh_min) & (img <= thresh_max)] = 1
    return binary_output
    
def combine2bin(img1,img2):
    # Combine the two binary thresholds
    combined_binary = np.zeros_like(img1)
    combined_binary[(img1 == 1) | (img2 == 1)] = 1
    return combined_binary

