import argparse
import cv2
import numpy as np
from ImgFilterProcess import red_thres, green_thres, mag_sobel_thresh, lab_b_thres, v_thres, single_channel_thres, blue_thres, abs_sobel_thresh, h_thres, s_thres, l_thres, combine2bin
import glob
from Undistort_Warp import undistort_warp,unwarp
from FindingLanes import line
from moviepy.editor import VideoFileClip


def get_color_threshold_binary(RGB_Img):
    
    # Thresholds Checking for White lanes using R- Channel
    r_min = 200 
    r_max = 255
    
    # Thresholds for checking Yellow lanes using S- channel
    s_min = 90 
    s_max = 180
    
    # Thresholds Checking for White lanes using V- Channel
    v_min = 225 
    v_max = 255
    
    #Applying thresholds and combining binary
    R_Img = red_thres(RGB_Img, r_min, r_max)
    S_Img = s_thres(RGB_Img, s_min, s_max)
    V_Img = v_thres(RGB_Img, v_min, v_max)
    RS_Img = combine2bin(R_Img, S_Img)
    RSV_Img = combine2bin(RS_Img, V_Img)
    
    return RSV_Img

def Video_Process(RGB_Img):
    
    #Used the Conversion if the input is RGB
    BGR_Img = cv2.cvtColor(RGB_Img, cv2.COLOR_RGB2BGR)
    
    #Warped the image so that the lane lines are vertical
    warp_img = undistort_warp(BGR_Img)
    
    #Applied sobel in x direction to get the vertical lines detected
    #CV_64F is used to find both positive and negative gradients
    sobel_img = cv2.Sobel(warp_img, cv2.CV_64F, 1, 0)
    
    #Converted the 64F output to Uint8
    minx = np.min(sobel_img)
    sobel_img = sobel_img-minx #to have only positive values
    maxx=np.max(sobel_img) 
    div = maxx / float(255) #calculate the normalize divisor
    if div == 0:
        div = 1
    sobelx_8u = np.uint8(np.round(sobel_img / div))
    
    #Used the S feature of the gradient to extract yellow lane lines
    sobelx_8uS = cv2.cvtColor(sobelx_8u, cv2.COLOR_RGB2HLS)
    sobelx_8uS = sobelx_8uS[:,:,2]  
    
    #Used the R feature of gradient to extract white lines.
    sobelx_8u = sobelx_8u[:,:,2]

    #A kernel in the negative sobel direction is created.
    #Note that applying sobel would have created both positive and negative
    #gradients back to back for a line in previous image.
    #Filtered out whether this pattern is appearing in the image using the 
    #created kernel.
    kernel = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    sobelx_8u3 = cv2.filter2D(sobelx_8u, cv2.CV_8U, kernel)
    sobelx_8u3S = cv2.filter2D(sobelx_8uS, cv2.CV_8U, kernel)
    
    #Applied thresholds to conver the outputs to binary.
    sobelx_8u3 = single_channel_thres(sobelx_8u3, 150, 255)
    sobelx_8u3S = single_channel_thres(sobelx_8u3S, 30, 255)
    sobelx_8u3 = combine2bin(sobelx_8u3S,sobelx_8u3)
    
    #Created a binary Image to process further.
    Binary_Img = combine2bin(sobelx_8u3,sobelx_8u3)
    
    
    #Used the color thresholds to have better capturing of lanes.
    if (Color_Threshold_Enable == 1):
        
        RSV_Img = get_color_threshold_binary(RGB_Img)
        undist_RSV = undistort_warp(RSV_Img)
        Binary_Img = combine2bin(sobelx_8u3,undist_RSV)
        
    #Applied a mask to search for lanes only in the region of interest.
    mask = np.zeros_like(Binary_Img)   
    ignore_mask_color = 255   
    vertices = np.array([[(350,0),(930,0), (690,720), (590,720)]], dtype=np.int32)
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_binary = cv2.bitwise_and(Binary_Img, mask)
    
    warped = masked_binary
    
    #Checking if the lane is detected properly in last frame.
    if(left_center.detected == True):
        #If detected, used the limited function to check for lane lines within the surroundings
        left_lane, left_fit = left_center.find_lanes_limited(warped)
    else:
        #Else Find the co-ordinates of each of the lanes using the sliding window method.
        left_lane, left_fit = left_center.find_lane_points(warped)
    
    #Check is done for both left and right lane lines
    if(right_center.detected == True):
        right_lane, right_fit = right_center.find_lanes_limited(warped)
    else:
        #Find the co-ordinates of each of the lanes
        right_lane, right_fit = right_center.find_lane_points(warped)

    #Calculated the curvature of each lanes.
    right_fit_reversed, right_curvature = right_center.calculate_curve()
    left_fit_reversed, left_curvature = left_center.calculate_curve()    
    
    output = RGB_Img
    
    #The sanity check for the lane length is performed.
    lanes_diff_check = right_fit_reversed[2] - left_fit_reversed[2]
    
    if ((lanes_diff_check) > 4) or ((lanes_diff_check) < 3):
        #If the sanity check fails, discarded the current detections, reset to perform sliding window method again.
        right_center.detected = False
        left_center.detected = False
        #if(len(right_center) > 0):
        right_center.recent_xfitted.pop()
        #if(len(left_center) > 0):
        left_center.recent_xfitted.pop()
        left_lane, left_fit = left_center.find_lane_points(warped)
        right_lane, right_fit = right_center.find_lane_points(warped)
         
    #For the overlay, a dummy road image is created
    road = np.zeros_like(BGR_Img)
    ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0] )
    
    #Based on the curve fits, line points are calculated.
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
    
    #Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    
    pts = np.hstack((pts_left, pts_right))
    
    # Draw the lane onto the warped blank image
    cv2.fillPoly(road, np.int_([pts]), (0,255, 0))
     

    #For visualizing lane lines
    cv2.fillPoly(road, [left_lane],color = [255,0,0])
    cv2.fillPoly(road, [right_lane],color = [0,0,255])
    
    #Unwarped the drawn lane lines
    road_warped = unwarp(road)
     
    #Output image is overlaid with the polygon drawn from lane lines
    output = cv2.addWeighted(RGB_Img, 1, road_warped, 0.8, 0)
  
    #Added Texts to display Curvature and deviations      
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    cv2.putText(output, 'Curvature: Left Lane = %.2lf m, Right Lane = %.2lf m' % (left_center.radius_of_curvature, right_center.radius_of_curvature), \
                (70, 70), font, 1, (0, 255, 80), 2, cv2.LINE_AA)
    cv2.putText(output, 'Lane offset from Road Center: %.2lf m' % (xm_per_pix*(output.shape[1]/2 - ((left_fit[2]+right_fit[2])/2))), \
                (70, 110), font, 1, (0, 255, 80), 2, cv2.LINE_AA)
    cv2.putText(output, 'Color_Filter: %d, w: %d, h: %d, m:%d, sf:%d' % (Color_Threshold_Enable,left_center.window_width,left_center.window_height,left_center.margin, left_center.smooth_factor), \
                (70, 150), font, 1, (0, 255, 80), 2, cv2.LINE_AA)

    return output

# window settings
window_width = 10 
window_height = 80 
margin = 20 
Color_Threshold_Enable = 0


Output_video = 'project_video_out.mp4'
Input_video = 'project_video.mp4'

ym_per_pix = (32/720)
xm_per_pix = (3.7/80)



#
#fnames = glob.glob('test_images/test*.jpg', recursive = True)
#max_files = len(fnames)
#file_index = 0
#RGB_Img = mpimg.imread(fnames[file_index])
#
#Gray = cv2.cvtColor(RGB_Img, cv2.COLOR_RGB2GRAY)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Advanced Lane Detection')
    parser.add_argument(
        '--i',
        type=str,
        default='project_video.mp4',
        help='File name of the input video file.'
    )
    parser.add_argument(
        '--o',
        type=str,
        default='project_video_out.mp4',
        help='File name of the output video file.'
    )
    parser.add_argument(
        '--w',
        type=int,
        default=10,
        help='Width of the sliding window')
    parser.add_argument(
        '--h',
        type=int,
        default=80,
        help='Height of the sliding window')
    parser.add_argument(
        '--m',
        type=int,
        default=20,
        help='Margin of the area to search for lanes')
    parser.add_argument(
        '--c',
        type=int,
        default=1,
        help='Enable/Disable Thrsholds based on Color')
    parser.add_argument(
        '--sf',
        type=int,
        default=15,
        help='Number for past frames to be used for Smoothing')
    
    args = parser.parse_args()

    window_width = args.w 
    window_height = args.h
    margin = args.m
    Color_Threshold_Enable = args.c
    Output_video = args.o
    Input_video = args.i
    smooth_factor = args.sf
    
    left_center = line(loc = 0, ym_per_pix= ym_per_pix, xm_per_pix = xm_per_pix, window_width = window_width, window_height = window_height, margin = margin, smooth_factor = smooth_factor)
    right_center = line(loc = 1, ym_per_pix= ym_per_pix, xm_per_pix = xm_per_pix, window_width = window_width, window_height = window_height, margin = margin, smooth_factor = smooth_factor)
        
    
    #    for idx, fname in enumerate(fnames):
    #        RGB_Img = cv2.imread(fname)
    #        write_name = 'test_images/' + str(idx) + '.jpg' 
    #        out = Video_Process(RGB_Img)
    #        cv2.imwrite(write_name, out)
    clip = VideoFileClip(Input_video)
    video_clip = clip.fl_image(Video_Process)
    video_clip.write_videofile(Output_video, audio = False)
