import numpy as np

class line():
    def __init__(self, loc = 0, ym_per_pix= (27/720), xm_per_pix = (3.7/80), window_width = 30, window_height = 80, margin = 20, smooth_factor = 15, max_skipped_windows = 2):
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = [] 
        #average x values of the fitted line over the last n iterations
        self.bestx = None     
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None  
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]  
        #radius of curvature of the line in some units
        self.radius_of_curvature_arr = [] 
        self.radius_of_curvature = 0 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None
        
        self.recent_centers = [] 
        self.window_width = window_width
        self.window_height = window_height
        self.margin = margin
        self.ym_per_pix = ym_per_pix
        self.xm_per_pix = xm_per_pix
        self.smooth_factor = smooth_factor
        self.loc = loc # location of lane 0 - left, 1 - right
        self.max_skipped_windows = max_skipped_windows

    def window_mask(self,width, height, img_ref, center,level):
        output = np.zeros_like(img_ref)
        output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
        return output

    def find_window_centroids(self, image):
        window_width = self.window_width
        window_height = self.window_height
        margin = self.margin
        
        window_centroids = [] # Store the window centroid positions per level
        window = np.ones(window_width) # Create our window template that we will use for convolutions
        
        # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
        # and then np.convolve the vertical image slice with the window template 
        
        # Sum quarter bottom of image to get slice, could use a different ratio
        if (self.loc == 0):
            l_sum = np.sum(image[int(3*image.shape[0]/4):,:int(image.shape[1]/2)], axis=0)
            l_center = np.argmax(np.convolve(window,l_sum))-window_width/2
        else:
            l_sum = np.sum(image[int(3*image.shape[0]/4):,int(image.shape[1]/2):], axis=0)
            l_center = np.argmax(np.convolve(window,l_sum))-window_width/2+int(image.shape[1]/2)
        
        # Add what we found for the first layer
        window_centroids.append((l_center))
        
        max_skipped_windows = self.max_skipped_windows
        
        # Go through each layer looking for max pixel locations
        for level in range(1,(int)(image.shape[0]/window_height)):
            
#            if (max_skipped_windows == 0):
#                break
            
            # convolve the window into the vertical slice of the image
            image_layer = np.sum(image[int(image.shape[0]-(level+1)*window_height):int(image.shape[0]-level*window_height),:], axis=0)
            conv_signal = np.convolve(window, image_layer)
            # Find the best left centroid by using past left center as a reference
            # Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
            offset = window_width/2
            l_min_index = int(max(l_center+offset-margin,0))
            l_max_index = int(min(l_center+offset+margin,image.shape[1]))
            l_conv = np.argmax(conv_signal[l_min_index:l_max_index])
            if (l_conv > 0):
                l_center = l_conv + l_min_index-offset
            else:
                l_center = l_center
                max_skipped_windows = max_skipped_windows - 1
            # Add what we found for that layer
            window_centroids.append((l_center))
    
        self.recent_centers.append(window_centroids)
        self.detected = True
        return np.average(self.recent_centers[-self.smooth_factor:],axis = 0)
    
        
    def find_lane_points(self, image):
        
        warped = image
        lane_window_centroids = self.find_window_centroids(warped)
        lane_visibility = 0
        if len(lane_window_centroids) > 0:

            # Points used to draw all the left and right windows
            l_points = np.zeros_like(warped)
            
            x_points = []
            lane_visibility = len(lane_window_centroids)   
            # Go through each level and draw the windows 	
            for level in range(0,lane_visibility):
            # Window_mask is a function to draw window areas
                l_mask = self.window_mask(self.window_width,self.window_height,warped,lane_window_centroids[level],level)
               
                x_points.append(lane_window_centroids[level])
                # Add graphic points from window mask here to total pixels found
                l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
                
        # Draw lanes only to the visible 
        
        yvals = range(0 , warped.shape[0])
        res_yvals = np.arange(warped.shape[0] - (self.window_height/2), 0, -self.window_height)
        
        lane_fit = np.polyfit(res_yvals, x_points, 2)
        lane_fitx = lane_fit[0]*yvals*yvals + lane_fit[1]*yvals + lane_fit[2]
        lane_fitx = np.array(lane_fitx, np.int32)
        
        
        #lane_points = np.array(list(zip(np.concatenate((lane_fitx - self.window_width/2,lane_fitx[::-1]+self.window_width/2), axis = 0), np.concatenate((yvals,yvals[::-1]), axis = 0))), np.int32)
        lane_points = np.array(list(zip(np.concatenate((lane_fitx,lane_fitx[::-1]), axis = 0), np.concatenate((yvals,yvals[::-1]), axis = 0))), np.int32)
        self.best_fit = lane_fit
        self.recent_xfitted.append(lane_fit)
        return lane_points,np.average(self.recent_xfitted[-self.smooth_factor:],axis = 0)
        
    def calculate_curve(self):
        lane_fit_in = self.best_fit
        ploty = np.linspace(0, 719, num=720)# to cover same y-range as image
        
        lanex = np.array([(y**2)*lane_fit_in[0] + (y)*lane_fit_in[1] + lane_fit_in[2]
                                  for y in ploty])
        
        
        lanex = lanex[::-1]  # Reverse to match top-to-bottom in y
        
        
        # Fit a second order polynomial to pixel positions in each fake lane line
        lane_fit = np.polyfit(ploty, lanex, 2)
        #lane_fitx = lane_fit[0]*ploty**2 + lane_fit[1]*ploty + lane_fit[2]
        
        y_eval = np.max(ploty)
        lane_curverad = ((1 + (2*lane_fit[0]*y_eval + lane_fit[1])**2)**1.5) / np.absolute(2*lane_fit[0])
        
        #print(lane_curverad, right_curverad)
        # Example values: 1926.74 1908.48
        
        
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = self.ym_per_pix # meters per pixel in y dimension
        xm_per_pix = self.xm_per_pix # meters per pixel in x dimension
        
        # Fit new polynomials to x,y in world space
        lane_fit_cr = np.polyfit(ploty*ym_per_pix, lanex*xm_per_pix, 2)
        
        # Calculate the new radii of curvature
        lane_curverad = ((1 + (2*lane_fit_cr[0]*y_eval*ym_per_pix + lane_fit_cr[1])**2)**1.5) / np.absolute(2*lane_fit_cr[0])
        
        # Now our radius of curvature is in meters
        #print(lane_curverad, 'm', right_curverad, 'm')
        self.radius_of_curvature_arr.append(lane_curverad)
        self.radius_of_curvature = np.average(self.radius_of_curvature_arr[-self.smooth_factor:],axis = 0)
        return lane_fit_cr, lane_curverad
    
    def find_lanes_limited(self, binary_warped):
        lane_fit = self.best_fit
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = self.margin
        lane_lane_inds = ((nonzerox > (lane_fit[0]*(nonzeroy**2) + lane_fit[1]*nonzeroy + 
        lane_fit[2] - margin)) & (nonzerox < (lane_fit[0]*(nonzeroy**2) + 
        lane_fit[1]*nonzeroy + lane_fit[2] + margin))) 
        
        # Again, extract lane and right line pixel positions
        lanex = nonzerox[lane_lane_inds]
        laney = nonzeroy[lane_lane_inds] 
        
        # Fit a second order polynomial to each
        lane_fit = np.polyfit(laney, lanex, 2)
        
        yvals = range(0 , binary_warped.shape[0])
        lane_fitx = lane_fit[0]*yvals*yvals + lane_fit[1]*yvals + lane_fit[2]
        lane_fitx = np.array(lane_fitx, np.int32)
        
        
        #lane_points = np.array(list(zip(np.concatenate((lane_fitx - self.window_width/2,lane_fitx[::-1]+self.window_width/2), axis = 0), np.concatenate((yvals,yvals[::-1]), axis = 0))), np.int32)
        lane_points = np.array(list(zip(np.concatenate((lane_fitx,lane_fitx[::-1]), axis = 0), np.concatenate((yvals,yvals[::-1]), axis = 0))), np.int32)

        self.best_fit = lane_fit
        self.recent_xfitted.append(lane_fit)
        
        return lane_points,np.average(self.recent_xfitted[-self.smooth_factor:],axis = 0)
 