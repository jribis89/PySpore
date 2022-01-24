# -*- coding: utf-8 -*-
"""
@author: John Ribis
"""
from ctypes import resize
import time, sys, os, math, pickle
import cv2
from skimage.registration import phase_cross_correlation
from skimage.filters import threshold_otsu, median, threshold_local, threshold_isodata
import skimage.morphology as morph
from skimage.segmentation import clear_border
from skimage import exposure
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFont, ImageDraw
import tqdm
from skimage.measure import label, regionprops, regionprops_table
from skimage.color import label2rgb
from scipy.interpolate import PchipInterpolator 
import multiprocessing as mp
import pandas as pd
import seaborn as sns
from skimage.io import MultiImage
from multiprocessing import Pool
from joblib import Parallel, delayed, cpu_count


class Pipeline:
    def __init__(self, paths, threshold, samp_interval = 1, exp_data = None, batch = False, masks = False, single = False, align = True):
        self.paths = paths
        self.threshold = threshold
        self.samp_interval = samp_interval
        self.exp_data = exp_data
        self.batch = batch
        self.masks = masks
        self.single = single
        self.exp_data = exp_data
        self.align = align
     
    def process(self):
        if self.batch is True and self.exp_data is not None:
            for self.path, self.img_name in zip(self.paths, self.exp_data):
                name = self.img_name#list(self.img_name)[0]
                self.strain = self.exp_data[name]['strain']
                self.replicate = self.exp_data[name]['replicate']
                self.position = self.exp_data[name]['position']
                self.run_pipeline(paths = self.path, threshold = self.threshold)

        elif self.batch is False and self.exp_data is not None:
            name = list(self.exp_data.keys())[0]
            self.strain = self.exp_data[name]['strain']
            self.replicate = self.exp_data[name]['replicate']
            self.position = self.exp_data[name]['position']            
            self.path = self.paths
            self.run_pipeline(threshold = self.threshold)

    def run_pipeline(self, paths = None, threshold = 0.05):
        if self.single is False and self.align is True:
            self.register_stack()
            self.segment_spores()
            self.track_spores()
            self.extract_features()
            self.save_data()
        
        elif self.single is False and self.align is False: #Can remove timings from this, eventually
            print('skipping registration')
            imgs = self.load_images()
            self.segment_spores(img = imgs)
            self.track_spores()
            self.extract_features(images = imgs)
            self.save_data()

        else:
            img = self.load_images()
            self.segment_spores(img = img, single = True)
            self.extract_features()
        
    def load_images(self, img_path = None):
        if img_path is not None:
            img_path = img_path
        else:
            img_path = self.path
        #had to change this from openCV imreadmulti. Still using cv2.imshow function for speed. Returns a 3d numpy array of all images in stack.
        imgs = MultiImage(img_path)[0]
        return imgs

    def register_stack(self, img_path = None):
        '''  
            Parameters
            ----------
            movie_dir : str containg path to .tif stack
                Function aligns movie frames using phase cross-correlation. Function 
                will likely work pretty well with any other time-lapse xy drift.
                THIS IS REALLY ONLY NEEDED WHEN THERE ARE BIG POSITIONAL MOVEMENTS FRAME TO FRAME. THE TRACKING IS FAIRLY ROBUST

            Returns
            -------
            Registered image stack as uint16 3d numpy array.
            '''

        #placing a default value for this so that the user can input a stack of images if they want to call this method 
        if img_path is not None:
            img_path = img_path
        else:
            img_path = self.path

        #load .tif stack 
        imgs = self.load_images(img_path)

        # Find movie dimensions
        dims = np.shape(imgs)
        x = dims[2]
        y = dims[1]
        t = dims[0]

        # initialize stack arrays
        registered_stack = np.zeros(dims, dtype=np.uint16)
        translated_moving = np.zeros((y, x), dtype=np.uint16)

        # loop through image stack
        print("Aligning Images...")

        for i in tqdm.tqdm(range(t)):

            # load/update moving and fixed image
            if i == 0:
                fixed = imgs[i]
                registered_stack[i, :, :] = fixed
                moving = imgs[i+1]
            elif i > 0 & i < t:
                registered_stack[i, :, :] = translated_moving
                fixed = translated_moving
                moving = imgs[i]
            else:
                break

            # Compute phase cross-correlation to find offset between frames
            shift = phase_cross_correlation(fixed, moving, upsample_factor=10)

            # generate warp matrix
            self.warp_matrix = np.eye(2, 3, dtype=np.float64)

            # populate warp matrix with x and y offset values computed from cross-correlaton
            self.warp_matrix[0, 2] = shift[0][1]  # xshift
            self.warp_matrix[1, 2] = shift[0][0]  # yshift

            # apply translation to align moiving and fixed image
            translated_moving = cv2.warpAffine(
                moving, self.warp_matrix, (fixed.shape[1], fixed.shape[0]))

            # resize translated image for display
            #aligned = cv2.resize(translated_moving, (int(1/2 * x), int(1/2 * y)))

            aligned = translated_moving

            # Display aligned images
            cv2.imshow('translated', aligned)
            cv2.waitKey(1)
            
        #get mask of zeros from the last image in the stack to crop all images prior to segmentation.
        #Need to do this to exclude spores and cells that touch the edge of the frame
        self.crop_mask = np.ones(np.shape(registered_stack[-1,:,:]), dtype = 'bool')
        self.crop_mask[registered_stack[-1,:,:] == 0] = False

        #generate new array of cropped image stack to be used for segmentation
        self.cropped_stack = np.stack([im[np.ix_(self.crop_mask.any(1),self.crop_mask.any(0))] 
                                for im in registered_stack],axis = 0)

        print('Images Successfully Aligned!')
        cv2.destroyAllWindows()
        
        #output the cropped stack
        return self.cropped_stack, self.warp_matrix, self.crop_mask

    
    def preprocess_image(self,img):
        """ Takes image and apply median filter and equalizes histogram """
        # Denoise image with median filter
        denoised = median(img, morph.selem.disk(2))

        # Equalize image histogram to improve contrast
        eq = exposure.equalize_adapthist(denoised)

        # apply top-hat transform to image. Need to use a large structuring element
        strel = morph.selem.disk(10)
        #openCV top-hat is considerably faster than scikit
        self.top_hat = cv2.morphologyEx(eq, cv2.MORPH_BLACKHAT, strel)

        return self.top_hat


    def get_mask(self, autothresh = False, strel_size = 8):

        # Do histogram based auto-thresholding (didnt always work well for images after top-hat)
        if autothresh is True:
            self.threshold = threshold_otsu(self.top_hat)
        else:
            #hard threshold seemed to work the best for these images 
            mask = np.uint8(self.top_hat > self.threshold)

        # do binary closing 
        strel = morph.selem.disk(strel_size)
        closed = morph.binary_closing(mask, strel)

        # Get rid of spore husks and other small junk
        opened = morph.area_opening(closed,area_threshold=100, connectivity=1)

        # remove cells and artifacts connected to the border of the mask,
        no_borders_mask = clear_border(opened)

        return no_borders_mask

    
    def segment_spores(self, img = None, single = False, threshold = None):
        print('Segmenting Spores...')
        #can pass in any image to segment by calling this method. Otherwise will default to the 'self' registered stack.
        if img is not None:
            img = img
        
        else:
            img = self.cropped_stack

        if threshold is not None:
            threshold = threshold
        else:
            threshold = self.threshold
        
        #initialize md array for labeled mask 
        self.labeled_mask = np.zeros(img.shape)

        if self.single is True or single is True:
            self.top_hat = self.preprocess_image(img)
            # clean up mask
            mask= self.get_mask(self.top_hat)

            # label mask
            self.labeled_mask = label(mask).astype('int32')

        else:
            #change this to autoselect the best number of cpu cores
            
            #Huge speed increase with joblib. Detects user cpus cores to distribute segmentation jobs.
            masks = Parallel(n_jobs= 6)(delayed(self.seg)(img[i,:,:]) for i in tqdm.tqdm(range(len(img))))
            self.labeled_mask = np.array(masks)

        return self.labeled_mask

    def seg(self, img):
        # preprocess image
        top_hat = self.preprocess_image(img)

        # clean up mask
        mask= self.get_mask(top_hat)

        # label mask
        labels = label(mask).astype('int32')
        
        return labels

    def link_pixels(self, refrence_mask, target_mask):

        # initialize matrix of zeros to populate with new labels
        corrected_mask = np.zeros(np.shape(refrence_mask)).astype('int32')
        
        #find the unique labels in the ref im
        labs = np.unique(refrence_mask[np.nonzero(refrence_mask)]).astype('int32')
        
        #loop though all of the labels in the ref im and chack to correct in target im
        for L1 in labs:
            
            m1 = refrence_mask == L1
            
            # find the most common label to see what the same component in the next image is labeled using the index from the first image
            target_label, counts = np.unique(target_mask[m1], return_counts=True)

            label_idx = counts.argmax()
            
            # find the label in the overlapping ROI in the next image
            L2 = target_label[label_idx].astype('int32')
            if L2 == 0:
                continue
            #mask tha
            else:
                m2 = target_mask == L2
            
            #if the overlapping label doesn't correspond then
            corrected_mask[m2] = L1

        return corrected_mask

    def track_spores(self):
        self.new_labs = np.zeros(np.shape(self.labeled_mask)).astype('int32')
        print('Tracking and Correcting IDs...')
        #algorithm compares labels based on location in the first image in the stack to the next, then uses the corrected image as a refrence for the rest.
        for i in tqdm.tqdm(range(len(self.labeled_mask))):
            
            if i + 1 == len(self.labeled_mask):
                break          
            else:
                if i == 0:
                    refrence_mask = self.labeled_mask[i,:,:]
                else:
                    refrence_mask = self.new_labs[i - 1,:,:]
                
                #define target mask
                target_mask = self.labeled_mask[i + 1,:,:]
                
                #Run function to spit out corrected mask
                corrected = self.link_pixels(refrence_mask, target_mask)

                #update refrence mask and populate new array with corrected image stack
                self.new_labs[i,:,:] = corrected

        print('Tracking Finished')    
        return self.new_labs
    
    def label_input_masks():
        pass

    def extract_features(self,label_stack = None, images = None):
    #add functionality to do fluorescence measurements as well, Need to pass intensity image as a list of registered image arrays.
    #initialize list for features dataframes.
    #HAVE DIALOG POPUP TO ALLOW USER TO ADD RELEVANT EXPERIMENTAL DATA (STRAIN, REPLICATE, POSITION)
        feats = []

        if images is not None:
            if label_stack is not None:
                label_stack = label_stack
            else:
                label_stack = self.new_labs    

            trans_images = images
        else:
            label_stack = self.new_labs
            trans_images = self.cropped_stack

        for mask, intensity_image in zip(label_stack, trans_images):

            table = pd.DataFrame((regionprops_table(mask, intensity_image = intensity_image, properties = 
                                                    ('label', 'bbox', 'centroid', 'area', 'perimeter', 'eccentricity', 
                                                    'solidity','mean_intensity', 'max_intensity'))))
            #set the dataframe index as the label.
            feats.append(table.set_index('label'))
        
        feat = []
        for ID in feats[0].index:
            print(ID)
            cell_feats = pd.concat([df.loc[ID] for df in feats if ID in df.index], axis = 1)
            transposed = cell_feats.transpose()
            transposed.index.names = ['ID']
            transposed.reset_index(inplace=True)

            transposed.insert(0, 'Time', np.arange(len(transposed))* self.samp_interval)
            
            feat.append(transposed)
        
        self.features = pd.concat(feat,axis = 0)
        num_cells = len(self.features)
        #add in experiment data
        self.features.insert(0, 'Position', [self.position] * num_cells)
        self.features.insert(0, 'Replicate', [self.replicate] * num_cells)
        self.features.insert(0, 'Strain', [self.strain] * num_cells)
        
        #return dataframe with all features and datapoints for all spores in the movie.
        self.features.reset_index(inplace = True)
        self.features.pop('index')
        
        return self.features


    #compile all data in a dictionary and pickle to save. Also automatically saves a .csv file
    def save_data(self):
        print('Saving Data to Pickle')
        output = {}
        path = os.path.splitext(self.path)[0]
        base, name = os.path.split(path)
        output['features'] = self.features
        if self.align is True:
            output['reg_ims'] = self.cropped_stack

        output['masks'] = self.new_labs 
        
        out_path = os.path.join(base,f'{name}_output.pkl')
        print(f'Output:{out_path}')

        print('Saving to .csv')
        csv_name = os.path.join(base,f'{name}_features.csv')
        print(csv_name)                        
        self.features.to_csv(csv_name, index=False)
        #save the dictionary with all data in a pickle file
        with open(out_path, "wb") as pkl_file:
            pickle.dump(output, pkl_file)
 
class DataHandling:
    #Just data io stuff and methods to join datasets
    def __init_(self, data_paths):
        self.data_path = data_paths

    def read_pkl(self, pkl_file = None):
        #load .pkl file.    
        with open(pkl_file, 'r+b') as pkl_file:
            self.data = pickle.load(pkl_file)
        return self.data
    
    def read_csv(self, csv_file = None):
        #load .csv_data
        self.data = pd.read_csv(csv_file)
        return self.data

    def join_data(self):
        #method concatenates feature data extracted from a .csv or from a .pkl file
        pass
        


   


class DataVis:
    '''Class contains a series of methods to visualize germination data '''
    def __init__(self, data):
        self.data = data
    
    def smooth_data(self, feature_key = 'mean_intensity', span = 5):
        '''Function smooths noisy continuous data a.la. matlabs' smooth fucntion'''
        #function to smooth noisy germination data. Function was taken from stack exchange:
        #https://stackoverflow.com/questions/40443020/matlabs-smooth-implementation-n-point-moving-average-in-numpy-python
        # data: pandas dataframe with data to be visualized
        # span: smoothing window size needs, which must be odd number,
        # as in the original MATLAB implementation
        # Need to be able to change the span in the GUI to see what works the best for the data
        feature_data = self.data[feature_key]

        out0 = np.convolve(feature_data, np.ones(span,dtype=int),'valid') / span    
        r = np.arange(1, self.span-1,2)
        start = np.cumsum(feature_data[:self.span-1])[::2] / r
        stop = (np.cumsum(feature_data[:-self.span:-1])[::2] / r) [::-1]
        smoothed = np.concatenate((start , out0, stop))

        #df_nat['mov_avg'] = df_nat['new_cases'].rolling(7).sum()
        return smoothed

    def fit_pchip(self, data = None, feature_key = 'mean_intensity', step = 1):
        #fit a pchip (similar to a spline) to data. 
        #Data tends to be noisy, so its often best to smooth the data beforehand to calculate stuff like max germination rate. 

        pchip  = PchipInterpolator(self.data['Time'], self.data[feature_key])
        #need to create a vector containing the range of values we're interested in 
        #and feed that into the pchip functions generated by the input data.
        start = self.data.Time[0]
        stop = self.data.Time[-1]
        
        x_new = np.arange(start, stop, step)
        #Evaluate fx for the values to be interpolated, and generate new interpolated data.
        self.interp = pchip(x_new)
        #Evaluate derivative of fx at the new values
        self.derivatives = pchip.derivative(x_new, nu = 1)

        return self.interp, self.derivatives

    def germination_rate(self):
        #use the derivatives output from the fit_spline method to find the maximum germination rate

        pass

    def random_subset(self):
        #method allows user to select a random subset of data to plot to avoid overplotting.
        #Takes subset of data for each strain. If replicates are present, that number needs to be taken from each replicate.
        pass

    def plot_curves(self, data, x = 'Time', y = 'mean_intensity', grouping = 'ID'):
        #plot lines of data using seaborn. 
        #grouping changes the hue parameter in seaborn. 
        #Can use any column variable. Have this work as a dropdown in pysimpleGUI
        sns.lineplot(data = data, x = x, y = y, hue = grouping)
    
    def plot_violins():
        pass

    def plot_jitter():
        pass

    def plot_double_y():
        # a = extract_features(new_labs, registered, replicate =1, strain = 'bsub', pos_number = 1, time_int = 2)
        # sns.lineplot(x = a.Time, y = a.mean_intensity, hue = a.ID, palette = palette)
        # ax2 = plt.twinx()
        # sns.lineplot(x = a.Time, y = a.area, hue = a.ID,palette = palette, legend = False)
        pass

#Standalone functions
def test_seg(image = None, path = None, threshold = 0.05):
    #instantiate a Pipeline object and call segment spores method
   
    #segment and return black top-hat transform as well
    pl = Pipeline(paths = path, threshold = threshold)
    seg = pl.segment_spores(img = image, single = True, threshold = threshold)
    #get contours with openCV
    im_copy = image.copy()
    im_copy = cv2.cvtColor(im_copy,cv2.COLOR_GRAY2RGB)
    contours, _ = cv2.findContours(seg, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    index = -1 #index of contour. Negative will draw all
    thickness = 1 #thickness of contour
    color = (0,65535,0) #weird things are happening when I convert my input image to 8 bit, so just keeping it like this for now.
    cv2.drawContours(im_copy, contours, contourIdx = index, color = color, thickness = thickness)
    
    #resize the image if it's really large. Need to change this down the road to make it so that the user can just resize the window by dragging.
    if im_copy.shape[1] > 1400 or im_copy.shape[0] > 1400:
        scale_percent = 40 # percent of original size
        width = int(im_copy.shape[1] * scale_percent / 100)
        height = int(im_copy.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized = cv2.resize(im_copy, dim) 
        cv2.imshow('Contours', resized) 
    else:  
        cv2.imshow('Contours', im_copy) 
   
    #https://stackoverflow.com/questions/35003476/opencv-python-how-to-detect-if-a-window-is-closed/37881722#37881722
    #Display segmentation result. Code below closes the open CV window using the 'x' 
    while cv2.getWindowProperty('window-name', 0) >= 0:
        keyCode = cv2.waitKey(50)

    return seg

def display_image(image):
#Trying to speed things up using openCV for image display.
    #resize the image if very large
    if image.shape[1] > 1400 or image.shape[2] > 1400:
        scale_percent = 40 # percent of original size
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized = cv2.resize(image, dim, cv2.INTER_CUBIC) 
        cv2.imshow('Image', resized) 
    else:  
        cv2.imshow('Image', image) 

    while cv2.getWindowProperty('window-name', 0) >= 0:
        keyCode = cv2.waitKey(50)

def load_images(image_directory):
    images = MultiImage(image_directory)[0]
    return images
def plot_runtime():
    pass
    # fig, ax = plt.subplots(figsize=(12,5))
    # ax.broken_barh([bar_load], (2,1), facecolors = 'tab:blue')
    # ax.broken_barh([bar_seg], (4,1), facecolors = 'tab:red')
    # ax.broken_barh([bar_track], (6,1), facecolors = 'tab:green')
    # ax.broken_barh([bar_features], (8,1),facecolors ='tab:purple')
    # ax.broken_barh([bar_save], (10,1), facecolors ='tab:orange')
    # ax.set_yticklabels(['Load Movie', 'Segmentation', 'Tracking', 'Feature Extraction', 'Saving Data'])
    # ax.set_xlabel('Time (min)')
    # ax.spines["top"].set_visible(False)  
    # ax.spines["right"].set_visible(False)
    #ax.set_yticks([2.5,4.5,6.5,8.5,10.5]) 