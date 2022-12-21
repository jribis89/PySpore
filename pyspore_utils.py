# -*- coding: utf-8 -*-
"""
@author: John Ribis
"""
import time, os, pickle
import cv2
from skimage.registration import phase_cross_correlation
from skimage.filters import threshold_otsu, median
import skimage.morphology as morph
from skimage.segmentation import clear_border
from skimage import exposure
import numpy as np
import tqdm
from skimage.measure import label, regionprops_table
import pandas as pd
from skimage.io import MultiImage
from multiprocessing import Pool
from joblib import Parallel, delayed, cpu_count
import plotly.express as px


class Pipeline:
    def __init__(self, paths, threshold, samp_interval=1, exp_data=None, batch=False, masks=False, single=False, align=True):
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
                self.condition = self.exp_data[name]['condition']
                self.position = self.exp_data[name]['position']
                self.run_pipeline(paths = self.path, threshold = self.threshold)

        elif self.batch is False and self.exp_data is not None:
            name = list(self.exp_data.keys())[0]
            self.strain = self.exp_data[name]['strain']
            self.replicate = self.exp_data[name]['replicate']
            self.condition = self.exp_data[name]['condition']
            self.position = self.exp_data[name]['position']            
            self.path = self.paths
            self.run_pipeline(threshold = self.threshold)

    def run_pipeline(self, paths = None, threshold = 0.05):
        if self.single is False and self.align is True:
            start = time.time()
            imgs = self.load_images()
            self.register_stack()
            self.segment_spores()
            self.track_spores()
            self.extract_features()
            self.save_data()
            end = time.time()
            print(end - start)
        
        elif self.single is False and self.align is False:
            start = time.time()
            print('skipping registration')
            imgs = self.load_images()
            self.segment_spores(img = imgs)
            self.track_spores()
            self.extract_features(images = imgs)
            self.save_data()
            end = time.time()
            print(end - start)

        else:
            img = self.load_images()
            self.segment_spores(img = img, single = True)
            self.extract_features()
        
    def load_images(self, img_path = None):
        '''Loads images as a list of 3D numpy stacks.
        Parameters: img_path: path(s) to .tif stack. Keep as none unless using as standalone method.
        Returns: 3d numpy array of images '''
        if img_path is not None:
            img_path = img_path
        else:
            img_path = self.path

        #Returns a 3d numpy array of all images in stack.
        imgs = MultiImage(img_path)[0]
        return imgs
    

    def register_stack(self, img_path = None):
        ''' 
            Parameters
            ----------
            img_path : (string) path to .tif stack
                Function aligns movie frames using phase cross-correlation. 
                THIS IS REALLY ONLY NEEDED WHEN THERE ARE BIG POSITIONAL MOVEMENTS FRAME TO FRAME. 
                THE TRACKING IS FAIRLY ROBUST TO DRIFT.

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
        """Inputs: single image
           Returns: black-hat, median filtered, histogram equalized image
           Description:"""
        # Denoise image with median filter
        denoised = median(img, morph.selem.disk(2))

        # Equalize image histogram to improve contrast
        eq = exposure.equalize_adapthist(denoised)

        # apply top-hat transform to image. Need to use a fairly large structuring element for spore coat
        strel = morph.selem.disk(10)
        #openCV top-hat is considerably faster than scikit
        self.top_hat = cv2.morphologyEx(eq, cv2.MORPH_BLACKHAT, strel)

        return self.top_hat


    def get_mask(self, autothresh = False, strel_size = 8):

        # Can do histogram based auto-thresholding (didnt always work well for images after top-hat)
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

        '''Method runs segmentation procedure on phase image'''

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

            #Huge speed increase with joblib. Detects user cpus cores -1 to distribute segmentation jobs.
            masks = Parallel(n_jobs= -2)(delayed(self.seg)(img[i,:,:]) for i in tqdm.tqdm(range(len(img))))
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

        # initialize empty mask of zeros to populate with new labels
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
            
            #if the overlapping label doesn't correspond then fix it.
            corrected_mask[m2] = L1

        return corrected_mask

    def track_spores(self):
        """
        Returns: 3d matrix (stack) of labeled masks with corrected IDs."""
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
                    refrence_mask = self.new_labs[i-1,:,:]
                
                #define target mask
                target_mask = self.labeled_mask[i+1,:,:]
                
                #Run function to spit out corrected mask
                corrected = self.link_pixels(refrence_mask, target_mask)

                #update refrence mask and populate new array with corrected image stack
                self.new_labs[i,:,:] = corrected

        print('Tracking Finished')    
        return self.new_labs
    

    def extract_features(self,label_stack = None, images = None):
    #add functionality to do fluorescence measurements as well, Need to pass intensity image as a list of registered/unregistered image arrays.
    #initialize list for features dataframes.
        
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
        #add experiment data to dataframe
        self.features.insert(0, 'Position', [self.position] * num_cells)
        self.features.insert(0, 'Replicate', [self.replicate] * num_cells)
        self.features.insert(0, 'Condition', [self.condition] * num_cells)
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

        print('Saving data to .csv')
        csv_name = os.path.join(base,f'{name}_features.csv')
        print(csv_name)                        
        self.features.to_csv(csv_name, index=False)
        #save the dictionary with all data in a pickle file
        with open(out_path, "wb") as pkl_file:
            pickle.dump(output, pkl_file)

 
class DataHandling:

    def __init__(self, data_paths, timeint = 30, time_offset = 6.5, min_frames = 100, max_area= 300):
        self.data_paths = data_paths
        self.timeint = timeint
        self.time_offset = time_offset
        self.min_frames = min_frames
        self.max_area = max_area

    def process_data(self):
        self.data = self.read_csv()
        if len(self.data_paths) > 1:
            joined = self.join_data()
            cleaned = self.clean_data(joined)
        else:
            cleaned = self.clean_data(self.data)
        return self.assign_uniqueid(cleaned)


    def read_pkl(self, pkl_file = None):
        #load .pkl file.    
        with open(pkl_file, 'r+b') as pkl_file:
            self.data = pickle.load(pkl_file)
        return self.pkldata
    
    def read_csv(self):
        #load .csv_data into a list of pandas dataframes or just as a single dataframe if only oone path is present.
        if len(self.data_paths) > 1:
            self.data = [pd.read_csv(path) for path in self.data_paths]
        else:
            self.data = pd.read_csv(self.data_paths[0])
        return self.data

    def join_data(self):
        #method concatenates whatever pandas dataframes are selected by user.
        #will take a list of dataframes and concatenate them.
        #concatentate everything into a huge dataframe.
        joined = pd.concat(self.data)
        joined_data = self.assign_uniqueid(joined)
        return joined_data

    def assign_uniqueid(self,df):
        #function assigns new identifiers to all cells in dataframe. Runs slow but works. Only needs to be run on a concatenated df.
        #initialize column in dataframe.
        print('Assigning new IDs...')
        df['unique_id'] = np.nan
        groups = df.groupby(['Strain', 'Replicate', 'Position', 'Condition', 'ID'])['ID'].count()
        df = df.set_index(['Strain', 'Replicate', 'Position', 'Condition', 'ID']).sort_index()
        #Loop through and just assign IDs sequentially to the data
        for i,index in enumerate(groups.index):
            df.loc[index,'unique_id'] = int(i)
        df.reset_index(inplace=True)
        return df
    
    def clean_data(self,df):
        #Simple function that removes shit data and assigns unique IDs for the frame to make plotting work with concatenated data.
        #take spores that are round enough and have an area corresponding to a single spore. Max area is in pixels. Add a conversion to correct for magnification wtih area threshold.
        #Take area at first timepoint
        cleaned = df.loc[df['area'] < self.max_area].copy()
        #make a column witht he absolute difference in area frame-frame (can use this to get rid of unrealistically jumpy data)
        cleaned['area_vari'] = cleaned['area'].diff().abs()
        #correct time and convert to minutes
        cleaned.loc[:,'Time'] = cleaned['Time'].multiply((self.timeint/60)) + self.time_offset
        #get rid of spores that werent tracked for long enough (can look at any variable here)
        counts = cleaned.groupby(['Strain','Position','Replicate', 'Condition', 'ID']).count()
        #only take spores tracked for a minimum of 100 frames then use this to generate a trimmed dataframe only with the data that we want.
        best_tracks = counts[counts['Time']>self.min_frames].reset_index()
        index_best = best_tracks.iloc[:,:5].copy()
        options = ['Strain','Position','Replicate', 'Condition','ID']
        #merge was the simplest solution to extract the best values
        trimmed = index_best.merge(cleaned, on = options)
        
        return trimmed
    
    def max_germ_rate(self, feature_key = 'mean_intensity', step = 1):
        #run function on entire dataframe 
        #fit pchip to get rate info
        xdata = self.data.Time.to_numpy()
        #fit pchip to data to get piecewise polynomials and take first derivative.
        pchip  = PchipInterpolator(xdata, self.data[feature_key]).derivative(nu=1)
        #create a new array of x-data (may not actually need to do this, can probably evaluate with the )
        start = xdata[0]
        stop = xdata[-1]
        
        x_new = np.arange(start, stop, step)

        #Evaluate piecewise derivatives with the new x-array and take absolute minimum value to get germiantion rate
        derivatives = abs(min(pchip(x_new)))
        return derivatives


    def get_int_ratio(self, df):
        #find first and last time of trace
        t_0 = df.Time.min()
        t_end = df.Time.max()

        #get first and last intensity, indexed by time
        int_0 = df.mean_intensity.loc[df.Time==t_0].to_numpy()
        int_end = df.mean_intensity.loc[df.Time==t_end].to_numpy()

        print(int_0, int_end)
        
        #Take the ratio of intensities and return as a float
        ratio = int_0/int_end

        #print(ratio)
        
        return ratio.astype(np.float64)[0]

    def germ_time(series, feature_key='mean_intensity', timeint = 30, window_size = 4, time_offset = 6.5):
        #Function is meant to run on a series following a groupby
        #function finds the time to germination usign a basic sliding window algorithm
        #Time interval between frames in seconds
        #time offset corresponds to the amount of time passed until the acquisition was started
        timeadj = (timeint/60) 

        #convert series data as a numpy vector for speed and simplicity
        timeseries = series[feature_key].to_numpy()

        #Set pointer indices corresponding to the window size
        pointer_start = 0
        pointer_end = window_size - 1
        end = len(timeseries)

        diffarr = np.empty(end-window_size)
        #run window across array taking the difference between the first and last values in the window
        
        while pointer_end < end:
            p1 = timeseries[pointer_start]
            p2 = timeseries[pointer_end]

            #populate array with difference between the first and second pointer
            diffarr[pointer_start] = p1-p2
            
            #Increment pointers for next iteration
            pointer_start = pointer_end + 1
            pointer_end = pointer_end + window_size 

        #find maximum in difference array 
        germ_time = diffarr.argmax() * timeadj + time_offset

        #leaving this commented out. Just keeping in case I decide to use the difference array for some reason.
        #output = (timepoint,timeseries[diffarr.argmax()], diff)
        return germ_time




class DataVis:
    '''Class contains a series of basic methods to interactivley visualize germination data with Plotly'''
    
    def __init__(self, dataframe):
        self.data = dataframe

    def plot_subset(self,num_samples=1, key='mean_intensity'):
        random_samples = self.subsetter(num_samples=num_samples)
        return self.plot_curves(random_samples, key=key)
        

    #Revise subsetter function to reflect function in newer version
    def subsetter(self, num_samples=1):
        #This function absolutely needs to  be sped up.
        #find unique IDs based on group. Will need to add treatment to this if comparing treatments
        unique = self.data.groupby(['Strain','Position','Replicate'])['ID'].apply(np.unique)
        #loop though the multiindex to find the mutliindex and take the number of smaples desired
        samples = [[index, np.unique(np.random.choice(sample, num_samples))] for index,sample in unique.items()]
        #initialize empy list to concatenate dataframes at the end of the loop
        rand_samps = []
        #Change the dataframe to a multiindexed dataframe to use the samples as 
        multiindexed = self.data.copy().set_index(['Strain','Position','Replicate'])

        for identifier, spore_ids in samples:
            #get indexes in full dataframe
            indexes = multiindexed.sort_index().loc[identifier]
            #use those indices to return a dataframe with random values
            output = indexes[indexes['ID'].isin(spore_ids)].reset_index()
            #append to random sample list
            rand_samps.append(output)

        #return dataframe with subsets for everything
        rand_samps = pd.concat(rand_samps).reset_index()
    
        return rand_samps

    def sort_plot_data(self, df, order, key = 'Condition'):
        #Function converts conditions as a categorical variable in pandas and allows us to specify a specific plotting order.
        df[key] = df[key].astype(order)
        df.sort_values(key, inplace=True, ascending=False)
        return df

    def plot_curves(self, data, key):
        labels = {'Time': 'Time (min)'}

        if 'unique_id' in data.columns:
            color = 'unique_id'
        else:
            color = 'ID'

        fig = px.line(data, x = 'Time', y = key, color=color, facet_row='Condition',
                labels=labels)

        fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[1]))
        fig.update_layout(width=800, showlegend=False)
        return fig

    


#Standalone functions--------------------------------------------------------------------------------------------------
def save_dataframe(df, outputdir, outputname):
            #save huge dataframe to whatever directory the user specifies.
    output_path = os.path.join(outputdir,f'{outputname}_joined.csv')
    df.to_csv(output_path, index=False)

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
        scale_percent = 60 # percent of original size
        width = int(im_copy.shape[1] * scale_percent / 100)
        height = int(im_copy.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized = cv2.resize(im_copy, dim) 
        cv2.imshow('Contours', resized) 
    else:  
        cv2.imshow('Contours', im_copy) 

    # cv2.imshow('Contours', im_copy) 
    #https://stackoverflow.com/questions/35003476/opencv-python-how-to-detect-if-a-window-is-closed/37881722#37881722
    #Display segmentation result. Code below closes the open CV window using the 'x'. Otherwise it will eb 
    while cv2.getWindowProperty('window-name', 0) >= 0:
        keyCode = cv2.waitKey(50)

    return seg

def display_image(image):
#Trying to speed things up using openCV for image display.
    #resize the image if very large
    if image.shape[1] > 1400 or image.shape[0] > 1400:
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
    return MultiImage(image_directory)[0]

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