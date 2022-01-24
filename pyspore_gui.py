import itertools
import time, os, sys, cv2
import pandas as pd
import pyspore_utils as psp
from pyspore_utils import Pipeline
from PIL import Image, ImageTk 
import PySimpleGUI as sg
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.widgets import RectangleSelector
import numpy as np
import glob, re


def image_processing_gui():
    #simple image processing interface with pysimplegui
    sg.theme('DarkBlue')
    
    vars = {'window': False,
             'fig_agg': False,
             'plt_fig': False,
             'images': False}
    
    #Construct layout
    column_1 = [[sg.Text('Directory')], 
                [sg.In(size=(25,1), enable_events=True ,key='-FOLDER-'), sg.FolderBrowse()],
                [sg.Listbox(values=[], enable_events=True, size=(30,20),key='-FILE LIST-', select_mode='extended')],
                [sg.Button('Load Image')],
                [sg.Button('Crop', visible = False)],
                [sg.Button('Test Segmentation')],
                [sg.Text('Threshold', key = 'Adjust Threshold'), sg.InputText(0.05, size=(5,1), key = 'thresh')],
                [sg.Checkbox('Align Images', default=False, enable_events=True, key = '-REG-')],
                [sg.Button('Process'), sg.Button('Batch Process')],
                [sg.Exit()],
                [sg.pin(sg.Text('Frame', key = 'Frame', visible = False)), 
                sg.pin(sg.Slider(range = (None,None), key = '-SLIDER-', 
                orientation = 'horizontal', visible = False, enable_events = True))]]
                    
    
    # column_2 = [[sg.Canvas(key='-CANVAS-')],
    #             [sg.pin(sg.Text('Frame', key = 'Frame', visible = False)), 
    #             sg.pin(sg.Slider(range = (None,None), key = '-SLIDER-', 
    #             orientation = 'horizontal', visible = False, enable_events = True))]]
    
    #layout = [[sg.Column(column_1), sg.VSeperator(), sg.Column(column_2)]]
    layout = [[sg.Column(column_1)]]
    #initialize window
    vars['window'] = sg.Window("Image Viewer", layout, font='Helvetica 12', no_titlebar = True, grab_anywhere=True, resizable=True)
    
    #event loop
    while True:
        event, values = vars['window'].read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        
        if event == '-FOLDER-':
            #find all .tif files in the selected directory
            names = [os.path.basename(x) for x in glob.glob(os.path.join(values['-FOLDER-'],'*.tif'))]
            #display images in the listbox
            vars['window']['-FILE LIST-'].update(names)
            #window.Element('_LISTBOX_').Update(select_mode='multiple')
        
        if event == 'Load Image':
            image_path = os.path.join(values['-FOLDER-'], values['-FILE LIST-'][0])
            image = psp.load_images(image_path)
            vars['images'] = image[0]

            if vars['fig_agg'] is not False:
                #delete_fig_agg(vars['fig_agg'])
                psp.display_image(vars['images'])
            else:
                psp.display_image(vars['images'])
            

            if len(image) > 1:
                #make frame slider appear if images are multipage (movies)
                vars['window']['-SLIDER-'].update(visible = True)
                vars['window']['Frame'].update(visible = True)
                vars['window']['-SLIDER-'].update(range = (1,len(image)))
       

        if event == '-SLIDER-':
            #use slider to access other images in the stack
            #use of openCV significantly improved performance of this.
            img_idx = int(values['-SLIDER-']-1)
            vars['images'] = image[img_idx]
            psp.display_image(vars['images'])

        if event == 'Test Segmentation':
            vars['window']['thresh'].update(visible = True)
            vars['window']['Adjust Threshold'].update(visible = True)
            threshold = float(values['thresh'])

            psp.test_seg(path = image_path, image = vars['images'], threshold = threshold)

        if event == '-REG-':
            vars['align'] = True
        else:
            vars['align'] = False

        if event == 'Process':
            image_path = os.path.join(values['-FOLDER-'], values['-FILE LIST-'][0])
            print(values['-FILE LIST-'])
            vars['window']['thresh'].update()
            threshold = float(values['thresh'])
            #popup window to load experiment data (package this in a metadata file, in case user wants to re-analyze later)
            ev, exp_data = enter_expt_data(values['-FILE LIST-'])

        
            if ev == 'Exit':
                #Instantiate pipeline object with current images
                expt = Pipeline(paths = image_path, threshold = threshold, exp_data = exp_data, align = vars['align'])
                expt.process()
            

        if event == 'Batch Process':
            #processes multiple selections from the list box.
            vars['window']['thresh'].update()
            threshold = float(values['thresh'])
            print(threshold)
            names = values['-FILE LIST-']

            batch_paths = [os.path.join(values['-FOLDER-'], name) for name in names]
            ev, exp_data = enter_expt_data(values['-FILE LIST-'])
            
            if ev == 'Exit':
                #have console window popup with progress data etc.
                batch = Pipeline(paths = batch_paths, threshold = threshold , exp_data = exp_data, batch = True, align = vars['align'])
                batch.process()   

    vars['window'].close()


def enter_expt_data(filenames):

    #Initialize dictionary for experiment data (output)
    exp = {name: {'strain': None, 'replicate': None, 'position': None} for i,name in enumerate(filenames)}

    #Layout is created dynamically depending on the number of image files 
    inputs = [[sg.Text(name), sg.Input('Strain Name', key = f'strain_{i}', size = (17,1), focus = True), 
    sg.Input('Replicate Number', key = f'replicate_{i}', size = (17,1)), 
    sg.Input('Position Number', key = f'position_{i}', size = (17,1))] for i,name in enumerate(filenames)]

    #define window layout
    layout = [inputs, [sg.Button('Save Experiment Data', key = 'Save'), sg.Exit()]]
    #initialize window
    window = sg.Window('Experiment Data', layout, element_justification='r')
   
    #event loop
    while True:
        event,values = window.read()
        if event == 'Save':
            #Once submit is hit, populate the exp dictionary with the values from the text input
            #Start by looping over image names (im)
            for i, im in enumerate(exp):
                #extract list of user inputs from dict ['630',1,1]
                res = [val for key, val in values.items() if f'{i}' in key]
                #use zip to replace the values in the sub-dictionaries with the res values
                exp[im] = dict(zip(exp[im], res))
                #output will look something like this:  {filename.tif: {'strain': 630, 'replicate': 1, 'position': 1}}
                #let user know that the experiment data was updated. User can do this as many times as they want.
                sg.popup_quick_message('Successfully Updated Experiment Data')

        if event == sg.WIN_CLOSED or event == 'Exit':    
            break
   
    window.close()
    return  event, exp

    

def data_analysis_gui():
    pass


if __name__ == "__main__":
    image_processing_gui()
