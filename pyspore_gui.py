'''Code to run PySpore GUIs'''


import os
import pandas as pd
import pyspore_utils as psp
import PySimpleGUI as sg
import glob
from ctypes import windll
#Make GUI DPI aware, producing much better looking widgets across all displays.
windll.shcore.SetProcessDpiAwareness(1)



def image_processing_gui():
    #simple image processing interface with pysimplegui
    sg.theme('DarkBlue')
    
    tooltips = ['Test the threshold on the currently displayed image.', 
                'Select a threshold value. Value should typically be around 0.05-0.15 \n Increase this value if background junk is segmented. Decrease value if no spores are detected.']

    vars = {'window': False,
             'fig_agg': False,
             'plt_fig': False,
             'images': False}
    
    
    #Construct layout
    column_1 = [[sg.Text('Directory')], 
                [sg.In(size=(25,1), enable_events=True ,key='-FOLDER-'), sg.FolderBrowse()],
                [sg.Listbox(values=[], enable_events=True, size=(30,20),key='-FILE LIST-', select_mode='extended', horizontal_scroll=True)],
                [sg.Button('Load Image')],
                [sg.Button('Test Segmentation', tooltip=tooltips[0])],
                [sg.Text('Threshold', key = 'Adjust Threshold'), sg.InputText(0.05, size=(5,1), key = 'thresh', tooltip=tooltips[1])],
                [sg.Checkbox('Align Images', default=False, enable_events=True, key = '-REG-')],
                [sg.Button('Process'), sg.Button('Batch Process')],
                [sg.Exit()],
                [sg.pin(sg.Text('Frame', key = 'Frame', visible = False)), 
                sg.pin(sg.Slider(range = (None,None), key = '-SLIDER-', 
                orientation = 'horizontal', visible = False, enable_events = True))]]
                    
    layout = [[sg.Column(column_1)]]
    #initialize window
    vars['window'] = sg.Window("Image Viewer", layout, font='Helvetica 10', no_titlebar = True, grab_anywhere=True, resizable=True)
    
    #event loop
    while True:
        event, values = vars['window'].read(timeout=100)
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        
        if event == '-FOLDER-':
            #find all .tif files in the selected directory
            names = [os.path.basename(x) for x in glob.glob(os.path.join(values['-FOLDER-'],'*.tif'))]
            #display images in the listbox
            vars['window']['-FILE LIST-'].update(names)
        
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


        if event == 'Process':
            image_path = os.path.join(values['-FOLDER-'], values['-FILE LIST-'][0])
            print(values['-FILE LIST-'])
            vars['window']['thresh'].update()
            threshold = float(values['thresh'])
            #popup window to load experiment data (package this in a metadata file, in case user wants to re-analyze later)
            ev, exp_data = enter_expt_data(values['-FILE LIST-'])

        
            if ev == 'Exit':
                #Instantiate pipeline object with current images
                expt = psp.Pipeline(paths = image_path, threshold = threshold, exp_data = exp_data, align = values['-REG-'])
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
                batch = psp.Pipeline(paths = batch_paths, threshold = threshold , exp_data = exp_data, batch = True, align = values['-REG-'])
                batch.process()

    vars['window'].close()


def enter_expt_data(filenames):
    #Function generates another window for the user to enter thier 
    #Initialize dictionary for experiment data (output)
    exp = {name: {'strain': None, 'replicate': None, 'condition': None, 'position': None} for i,name in enumerate(filenames)}

    #Layout is created dynamically depending on the number of image files 
    inputs = [[sg.Text(name), sg.Input('Strain Name', key = f'strain_{i}', size = (17,1), focus = True), 
    sg.Input('Replicate Number', key = f'replicate_{i}', size = (17,1)), sg.Input('Condition', key = f'condition_{i}', size = (17,1)), 
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
                #let user know that the experiment data was updated. User can do this as many times as they want if they mess up something in the input.
                sg.popup_quick_message('Successfully Updated Experiment Data')

        if event == sg.WIN_CLOSED or event == 'Exit':    
            break
   
    window.close()
    return  event, exp


'''Functions below are for the multiwindow data analysis GUI. Window 1 contains all of the stuff to join, clean, and plot data. The second window is just made to 
display the .csv as a table.'''

#Tooltips list
tooltips = [
    'Use this to look at a single unprocessed dataframe, or a processed dataframe.\n If multiple dataframes are selected, they will automatically be processed according to the parameters listed below.',
    'Cleans up data based on parameters listed below. \n If multiple datasets are selected, they will be joined together and processed.',
    'Displays currently loaded data as a table in new window.\n Will be slow with very large datasets.',
    'Time interval (in seconds) frame-frame.',
    'Delay (in mins) before movie started.',
    'Minimum number of frames spores need to be tracked.',
    'Maximum allowable spore area. \n Avoids selecting clumps.',
    'Pixel size (in μm) of image.',
    'Plots intensity change for a random subset of spores from each position.',
    'Number of random spores to select from each position.',
    'Plots overlayed histograms and marginal box plots.'
]

#Change these defaults to suit experiment
filterdefaults = {'sampint': 30, 
                'timeoffset': 6.5,
                'minframes': 100,
                'maxarea': 400,
                'pixsize': 0.103}

def gen_window_1():
    #Define GUI layout.
    print = sg.Print
    layout = [[sg.Text('Select Data')], 
                [sg.In(size=(25,1), enable_events=True ,key='-FOLDER-'), sg.FolderBrowse()],
                [sg.Listbox(values=[], enable_events=True, size=(33,10),key='-FILE LIST-', select_mode='extended', horizontal_scroll=True)],
                [sg.Button('Load Data', tooltip=tooltips[0]), 
                sg.Button('Process Data', tooltip=tooltips[1]), 
                sg.Button('Display Table', tooltip=tooltips[2])],
                [sg.Text('Sampling Interval (sec)', expand_x=True), sg.InputText(filterdefaults['sampint'], size=(5,1), key='timeint', tooltip=tooltips[3])], 
                [sg.Text('Time Offset (min)',expand_x=True), sg.InputText(filterdefaults['timeoffset'], size=(5,1), key = 'offset', tooltip=tooltips[4])],
                [sg.Text('Min Number Frames',expand_x=True), sg.InputText(filterdefaults['minframes'], size=(5,1), key = 'minframes', tooltip=tooltips[5])], 
                [sg.Text('Maximum Spore Area (px)', expand_x=True), sg.InputText(filterdefaults['maxarea'], size=(5,1), key = 'maxarea', tooltip=tooltips[6])],
                [sg.Text('Pixel Size (μm)', expand_x=True), sg.InputText(filterdefaults['pixsize'], size=(5,1), key = 'pixsize', tooltip=tooltips[7])],
                [sg.Button('Plot Random Traces', tooltip=tooltips[8]), sg.Combo(values=[], size=(15,1), visible=False, key='trace_var')], 
                [sg.Text('Num Samples'), sg.InputText(1, size=(5,1), key = 'sample', tooltip=tooltips[9])],
                [sg.Button('Plot Distribution', tooltip=tooltips[10]), sg.Combo(values=['Time to germination','Max Germination Rate'], default_value='Time to germination', key='plotsel')],
                [sg.Button('Save Data'), sg.Button('Save Plots')],
                [sg.Exit()]]

    return sg.Window('Data Exploration', layout, font='Helvetica 10', no_titlebar = True, grab_anywhere=True, resizable=True, finalize=True)

def gen_window_2(table_data):
    '''Window displays table with data. The '''
    headings = table_data.columns.values.tolist()
    values = table_data.to_numpy()

    layout = [[sg.Table(values=values, headings=headings, vertical_scroll_only=False)],
             [sg.Exit()]]
    if len(values) > 0:
        return sg.Window('Data', layout,  grab_anywhere=True, resizable=True, finalize=True)
    else:
        sg.Print('Number of rows in table < 1, Check filtering values.')

def data_analysis_gui():
    '''GUI is for very basic data exploration and visualization.'''
    #Construct layout
    #start with one window open.
    win1, win2= gen_window_1(), None
    
    
    #event loop
    while True:
        window, event, values = sg.read_all_windows(timeout=100)

        if event == sg.WIN_CLOSED or event == 'Exit':
            if window == win2:
                #Close the second window but not the first
                win2.close()
            elif window == win1: 
                break    
        
        #functional events
        if event == '-FOLDER-':
            #find all .csv files in the selected directory
            names = [os.path.basename(x) for x in glob.glob(os.path.join(values['-FOLDER-'],'*.csv'))]
            #display list of files in the listbox
            window['-FILE LIST-'].update(names)


        if event == 'Load Data':
            if len(values['-FILE LIST-']) >= 1:
            #if loading data for the first time the user can process the data to clean it up. If reloading, just calling the basic pandas read csv method
                files  = [os.path.join(values['-FOLDER-'], file) for file in values['-FILE LIST-']]
                table = pd.read_csv(files[0]) 
                sg.popup_quick_message('.csv Data Loaded successfully', background_color='dark blue')
                window['trace_var'].update(values=table.columns.tolist(), visible=True)
            else:
                sg.popup_quick_message('No file selected from list!', background_color='red')
        

        if event == 'Process Data':
            sg.Print('Processing data', files)
            table = psp.DataHandling(files, timeint=float(values['timeint']), 
                time_offset=float(values['offset']), min_frames=float(values['minframes']), max_area=float(values['maxarea'])).process_data()
            sg.Print('Done Processing')
            window['trace_var'].update(values=table.columns.tolist(), visible=True)

        if event == 'Plot Random Traces':
            window['sample'].update()
            #Update user selection of which measurement to plot.
            window['trace_var'].update()
            plot = psp.DataVis(dataframe=table).plot_subset(num_samples=int(values['sample']), key=values['trace_var'])
            plot.show()

    
        if event == 'Display Table':
            #can get very slow for massive dataframes. 
            win2 = gen_window_2(table)

        if event == 'Save Data':
            #Need to have another save data button for the other datasets
            #open up a save dialog. User can select the name and the ouput directory.
            outputpath = sg.popup_get_file(message = 'Save Path', default_path=values['-FOLDER-'], save_as=True,
            default_extension='.csv')
            #use the built in pandas method to save as .csv
            table.to_csv(outputpath,index=False)


    window.close()

def psp_launcher():
    #Display Splash Screen from Pyspore Directory
    splash_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),'static_splash_v1.png')
    splash_timeout_ms = 1000

    sg.Window('Splash', layout=[[sg.Image(splash_dir)]], no_titlebar=True,
    keep_on_top=True,transparent_color=sg.theme_background_color()).read(timeout=splash_timeout_ms, close=True)

    #launch simple popup that will allow users to launch either gui
    layout = [[sg.Text('Welcome to PySpore!')],
              [sg.Button('Launch Image Processing')],
              [sg.Button('Launch Data Explorer')],
              [sg.Exit()]]
    #initialize window
    window = sg.Window('PySpore Launcher', layout, element_justification='c',no_titlebar=True, font='Helvetica 11',
    grab_anywhere=True)

    #event loop
    while True:
        event,vals = window.read()
        if event == sg.WIN_CLOSED or event == 'Exit':    
            break
        if event == 'Launch Image Processing':
            image_processing_gui()
        elif event == 'Launch Data Explorer':
            data_analysis_gui()
    
    window.close()
    

if __name__ == "__main__":
    psp_launcher()
