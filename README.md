# PySpore

PySpore is a simple python application to track single bacterial endospore germination and to process and visualize the output. The code functions best in Windows 10/11. Usage on mac will result in GUIs that do not scale properly for displays of different sizes. A demo movie showing how to use the app and example data can be found [here](https://www.dropbox.com/sh/uolszoie5hjx8mo/AACPZUDvA2R86PbXEnpSWsDda?dl=0).

## Project Status
This program was developed as part of my Ph.D. and I have since moved on to another  position, leaving me with limited opportunities to develop PySpore further. I have tried to make the underlying source modular for this to be integrated into projects as needed.

## Installation

1)	Download the anaconda python distribution https://www.anaconda.com/products/distribution 

2)	Download the .zip folder containing code from https://github.com/jribis89/PySpore and extract all files to your desired location. All files must be kept within the same directory.

3)	Launch Powershell or the Anaconda Powershell Prompt and change the directory to where PySpore is located.
```Powershell
cd disk:\directory\PySpore
```

4)	Install dependencies by installing the included environment file.  
```Powershell
(base) conda env create -f pyspore_env.yml
```

5)	Activate the environment.
```Powershell
(base) conda activate pyspore_env
```

## Usage
Launch the main GUI using Powershell or Anaconda Powershell Prompt. Make sure the pyspore_env environment is active and your working directory is the PySpore directory (see step 3 of install above). A splash screen will be displayed and a launcher will pop up.
```Powershell
(pyspore_env) python "pyspore_gui.py"
```
From the launcher, you can open the image processing GUI. From here you will load images using the browse button. 
- To optimize thresholding, select a single image and hit the "test segmentation" button. You will be presented with an image with the spore outlines highlighted. If small objects other than spores are highlighted, increase the value, else decrease it. Use small increments of 0.01 and hit test segmentation again until you are satisfied with the results. Test this on a frame in the middle of the movie and the end and adjust accordingly to get good segmentation.
- Select all images and click "batch process". For a single image, select "process".
- Note: If there is significant drift between frames, check the "align images" box to run a registration algorithm on the movie. Otherwise the single-particle tracking algorithm that is always applied is sufficient.

![pysporeGUIs](https://github.com/jribis89/PySpore/assets/91898442/bbdf97bb-80ad-4af1-977c-9db6ed551a39)

The Data Explorer GUI is aimed to provide a quick means for the user to join and clean up their datasets and make some basic plots.

- Here you can load datasets for images and apply filtering parameters along with pixel size and time conversions. A tooltip explaining each filter will appear when the box to input each value is hovered over.
- When working with 2 or more datasets, you will need to click "process data" and apply the filters. This will join the datasets and apply unique identifiers to each spore tracked in the dataset. If unique IDs are not assigned for each spore, the joined datasets will not plot correctly.
-To save the filtered/joined data just click "Save Data"
- For an example of more complex analysis of the output refer to the included [Jupyter Notebook](https://github.com/jribis89/PySpore/blob/main/2022_Ribis_etal_calcium_manuscript.ipynb). It includes examples of data smoothing, germination rate calculation, plotting, etc... The data output is a .csv, so more complex analysis is easily done in statistical software including R, GraphPad Prism, and Microsoft Excel.

## License

[MIT](https://choosealicense.com/licenses/mit/)
