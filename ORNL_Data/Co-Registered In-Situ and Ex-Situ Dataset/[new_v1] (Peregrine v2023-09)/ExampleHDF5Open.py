# Date: 2023-06-27
# Description: example script to load an exported HDF5 file
#-----------------------------------------------------------------------------

# Load external modules
import h5py
import matplotlib.pyplot as plt
import numpy as np

# Needed to enable plotting if using the Spyder IDE
try:
    from IPython import get_ipython # needed to run magic commands
    ipython = get_ipython() # needed to run magic commands
    ipython.magic('matplotlib qt') # display figures in a separate window
except: pass

# Parameters
path_file = 'D:/Temp/HDF5 Exports/2018-12-31 037-R1256_BLADE ITERATION 13_5.hdf5' # path to your HDF5 file
probe_layer = 50 # layer number to view

#-----------------------------------------------------------------------------

# Check the HDF5 file
print('\nChecking the HDF5 contents...\n')
with h5py.File(path_file,'r') as build:
    
    def print_attrs(name,obj):
        
        # Access datasets
        if(isinstance(build[name],h5py.Dataset)):
            
            # Report dataset path
            if(name.split('/')[0]!='scans'):  print('--',name)
            
            # Display slice-type data
            if(name.split('/')[0]=='slices'):
                plt.figure(name)
                plt.imshow(build[name][probe_layer,...],cmap='jet',interpolation='none')
                
            # Display temporal data
            elif(name.split('/')[0]=='temporal'):
                plt.figure(name)
                plt.scatter(np.arange(build[name].shape[0]),build[name])
                
            # Display part and sample data
            elif((name.split('/')[0]=='parts')|(name.split('/')[0]=='samples')):
                for i in range(1,min(build[name].shape[0],10),1):
                    print('  %i ' %(i),build[name][i])
                    
            # Display reference images and micrographs
            elif((name.split('/')[0]=='reference_images')|(name.split('/')[0]=='micrographs')):
                plt.figure(name)
                plt.imshow(build[name][...],interpolation='none')
        
        # Access attributes
        for key in obj.attrs:
            print('%s/%s:' %(name,key),str(build[name].attrs[key]).split('\n')[0])
    
    # Walk through the HDF5 file and display a sub-set of the data
    print('\n---HDF5 CONTENTS---\n')
    for key in build.attrs: print('%s:' %(key),str(build.attrs[key]).split('\n')[0]) # top-level metadata
    build.visititems(print_attrs) # nested levels