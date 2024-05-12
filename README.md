# PyThermalcam
Python Software to use the Topdon TC001 Thermal Camera on Python.
It **may** work with other similar cameras.

Huge kudos to LeoDJ on the EEVBlog forum for reverse engineering the image format from these kind of cameras (InfiRay P2 Pro) to get the raw temperature data!
https://www.eevblog.com/forum/thermal-imaging/infiray-and-their-p2-pro-discussion/200/
Check out Leo's Github here: https://github.com/LeoDJ/P2Pro-Viewer/tree/main


**NOTE NOTE NOTE: This is a heavy modified version of leswright1977/PyThermalCamera
 It has different features and different key bindings!**


## Introduction

This is a quick and dirty Python implimentation of Thermal Camera software for the Topdon TC001!
(https://www.amazon.co.uk/dp/B0BBRBMZ58)
No commands are sent the the camera, instead, we take the raw video feed, do some openCV magic, and display a nice heatmap along with relevant temperature points highlighted.

![Screenshot](media/TC00120230701-131032.png)

This program, and associated information is Open Source (see Licence), but if you have gotten value from these kinds of projects and think they are worth something, please consider donating: https://paypal.me/leslaboratory?locale.x=en_GB 


## Features

The following features have been implemented:

- Bicubic image upscaling
- Image smoothing using blur filter
- Fullscreen and window mode
- thermal color maps
- Variable contrast and brightness
- Center of scene temperature monitoring (crosshairs).
- Floating maximum and minimum temperature values within the scene
- Video recording
- Snapshot image as png

The current settings are displayed in a box at the top left of the screen (the OSD):

- Image scale
- Image filtering (blur)
- Colormap
- Brightness and contrast value
- Scaling multiplier
- Image rotation
- Recording status


## Dependencies

Python3 OpenCV Must be installed:


Run: **sudo apt-get install python3-opencv**

NOTE: On Win32/mingw64, opencv doesn't include the neccessary video import filter.
You may want to install a better opencv.


## Running the Program

In src you will find two programs:

**tc001-RAW.py** Just demonstrates how to grab raw frames from the Thermal Camera, a starting point if you want to code your own app.


**tc001v4.2.py** The main program!

To run it plug in the thermal camera and run: **v4l2-ctl --list-devices** to list the devices on the system. You will need its device number.

Assuming the device number is 0 simply issue: **python3 tc001v4.2.py --device 0**

**Note**
This is in Alpha. No error checking has been implemented yet! So if the program tries to start, then quits, either a camera is not connected, or you have entered the wrong device number.



## Key Bindings

- +  -  : increase/decrease scale and image size
- f     : toggle fullscreen
- m     : toggle menu
- b     : toggle through brightnesss values
- c     : toggle through contrast values
- r     : rotate clockwise
- CTRL+s: print snapshot to png
- 0     : toggle recording
- v     : toggle colormap
- q, ESC: quit program
