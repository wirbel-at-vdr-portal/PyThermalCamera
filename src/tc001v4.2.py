#!/usr/bin/env python3
'''
Les Wright 21 June 2023
https://youtube.com/leslaboratory
A Python program to read, parse and display thermal data from the Topdon TC001 Thermal camera!
'''
print('Les Wright 21 June 2023')
print('https://youtube.com/leslaboratory')
print('A Python program to read, parse and display thermal data from the Topdon TC001 Thermal camera!')
print('')
print('Tested on Debian all features are working correctly')
print('This will work on the Pi However a number of workarounds are implemented!')
print('Seemingly there are bugs in the compiled version of cv2 that ships with the Pi!')
print('')
print('Key Bindings:')
print("+, -  : increase/decrease scale and image size")
print("f     : toggle fullscreen")
print("m     : toggle menu")
print("b     : toggle through brightnesss values")
print("c     : toggle through contrast values")
print("r     : rotate clockwise")
print("CTRL+s: print snapshot to png")
print("0     : toggle recording")
print("v     : toggle colormap")
print("q, ESC: quit program")


import cv2
import numpy as np
import argparse
import time
import io
import platform

font = cv2.FONT_HERSHEY_SIMPLEX

# see lazarus or delphi for naming. BGR, not RGB!
clBlack         = (  0,  0,  0)
clWhite         = (255,255,255)
clRed           = (  0,  0,255)
clBlue          = (255,  0,  0)
clYellow        = (  0,255,255)
clDarkWashedRed = ( 40, 40,255)


#We need to know if we are running on the Pi, because openCV behaves a little oddly on all the builds!
#https://raspberrypi.stackexchange.com/questions/5100/detect-that-a-python-program-is-running-on-the-pi
def is_raspberrypi():
    try:
        with io.open('/sys/firmware/devicetree/base/model', 'r') as m:
            if 'raspberry pi' in m.read().lower(): return True
    except Exception: pass
    return False

isPi = is_raspberrypi()

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=int, default=0, help="Video Device number e.g. 0, use v4l2-ctl --list-devices")
args = parser.parse_args()
   
if args.device:
   dev = args.device
else:
   dev = 0
   
#init video
if platform.system() == 'Windows':
   cap = cv2.VideoCapture(int(dev))
else:
   cap = cv2.VideoCapture('/dev/video'+str(dev), cv2.CAP_V4L)

# pull in the video but do NOT automatically convert to RGB, else it breaks the temperature data!
# https://stackoverflow.com/questions/63108721/opencv-setting-videocap-property-to-cap-prop-convert-rgb-generates-weird-boolean
if isPi == True:
   cap.set(cv2.CAP_PROP_CONVERT_RGB, 0.0)
else:
   try:
      cap.set(cv2.CAP_PROP_CONVERT_RGB, False)
   except TypeError:
      cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)      # on Windows (and maybe other platforms) OpenCV expects 0 instead of False

# General settings
width = 256                 # raw video width
height = 192                # raw video height
scale = 3                   # video upscaling multiplier
newWidth  = width  * scale  # will change
newHeight = height * scale  # will change  
alpha = 1.0                 # gain   or 'contrast', (1.0-3.0)
beta  = 0.0                 # offset or 'brightness', +/- 100 in steps of ten
rad = 0                     # image smoothing filter radius for blur
colormap = 0                # color map index
fullscreen = False          # fullscreeen or windowed
rotate = 0                  # image not rotated, 0..3 clockwise
threshold = 2               # mark points below or above average temp
osd = True                  # show osd
recording = False           # don't record by default
elapsed = "00:00:00"        # recording - elapsed time
snaptime = "None"           # may be removed later
frameCounter = 24;          # initialized to update center, min, may temp


def rec():
   now = time.strftime("%Y%m%d--%H%M%S")
   #do NOT use mp4 here, it is flakey!
   videoOut = cv2.VideoWriter(now+'output.avi', cv2.VideoWriter_fourcc(*'XVID'),25, (newWidth,newHeight))
   return(videoOut)

def ToUint16(t, row, col):
   return t[row][col][0] | (t[row][col][1] << 8)

def calctemp(u):
   return (u/64.0) - 273.15



cv2.namedWindow("TC001",cv2.WINDOW_GUI_NORMAL)
cv2.resizeWindow("TC001", newWidth,newHeight)


while(cap.isOpened()):
   # Capture frame-by-frame
   ret, frame = cap.read()
   if ret == True:
      if platform.system() == 'Windows':               # on Windows, OpenCV outputs the frame in a different format
         frame = np.reshape(frame[0], (192*2, 256, 2)) # if there's a "can't reshape size xxxx into shape ..." error here, you most likely used the wrong --device ID

      imdata,thdata = np.array_split(frame, 2)
      #now parse the data from the bottom frame and convert to temp!
      #https://www.eevblog.com/forum/thermal-imaging/infiray-and-their-p2-pro-discussion/200/
      #Huge props to LeoDJ for figuring out how the data is stored and how to compute temp from it.
      #grab data from the center pixel...

      frameCounter += 1
      if frameCounter >= 12:
         frameCounter = 0

         temp = round(calctemp(ToUint16(thdata, 96, 128)), 1)

         Max = 0
         Min = 65535
         Sum = 0.0
         MinRow = 0
         MinCol = 0
         MaxRow = 0
         MaxCol = 0

         for row in range (0, 192):
            for col in range(0, 256):
               u = ToUint16(thdata, row, col)
               if u < Min:
                  Min = u
                  MinRow = row
                  MinCol = col
               if u > Max:
                  Max = u
                  MaxRow = row
                  MaxCol = col
               Sum += u
         Avg = Sum/(192 * 256);
         maxtemp = round(calctemp(Max), 1)
         mintemp = round(calctemp(Min), 1)
         avgtemp = round(calctemp(Avg), 1)

      # Convert the real image to RGB
      bgr = cv2.cvtColor(imdata,  cv2.COLOR_YUV2BGR_YUYV)

      # rotate = 0..3 -> 0 .. 270
      # -> rotating image requires resizeWindow, otherwise the image would be distorted
      if rotate > 0:
         bgr = cv2.rotate(bgr, rotate-1);

      # brightness and contrast
      bgr = cv2.convertScaleAbs(bgr, alpha=alpha, beta=beta)

      #bicubic interpolate, upscale and blur
      bgr = cv2.resize(bgr,(newWidth,newHeight),interpolation=cv2.INTER_CUBIC)#Scale up!
      if rad>0:
         bgr = cv2.blur(bgr,(rad,rad))

      #apply colormap
      match(colormap):
         case 0:
            heatmap = cv2.applyColorMap(bgr, cv2.COLORMAP_JET)
            cmapText = 'Jet'
         case 1:
            heatmap = cv2.applyColorMap(bgr, cv2.COLORMAP_HOT)
            cmapText = 'Hot'
         case 2:
            heatmap = cv2.applyColorMap(bgr, cv2.COLORMAP_MAGMA)
            cmapText = 'Magma'
         case 3:
            heatmap = cv2.applyColorMap(bgr, cv2.COLORMAP_INFERNO)
            cmapText = 'Inferno'
         case 4:
            heatmap = cv2.applyColorMap(bgr, cv2.COLORMAP_PLASMA)
            cmapText = 'Plasma'
         case 5:
            heatmap = cv2.applyColorMap(bgr, cv2.COLORMAP_BONE)
            cmapText = 'Bone'
         case 6:
            heatmap = cv2.applyColorMap(bgr, cv2.COLORMAP_SPRING)
            cmapText = 'Spring'
         case 7:
            heatmap = cv2.applyColorMap(bgr, cv2.COLORMAP_AUTUMN)
            cmapText = 'Autumn'
         case 8:
            heatmap = cv2.applyColorMap(bgr, cv2.COLORMAP_VIRIDIS)
            cmapText = 'Viridis'
         case 9:
            heatmap = cv2.applyColorMap(bgr, cv2.COLORMAP_PARULA)
            cmapText = 'Parula'
         case 10:
            heatmap = cv2.applyColorMap(bgr, cv2.COLORMAP_RAINBOW)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            cmapText = 'Inv Rainbow'

      # draw crosshairs
      cv2.line(heatmap,(int(newWidth/2),int(newHeight/2)+20),       (int(newWidth/2)   ,int(newHeight/2)-20), clWhite, 2) #vline
      cv2.line(heatmap,(int(newWidth/2),int(newHeight/2)+20),       (int(newWidth/2)   ,int(newHeight/2)-20), clBlack, 1) #vline      
      cv2.line(heatmap,(int(newWidth/2)+20,int(newHeight/2)),       (int(newWidth/2)-20,int(newHeight/2))   , clWhite, 2) #hline
      cv2.line(heatmap,(int(newWidth/2)+20,int(newHeight/2)),       (int(newWidth/2)-20,int(newHeight/2))   , clBlack, 1) #hline
      #show temp
      cv2.putText(heatmap,str(temp)+' C', (int(newWidth/2)+10, int(newHeight/2)-10), font, 0.45, clBlack , 2, cv2.LINE_AA)
      cv2.putText(heatmap,str(temp)+' C', (int(newWidth/2)+10, int(newHeight/2)-10), font, 0.45, clYellow, 1, cv2.LINE_AA)
    
      #Yeah, this looks like we can probably do this next bit more efficiently!
      #display floating max temp
      if maxtemp > avgtemp+threshold:
         match(rotate):
            case 0:
               x = MaxCol
               y = MaxRow
            case 1:
               x = width - MaxRow
               y = MaxCol
            case 2:
               x = width - MaxCol
               y = height - MaxRow
            case 3:
               x = MaxRow
               y = height - MaxCol          
         cv2.circle(heatmap, (x*scale, y*scale), 5, clBlack, 2)
         cv2.circle(heatmap, (x*scale, y*scale), 5, clRed  ,-1)
         cv2.putText(heatmap,str(maxtemp)+' C', ((x*scale)+10, (y*scale)+5), font, 0.45, clBlack , 2, cv2.LINE_AA)
         cv2.putText(heatmap,str(maxtemp)+' C', ((x*scale)+10, (y*scale)+5), font, 0.45, clYellow, 1, cv2.LINE_AA)

      #display floating min temp
      if mintemp < avgtemp-threshold:
         match(rotate):
            case 0:
               x = MinCol
               y = MinRow
            case 1:
               x = width - MinRow
               y = MinCol
            case 2:
               x = width - MinCol
               y = height - MinRow
            case 3:
               x = MinRow
               y = height - MinCol
         cv2.circle(heatmap, (x*scale, y*scale), 5, clBlack, 2)
         cv2.circle(heatmap, (x*scale, y*scale), 5, clBlue ,-1)
         cv2.putText(heatmap,str(mintemp)+' C', ((x*scale)+10, (y*scale)+5), font, 0.45, clBlack , 2, cv2.LINE_AA)
         cv2.putText(heatmap,str(mintemp)+' C', ((x*scale)+10, (y*scale)+5), font, 0.45, clYellow, 1, cv2.LINE_AA)

      if osd==True:
         # display black box for our data
         cv2.rectangle(heatmap, (0, 0),(160, 160), (0,0,0), -1)
         p = 14

         cv2.putText(heatmap,'Scale (+/-): '+str(scale)    , (10, p), font, 0.4, clYellow, 1, cv2.LINE_AA)
         p += 14

         cv2.putText(heatmap,'Blur (s): '+str(rad)         , (10, p), font, 0.4, clYellow, 1, cv2.LINE_AA)
         p += 14

         cv2.putText(heatmap,'Bright (b): '+str(beta)      , (10, p), font, 0.4, clYellow, 1, cv2.LINE_AA)
         p += 14

         cv2.putText(heatmap,'Contrast (c): '+str(alpha)   , (10, p), font, 0.4, clYellow, 1, cv2.LINE_AA)
         p += 14

         cv2.putText(heatmap,'Colormap (v): '+cmapText     , (10, p), font, 0.4, clYellow, 1, cv2.LINE_AA)
         p += 14

         cv2.putText(heatmap,'Rotate (r): '+str(rotate*90) , (10, p), font, 0.4, clYellow, 1, cv2.LINE_AA)
         p += 14

         cv2.putText(heatmap,'Menu (m): '+str(osd)         , (10, p), font, 0.4, clYellow, 1, cv2.LINE_AA)
         p += 14

         cv2.putText(heatmap,'Snapshot (CTRL+s)'           , (10, p), font, 0.4, clYellow, 1, cv2.LINE_AA)
         p += 14

         cv2.putText(heatmap,'Close (ESC or q)'            , (10, p), font, 0.4, clYellow, 1, cv2.LINE_AA)
         p += 14

         if recording:
            cv2.putText(heatmap,'Recording: '+elapsed         , (10, p), font, 0.4, clDarkWashedRed, 1, cv2.LINE_AA)
            p += 14

      #display image
      cv2.imshow("TC001",heatmap)

      if recording == True:
         elapsed = (time.time() - start)
         elapsed = time.strftime("%H:%M:%S", time.gmtime(elapsed)) 
         #print(elapsed)
         videoOut.write(heatmap)

      keyPress = cv2.waitKey(1)
      if keyPress < 0:
         continue
      keyPress = chr(keyPress)
      KEY_ESCAPE = chr(27)
      match(keyPress):
         case '+':        # increase scale
            scale = min(6, scale + 1)
            newWidth  = width  * scale 
            newHeight = height * scale
            if not fullscreen and not isPi:
               cv2.resizeWindow("TC001", newWidth, newHeight)
         case '-':        # decrease scale
            scale = max(1, scale - 1)
            newWidth  = width  * scale 
            newHeight = height * scale
            if not fullscreen and not isPi:
               cv2.resizeWindow("TC001", newWidth, newHeight)
         case 'f':        # toggle fullscreen
            fullscreen = not fullscreen
            if fullscreen:
               cv2.setWindowProperty("TC001", cv2.WND_PROP_FULLSCREEN, 1.0);
            else:
               cv2.setWindowProperty("TC001", cv2.WND_PROP_FULLSCREEN, 0.0);
               cv2.resizeWindow("TC001", newWidth, newHeight);
         case 'm':        # toggle menu
            osd = not osd
         case 'b':        # toggle through brightnesss values
            beta += 10.0
            beta = round(beta,0)
            if beta > 100.01:
               beta = -100
         case 'c':        # toggle through contrast values
            alpha += 0.1
            alpha = round(alpha,1)#fix round error
            if alpha >= 3.01:
               alpha = 0.0
         case 'r':        # rotate clockwise
            rotate += 1
            if rotate > 3:
               rotate = 0
            tmp = width
            width = height
            height = tmp
            newWidth  = width  * scale
            newHeight = height * scale
            if not fullscreen:
               cv2.resizeWindow("TC001", newWidth, newHeight);
         case 's':        # toggle through image smoothing radius values
            rad += 1
            if rad > 5:
               rad = 0
         case '0':
            recording = not recording
            if recording:
               videoOut = rec()
               start = time.time()
            else:
               elapsed = "00:00:00"
         case 'v':
            colormap = (colormap + 1) % 11
         case 'q':        # exit program
            break
         case KEY_ESCAPE: # exit program
            break

cap.release()
cv2.destroyAllWindows()
      
