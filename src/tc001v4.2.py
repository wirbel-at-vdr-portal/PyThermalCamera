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
#cap = cv2.VideoCapture(0)
#pull in the video but do NOT automatically convert to RGB, else it breaks the temperature data!
#https://stackoverflow.com/questions/63108721/opencv-setting-videocap-property-to-cap-prop-convert-rgb-generates-weird-boolean
if isPi == True:
   cap.set(cv2.CAP_PROP_CONVERT_RGB, 0.0)
else:
   try:
      cap.set(cv2.CAP_PROP_CONVERT_RGB, False)
   except TypeError:
      cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)      # on Windows (and maybe other platforms) OpenCV expects 0 instead of False

#256x192 General settings
width = 256 #Sensor width
height = 192 #sensor height
scale = 3 #scale multiplier
newWidth = width*scale 
newHeight = height*scale
alpha = 1.0               # gain   or 'contrast', (1.0-3.0)
beta  = 0.0               # offset or 'brightness', +/- 100 in steps of ten
colormap = 0
font=cv2.FONT_HERSHEY_SIMPLEX
fullscreen = False
rotate = 0                # image not rotated, 0..3 clockwise
cv2.namedWindow('Thermal',cv2.WINDOW_GUI_NORMAL)
cv2.resizeWindow('Thermal', newWidth,newHeight)
rad = 0 #blur radius
threshold = 2
osd = True
recording = False
elapsed = "00:00:00"
snaptime = "None"

def rec():
   now = time.strftime("%Y%m%d--%H%M%S")
   #do NOT use mp4 here, it is flakey!
   videoOut = cv2.VideoWriter(now+'output.avi', cv2.VideoWriter_fourcc(*'XVID'),25, (newWidth,newHeight))
   return(videoOut)

def snapshot(heatmap):
   #I would put colons in here, but it Win throws a fit if you try and open them!
   now = time.strftime("%Y%m%d-%H%M%S") 
   snaptime = time.strftime("%H:%M:%S")
   cv2.imwrite("TC001"+now+".png", heatmap)
   return snaptime
 

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
      hi = thdata[96][128][0]
      lo = thdata[96][128][1]
      #print(hi,lo)
      lo = lo*256
      rawtemp = hi+lo
      #print(rawtemp)
      temp = (rawtemp/64)-273.15
      temp = round(temp,2)
      #print(temp)
      #break

      #find the max temperature in the frame
      lomax = thdata[...,1].max()
      posmax = thdata[...,1].argmax()
      #since argmax returns a linear index, convert back to row and col
      mcol,mrow = divmod(posmax,width)
      himax = thdata[mcol][mrow][0]
      lomax=lomax*256
      maxtemp = himax+lomax
      maxtemp = (maxtemp/64)-273.15
      maxtemp = round(maxtemp,2)

      
      #find the lowest temperature in the frame
      lomin = thdata[...,1].min()
      posmin = thdata[...,1].argmin()
      #since argmax returns a linear index, convert back to row and col
      lcol,lrow = divmod(posmin,width)
      himin = thdata[lcol][lrow][0]
      lomin=lomin*256
      mintemp = himin+lomin
      mintemp = (mintemp/64)-273.15
      mintemp = round(mintemp,2)

      #find the average temperature in the frame
      loavg = thdata[...,1].mean()
      hiavg = thdata[...,0].mean()
      loavg=loavg*256
      avgtemp = loavg+hiavg
      avgtemp = (avgtemp/64)-273.15
      avgtemp = round(avgtemp,2)

      

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
      cv2.line(heatmap,(int(newWidth/2),int(newHeight/2)+20),\
      (int(newWidth/2),int(newHeight/2)-20),(255,255,255),2) #vline
      cv2.line(heatmap,(int(newWidth/2)+20,int(newHeight/2)),\
      (int(newWidth/2)-20,int(newHeight/2)),(255,255,255),2) #hline

      cv2.line(heatmap,(int(newWidth/2),int(newHeight/2)+20),\
      (int(newWidth/2),int(newHeight/2)-20),(0,0,0),1) #vline
      cv2.line(heatmap,(int(newWidth/2)+20,int(newHeight/2)),\
      (int(newWidth/2)-20,int(newHeight/2)),(0,0,0),1) #hline
      #show temp
      cv2.putText(heatmap,str(temp)+' C', (int(newWidth/2)+10, int(newHeight/2)-10),\
      cv2.FONT_HERSHEY_SIMPLEX, 0.45,(0, 0, 0), 2, cv2.LINE_AA)
      cv2.putText(heatmap,str(temp)+' C', (int(newWidth/2)+10, int(newHeight/2)-10),\
      cv2.FONT_HERSHEY_SIMPLEX, 0.45,(0, 255, 255), 1, cv2.LINE_AA)

      
      #Yeah, this looks like we can probably do this next bit more efficiently!
      #display floating max temp
      if maxtemp > avgtemp+threshold:
         cv2.circle(heatmap, (mrow*scale, mcol*scale), 5, (0,0,0), 2)
         cv2.circle(heatmap, (mrow*scale, mcol*scale), 5, (0,0,255), -1)
         cv2.putText(heatmap,str(maxtemp)+' C', ((mrow*scale)+10, (mcol*scale)+5),\
         cv2.FONT_HERSHEY_SIMPLEX, 0.45,(0,0,0), 2, cv2.LINE_AA)
         cv2.putText(heatmap,str(maxtemp)+' C', ((mrow*scale)+10, (mcol*scale)+5),\
         cv2.FONT_HERSHEY_SIMPLEX, 0.45,(0, 255, 255), 1, cv2.LINE_AA)

      #display floating min temp
      if mintemp < avgtemp-threshold:
         cv2.circle(heatmap, (lrow*scale, lcol*scale), 5, (0,0,0), 2)
         cv2.circle(heatmap, (lrow*scale, lcol*scale), 5, (255,0,0), -1)
         cv2.putText(heatmap,str(mintemp)+' C', ((lrow*scale)+10, (lcol*scale)+5),\
         cv2.FONT_HERSHEY_SIMPLEX, 0.45,(0,0,0), 2, cv2.LINE_AA)
         cv2.putText(heatmap,str(mintemp)+' C', ((lrow*scale)+10, (lcol*scale)+5),\
         cv2.FONT_HERSHEY_SIMPLEX, 0.45,(0, 255, 255), 1, cv2.LINE_AA)

      if osd==True:
         clYellow        = (  0,255,255)
         clTelegrey4     = (200,200,200)
         clDarkWashedRed = ( 40, 40,255)

         if recording:
            recColor = clDarkWashedRed
         else:
            recColor = clTelegrey4

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

         cv2.putText(heatmap,'Recording: '+elapsed         , (10, p), font, 0.4, recColor, 1, cv2.LINE_AA)
         p += 14

         cv2.putText(heatmap,'Close (ESC or q)'            , (10, p), font, 0.4, clYellow, 1, cv2.LINE_AA)
         p += 14

      #display image
      cv2.imshow('Thermal',heatmap)

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
               cv2.resizeWindow('Thermal', newWidth, newHeight)
         case '-':        # decrease scale
            scale = max(1, scale - 1)
            newWidth  = width  * scale 
            newHeight = height * scale
            if not fullscreen and not isPi:
               cv2.resizeWindow('Thermal', newWidth, newHeight)
         case 'f':        # toggle fullscreen
            fullscreen = not fullscreen
            if fullscreen:
               cv2.setWindowProperty('Thermal', cv2.WND_PROP_FULLSCREEN, 1.0);
            else:
               cv2.setWindowProperty('Thermal', cv2.WND_PROP_FULLSCREEN, 0.0);
               cv2.resizeWindow('Thermal', newWidth, newHeight);
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
               cv2.resizeWindow('Thermal', newWidth, newHeight);
         case 's':        # toggle through image smoothing radius values
            rad += 1
            if rad > 5:
               rad = 0
         case 'p':        # print snapshot to png 
            snaptime = snapshot(heatmap)

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
      
