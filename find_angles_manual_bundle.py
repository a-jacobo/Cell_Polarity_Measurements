#! /Users/ajacobo/anaconda/bin/pythonw

from __future__ import division
from __future__ import print_function
from builtins import zip
from builtins import str
from builtins import object
from past.utils import old_div
import numpy as np
from skimage import io
from matplotlib import pyplot as pl
from skimage.filters import gaussian, threshold_otsu,sobel,scharr
from skimage.transform import hough_circle,hough_ellipse
from skimage.exposure import rescale_intensity
from skimage.segmentation import active_contour
from skimage.feature import canny
from skimage.morphology import watershed,binary_closing,remove_small_holes,disk
from skimage.morphology import binary_opening
import pandas as pd
import sys

def crop(img,x0,y0,size):
    return img[np.int(y0-old_div(size,2)):np.int(y0+old_div(size,2)),
               np.int(x0-old_div(size,2)):np.int(x0+old_div(size,2))]

def draw(background,cells,cell_center_radii,kinos,ref_angle,bundle_lines):
        ax1.clear()
        ax1.imshow(background)
        if len(ref_angle)==2:
            x0,y0=ref_angle[0]
            x1,y1=ref_angle[1]
            ax1.plot([x0,x1],[y0,y1],'-y',lw=1.)
        for init in cells:
            ax1.plot(init[:, 0], init[:, 1], '--w', lw=1.)
        for k,arrow in zip(kinos,cell_center_radii):
            x0=arrow[0]
            y0=arrow[1]
            r= arrow[2]
            k_theta= k[0]-old_div(np.pi,2)
            k_r = k[1]*r
            xk = x0 - k_r*np.sin(k_theta)
            yk = y0 - k_r*np.cos(k_theta)
            ax1.plot([xk],[yk],'ro',markersize=10.)
        for c0,c1 in bundle_lines:
            ax1.plot([c0[0],c1[0]],[c0[1],c1[1]],'w', lw=1.)
        ax1.set_xticks([]), ax1.set_yticks([])
        ax1.axis([0, ysize, xsize, 0])
        fig.canvas.draw()

def update_title(stack_counter,hc_auto,bundle_auto):
    title = 'z = '+str(stack_counter)
    if hc_auto == 1:
        title = title +' |HC detection = On'
    else:
        title = title +' |HC detection = Off'
    # if bundle_auto == 1:
    #     title = title +' |Bundle detection = On'
    # else:
    #     title = title +' |Bundle detection = Off'
    return title

def find_cell_circle(x0,y0,hc_slice):
        image = crop(hc_slice,x0,y0,crp_size)
        #image = rescale_intensity(image)
        image = gaussian(image,1)
        thresh=threshold_otsu(image)
        edge=sobel(image<0.4*thresh)+sobel(image>0.8*thresh)
        integral=[]
        radii = np.arange(20,50,2.)
        # Find the standard deviation along circles of different radii
        for radius in radii:
            tc=init_circle(old_div(crp_size,2),old_div(crp_size,2),radius).astype(np.int)
            integral.append(old_div(np.std(edge[tc[:,0],tc[:,1]]),np.sum(edge[tc[:,0],tc[:,1]])))

        # The circle for which the image pixels have the smallest standard deviation
        # is the one that best fits the perimeter of the cells
        mx=np.argmin(integral)
        radius=radii[mx]#-2.
        coord=old_div(crp_size,2)+np.arange(-15,15)
        integral=np.zeros((len(coord),len(coord),len(radii)))

        mx = np.unravel_index(np.argmin(integral, axis=None), integral.shape)
        cx=old_div(crp_size,2)#coord[mx[0]]
        cy=old_div(crp_size,2)#coord[mx[1]]
        init=init_circle(cx,cy,radius)

        img_circle=np.zeros((crp_size, crp_size))
        circ_x=init[:,0].astype(np.int)
        circ_y=init[:,1].astype(np.int)
        img_circle[circ_x,circ_y] = 1

        edge = image*(1.+10.*gaussian(img_circle,5))

        # Shift the circle around in x and y to find the best center of the circle
        j=0
        for x in coord:
            i=0
            for y in coord:
                tc=init_circle(x,y,radius).astype(np.int)
                integral[j,i]=np.std(edge[tc[:,0],tc[:,1]])
                i+=1
            j+=1

        mx = np.unravel_index(np.argmin(integral, axis=None), integral.shape)
        init=init_circle(coord[mx[0]],coord[mx[1]],radius)

        cx=coord[mx[0]]
        cy=coord[mx[1]]

        init=init_circle(cx,cy,radius)

        return x0-(old_div(crp_size,2)-cx),y0-(old_div(crp_size,2)-cy),radius

def find_bundle_angle(new_x,new_y,radius,hc_slice):
    print("I should be findind a bundle, but I'm not")

def delete_elements_from_lists():
    global marker_status
    # If we are in cell marking mode, delete the last bundle and
    # go back to bundle marking mode
    if 'hair_cell_' in marker_status:
        try:
            bundles.pop()
            bundle_lines.pop()
        except:
            pass
        marker_status = 'bundle_1'
    # If there are the same number of bundles as cell centers, delete the
    # last bundle
    elif 'bundle' in marker_status:
        try:
            kinos.pop()
        except:
            pass
        marker_status = 'kino'
    # If there are more cells than bundles, delete the last cell, and go
    # back to cell marking mode.
    elif 'kino' in marker_status:
        try:
            cell_center_radii.pop()
            cells.pop()
        except:
            pass
        marker_status = 'hair_cell_1'

def init_circle(x0,y0,r):
    s = np.linspace(0, 2*np.pi, 200)
    x = x0 + r*np.cos(s)
    y = y0 + r*np.sin(s)
    return np.array([x, y]).T

class EventHandler(object):
    def __init__(self):
        #fig.canvas.mpl_connect('button_press_event', self.onpress)
        fig.canvas.mpl_connect('button_release_event', self.on_release)
        fig.canvas.mpl_connect('key_press_event', self.on_key )
    def on_release(self, event):
        global marker_status,prev_status,hc_pos,bundle_pos
        x0=event.xdata
        y0=event.ydata
        if marker_status == 'reference_angle_1':
            # Mark the firs point of the reference line
            ref_angle.append([x0,y0])
            marker_status = 'reference_angle_2'
        elif marker_status == 'reference_angle_2':
            # Mark the second point of the reference line, and go back to
            # the previous marking mode
            ref_angle.append([x0,y0])
            marker_status = prev_status
            xk=ref_angle[0][0]-ref_angle[1][0]
            yk=ref_angle[0][1]-ref_angle[1][1]
            k_theta= np.arctan(old_div(-yk,xk))#+np.pi
            #print k_theta,np.rad2deg(k_theta)
        elif marker_status == 'hair_cell_1':
            if hc_auto==1:
                #Fit a circle inside the cell
                new_x,new_y,radius  = find_cell_circle(x0,y0,hc_slice)
                #Update the list of circles
                init=init_circle(new_x,new_y,radius)
                cells.append(init)
                cell_center_radii.append([new_x,new_y,radius])
                #Draw the circles and cell_center_radii
                marker_status = 'kino'
            else:
                #Manually mark the center of the cell
                hc_pos=[[x0,y0]]
                #Catch the next click to mark the edge of the cell
                marker_status = 'hair_cell_2'
                return
        elif marker_status == 'hair_cell_2':
            #Mark the edge of the cell
            hc_pos.append([x0,y0])
            hc_pos= np.array(hc_pos)
            new_x,new_y=hc_pos[0]
            #Calculate the radius of the cell
            radius= np.linalg.norm(hc_pos[0]-hc_pos[1])
            init=init_circle(new_x,new_y,radius)
            #Add the circle of this cell to the list
            cells.append(init)
            #Add the center and radius of this cell to the list
            cell_center_radii.append([new_x,new_y,radius])
            #draw(img[stack_counter,:,:,:],cells,cell_center_radii,kinos,ref_angle,bundle_lines)
            marker_status = 'kino'
        elif marker_status == 'kino':
                # Mark the position of the kinocilium and update the list.
                new_x,new_y,radius = cell_center_radii[-1]
                xk=-int(x0-new_x)
                yk=int(y0-new_y)
                k_theta= np.arctan2(yk,xk)+np.pi
                k_r = old_div(np.sqrt(xk**2+yk**2),radius)
                kinos.append([k_theta,k_r])
                # Once the kinocilium is marked go back to cell marking mode
                marker_status = 'bundle_1'
        elif marker_status=='bundle_1':
            if bundle_auto==0:
                 # Mark the first end of the bundle
                 bundle_pos=[[x0,y0]]
                 # Catch the next click as the second end of the bundle
                 marker_status = 'bundle_2'
        elif marker_status=='bundle_2':
            # Mark the second end of the bundle
            bundle_pos.append([x0,y0])
            bundle_lines.append(bundle_pos)
            # Calculate the angle of the bundle and add to list
            xk=bundle_pos[1][0]-bundle_pos[0][0]
            yk=bundle_pos[1][1]-bundle_pos[0][1]
            # b_theta= np.arctan(-yk/xk)
            # print(np.rad2deg(b_theta))
            b_theta= (np.arctan2(-yk,xk)+2.*np.pi)%(2.*np.pi)
            bundles.append(b_theta)
            print(np.rad2deg(b_theta))
            # Once the bundle is marked go into kinocilium marking mode
            marker_status = 'hair_cell_1'
        #Draw the circles and cell_center_radii
        draw(img[stack_counter,:,:,:],cells,cell_center_radii,kinos,ref_angle,bundle_lines)
        return

    def on_key(self,event):
        global stack_counter,marker_status,mark_ref_angle,ref_angle
        global hc_auto,bundle_auto,prev_status

        if (event.key).lower()=='a':
            hc_auto = not hc_auto
        # if (event.key).lower()=='b':
        #     bundle_auto = not bundle_auto
        if (event.key).lower()=='r':
            prev_status=marker_status
            marker_status = 'reference_angle_1'
            ref_angle=[]
        if event.key=='left':
            stack_counter = min(stack_counter+1,zsize-1)
        if event.key=='right':
            stack_counter = max(stack_counter-1,0)
        if (event.key).lower()=='d':
            delete_elements_from_lists()

        draw(img[stack_counter,:,:,:],cells,cell_center_radii,kinos,ref_angle,bundle_lines)
        title= update_title(stack_counter,hc_auto,bundle_auto)

        fig.canvas.set_window_title(title)
        return

# Load the image.
filename = sys.argv[1]
img=io.imread(filename).astype(np.float32)
img=(old_div(255*img,np.max(img))).astype(np.uint8)
#print img.dtype
#print img.shape

zsize,xsize,ysize,nchannels=img.shape
dpi=50

#Hair Cells Perimeter Stack
#hc_stack = 6 #zsize-1
#Hair Cells Channel
hc_color = 2 #Red =0, Green =1, Blue =2

hc_slice = rescale_intensity(np.max(img[:,:,:,hc_color],axis=0))
#red_channel = (np.mean(img[:,:,:,ga_color],axis=0)).astype(np.uint8)
#red_channel = rescale_intensity(red_channel)

crp_size=200
yy,xx=np.mgrid[-50:50,-50:50]
yy=np.flipud(yy)
r = np.sqrt(xx**2+yy**2)
stack_counter=7
marker_status = 'hair_cell_1'
prev_status = ''
hc_auto=1
bundle_auto=0

cells=[]
cell_center_radii=[]
kinos=[]
bundles =[]
bundle_lines=[]

ref_angle=[]
hc_pos=[]
bundle_pos=[]

fig=pl.figure(num=None, figsize=(old_div(ysize,dpi),old_div(xsize,dpi)), dpi=dpi, facecolor='w')
ax1 = fig.add_subplot(111)

title= update_title(stack_counter,hc_auto,bundle_auto)

fig.canvas.set_window_title(title)

fig.subplots_adjust(bottom=0.,left=0.,top=1.,right=1.)
draw(img[stack_counter,:,:,:],cells,cell_center_radii,kinos,ref_angle,bundle_lines)
handler = EventHandler()

pl.show()

complete_cells=len(cell_center_radii)
kinos=kinos[:complete_cells]
bundles=bundles[:complete_cells]
#Save the data after closing the main window
out=np.hstack((cell_center_radii,kinos))
# print np.rad2deg(bundles)
bundles=np.reshape(np.array(bundles),(len(bundles),1))
out=np.hstack((out,bundles))
if len(out)>0:
    if len(ref_angle)!=0:
        xk=ref_angle[0][0]-ref_angle[1][0]
        yk=ref_angle[0][1]-ref_angle[1][1]
        k_theta= np.arctan(old_div(-yk,xk))
    else:
        k_theta = 0
        #bundles=np.reshape(np.array(bundles),(len(bundles),1))
    kangle = np.array(kinos)[:,0]
    kangle = kangle.reshape((len(kangle),1))
    # Add corrected Kinocilia angles
    out=np.hstack((out,kangle-k_theta))
    # Add corrected Bundle angles
    out=np.hstack((out,bundles-k_theta))
    df = pd.DataFrame(out, columns=['Cell_X','Cell_Y','Cell_Radius','Kino_Angle','Kino_Radius',
                                    'Bundle_Angle','Corrected_Kino_Angle','Corrected_Bundle_Angle'])
    #else:
#        df = pd.DataFrame(out, columns=['Cell_X','Cell_Y','Cell_Radius','Kino_Angle','Kino_Radius','Bundle_Angle'])
    df.to_csv(filename[0:-3]+'csv')

#f=open('data.txt','w')
#f.write('Cell_X\tCell_Y\tCell_Radius\tPolarity\tKino_Raidus\tKino_Angle\n')
#np.savetxt(f,out,delimiter='\t',fmt='%.3f')
#f.close()
