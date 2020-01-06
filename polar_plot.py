from __future__ import division
from __future__ import print_function
from builtins import str
from past.utils import old_div
import numpy as np
from matplotlib import pyplot as pl
import pandas as pd
import os
import matplotlib.ticker as ticker
from scipy.stats import circmean,circvar

''''========================================================================'''
pl.rc('axes', linewidth=1)
pl.rc('lines', markeredgewidth=1,antialiased=True)
pl.rcParams['xtick.major.pad']='6'
pl.rcParams['ytick.major.pad']='6'
pl.rcParams['xtick.labelsize']='8'
pl.rcParams['ytick.labelsize']='8'
pl.rcParams['font.size']='10'
pl.rcParams['font.family']='sans-serif'
pl.rcParams['ps.fonttype']=3
pl.rcParams['patch.linewidth']=0.1
''''========================================================================'''

def draw_straight_axis(xpos,tick_locations,ax):
    for tk in tick_locations:
        ax.text(np.arctan2(tk,xpos),np.hypot(tk,xpos),str(tk),verticalalignment='center')

files_list=['E18.5_L131_L132_L136_Control_Base_Pericentrin_Bundle_M2',
            'E18.5_L131_L132_L136_Control_Mid_Pericentrin_Bundle_M2',
            'E18.5_L131_L132_L136_Wls_cKO_Base_Pericentrin_Bundle_M2',
            'E18.5_L131_L132_L136_Wls_cKO_Mid_Pericentrin_Bundle_M2']

extension='.eps'
bins=30

for filename in files_list:
    xls = pd.ExcelFile(filename+'.xlsx')

    for sheet in xls.sheet_names:
        data = pd.read_excel(filename+'.xlsx', sheet_name=sheet)

        # Plot scatter plot of kinocilia positions
        fig = pl.figure()
        ax = fig.add_subplot(111, projection='polar')
        ax.set_axisbelow(True)
        c = ax.scatter(data['Corrected_Kino_Angle'], data['Kino_Radius'],color='#009e19',edgecolors='#057d18')

        tick_locations=[0.,0.25,0.5,0.75,1.]
        ax.yaxis.set_major_locator(pl.FixedLocator(tick_locations))
        ticks = ticker.NullFormatter()
        ax.yaxis.set_major_formatter(ticks)
        ax.xaxis.set_major_formatter(ticks)
        ax.set_ylim([0.,1.])
        xpos=-1.5
        draw_straight_axis(xpos,tick_locations,ax)
        # for tk in tick_locations:
        #     xpos=-1.5
        #     ax.text(np.arctan2(tk,xpos),np.hypot(tk,xpos),str(tk))


        pl.savefig('kinos_scatter_'+filename+'_'+sheet+extension)
        pl.close(fig)

        print(filename)
#===========================================================================
        # Plot histogram of kinocilia orientations
        hist,edges=np.histogram(data['Corrected_Kino_Angle'],density=False,bins=bins,range=[0,2.*np.pi])
        print('Kinocilia angles, number of bins: ',len(hist))
        width = np.diff(edges)[0]
        edges = edges + width/2.

        fig2 = pl.figure()
        ax1 = fig2.add_subplot(111, projection='polar')
        ax1.set_axisbelow(True)
        edgecolors=['k' for i in hist]
        bars = ax1.bar(edges[:-1],hist,width=width,color='#737373',edgecolor=edgecolors)
        #ax1.yaxis.set_major_locator(pl.MaxNLocator(3))
        #ax1.yaxis.set_major_locator(pl.LinearLocator(3))
        max_hist=np.max(hist)
        tick_locations=[int(max_hist/2.),max_hist]
        print(tick_locations)
        ax1.yaxis.set_major_locator(pl.FixedLocator(tick_locations))
        xpos=-1.5*max_hist

        #draw_straight_axis(xpos,ax1.yaxis.get_ticklocs(),ax1)
        #ax1.yaxis.set_major_formatter(ticks)
        ax1.yaxis.set_major_formatter(ticker.FixedFormatter(['',str(max_hist)]))
        ax1.set_rlabel_position(90)
        ax1.xaxis.set_major_formatter(ticks)
        mean_angle=circmean(data['Corrected_Kino_Angle'])
        std_angle=np.sqrt(circvar(data['Corrected_Kino_Angle']))

        ymin,ymax=ax1.get_ylim()
        ax1.plot([mean_angle,mean_angle],[0,ymax],'b')
        std_angles=np.linspace(mean_angle-old_div(std_angle,2),mean_angle+old_div(std_angle,2),11)
        std_radii=np.ones(11)*2.*max_hist/3.
        #ax1.plot([mean_angle-std_angle/2,mean_angle+std_angle/2],[2.*max_hist/3.,2.*max_hist/3.],'r')
        ax1.plot(std_angles,std_radii,'r')
        ax1.set_ylim([0,max_hist])

        pl.savefig('kinos_histogram_'+filename+'_'+sheet+extension)
        pl.close(fig2)
#===========================================================================
        # Plot histogram of bundle orientations
        hist,edges=np.histogram(data['Corrected_Bundle_Angle'],density=False,bins=bins,range=[0,2.*np.pi])
        print('Bundle angles, number of bins: ',len(hist))

        width = np.diff(edges)[0]
        edges = edges + width/2.

        fig3 = pl.figure()
        ax2 = fig3.add_subplot(111, projection='polar')
        ax2.set_axisbelow(True)
        #bars = ax2.bar(edges[:-1],hist,width=width,color=pl.cm.Accent(hist/float(len(hist))))
        edgecolors=['k' for i in hist]
        bars = ax2.bar(edges[:-1],hist,width=width,color='#737373',edgecolor=edgecolors)
        #ax2.yaxis.set_major_locator(pl.MaxNLocator(3))
        #ax2.yaxis.set_major_locator(pl.LinearLocator(3))

        max_hist=np.max(hist)
        tick_locations=[int(max_hist/2.),max_hist]
        ax2.yaxis.set_major_locator(pl.FixedLocator(tick_locations))
        #xpos=-1.5*max_hist
        #draw_straight_axis(xpos,ax1.yaxis.get_ticklocs(),ax2)
        #ax2.yaxis.set_major_formatter(ticks)
        ax2.yaxis.set_major_formatter(ticker.FixedFormatter(['',str(max_hist)]))
        ax2.set_rlabel_position(90)
        ax2.xaxis.set_major_formatter(ticks)

        mean_angle=circmean(data['Corrected_Bundle_Angle'])
        std_angle=np.sqrt(circvar(data['Corrected_Bundle_Angle']))
        std_angles=np.linspace(mean_angle-old_div(std_angle,2),mean_angle+old_div(std_angle,2),11)
        std_radii=np.ones(11)*2.*max_hist/3.

        ymin,ymax=ax2.get_ylim()
        ax2.plot([mean_angle,mean_angle],[0,ymax],'b')
        ax2.plot(std_angles,std_radii,'r')
        #ax2.plot([mean_angle-std_angle/2,mean_angle+std_angle/2],[2.*max_hist/3.,2.*max_hist/3.],'r')
        ax2.set_ylim([0,max_hist])
        pl.savefig('bundles_histogram_'+filename+'_'+sheet+extension)
        pl.close(fig3)
