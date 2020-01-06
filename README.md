# Cell Polarity Measurements

Scripts to measure and display polairty and kinociliary position information from images of the Organ of Corti.

## Measurement of polarity angles and kinociliary locations.

`find_angles_manual_bundle.py` loads images and provides a graphical interface to measure the polarity of hair bundles and the position of kinocilia.
It has been tested in Python 3.6, it might also work with Python 2.

Dependencies:

* numpy
* matplotlib
* pandas
* scikit-image

To run it type:

> pythonw find_angles_manual_bundle.py <name_of_the_image.tif>

If the images have a good cell junction marker in the blue channel the script can try to automatically identify the apical surfaces of cells. The automatic mode can be turned off by pressing 'a'. Once the image is loaded click on the center of the first cell you want to measure, if the auto mode is turned on it will try to detect the apical surface of the cell and draw a circle on top. If the auto mode is turned off do a second click on the periphery of the cell to define the size of the apical surface. After the cell periphery is marked click on the base of the kinocilium to mark its position, it will be signaled by a red spot. Finally make two clicks to define a line perpendicular to the orientation of the hair bundle, this will measure its orientation which will be signaled by a line. If you make a mistake at any point pressing 'd' will delete the last measurement made.
Repeat this process until you have measured all the cells in an image. If the field of cells is tilted with respect to the border of the image you can press 'r' at any time and make two clicks to define a reference line, all the angles will then be calculated with respect to this line, instead of the border of the image. Use the left and right arrow keys to move through the image stack.
After finishing with an image close the window and the script will generate a `.csv` with the same base name as the image and will save there all the measurements.

## Plotting results.

After measuring several images we combine them into Excel files manually creating one tab for each outer hair cell row and combining measurements of other images. Some `.xslx` files are provided as example. These files can by plotted using `polar_plot.py`. The list of files to be plotted is called 'files_list' inside the python script. This script will generate bundle angle histograms, kinociliary angle histograms, and kinociliary position scatter plots for each tab in each Excel file.
