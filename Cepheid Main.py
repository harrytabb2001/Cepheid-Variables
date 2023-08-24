#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 10:51:11 2022

@author: harrytabb
"""
from astropy.visualization import astropy_mpl_style
import numpy as np
import matplotlib.pyplot as plt
from astropy.utils.data import get_pkg_data_filename
from astropy.io import fits

#File Directories for each picture
#Ultraviolet
file1 = '/Users/harrytabb/Desktop/Uni/Uni Year 2/Cepheid/cepheid/u01_2.fits'
file2 = '/Users/harrytabb/Desktop/Uni/Uni Year 2/Cepheid/cepheid/u02_2.fits'
file3 = '/Users/harrytabb/Desktop/Uni/Uni Year 2/Cepheid/cepheid/u03_2.fits'
file4 = '/Users/harrytabb/Desktop/Uni/Uni Year 2/Cepheid/cepheid/u04_2.fits'
file5 = '/Users/harrytabb/Desktop/Uni/Uni Year 2/Cepheid/cepheid/u05_2.fits'
file6 = '/Users/harrytabb/Desktop/Uni/Uni Year 2/Cepheid/cepheid/u06_2.fits'
file7 = '/Users/harrytabb/Desktop/Uni/Uni Year 2/Cepheid/cepheid/u07_2.fits'
file8 = '/Users/harrytabb/Desktop/Uni/Uni Year 2/Cepheid/cepheid/u08_2.fits'
file9 = '/Users/harrytabb/Desktop/Uni/Uni Year 2/Cepheid/cepheid/u09_2.fits'
file10 = '/Users/harrytabb/Desktop/Uni/Uni Year 2/Cepheid/cepheid/u10_2.fits'
file11 = '/Users/harrytabb/Desktop/Uni/Uni Year 2/Cepheid/cepheid/u11_2.fits'
file12 = '/Users/harrytabb/Desktop/Uni/Uni Year 2/Cepheid/cepheid/u12_2.fits'

#Infrared
filei1 = '/Users/harrytabb/Desktop/Uni/Uni Year 2/Cepheid/cepheid/i1_2.fits'
filei2 = '/Users/harrytabb/Desktop/Uni/Uni Year 2/Cepheid/cepheid/i2_2.fits'
filei3 = '/Users/harrytabb/Desktop/Uni/Uni Year 2/Cepheid/cepheid/i3_2.fits'

#Coordinates of each cepheid
c1 = [157, 603,0]
c2 = [102, 455,1]
c3 = [363, 316,2]
c5 = [404, 603,4]
c6 = [430, 554,5]
c7 = [259,651,6]

c41 = [508, 89]
c42 = [509, 87]
c43 = [508, 89]
c44 = [509, 87]
c45 = [509, 88]
c46 = [509, 88]
c47 = [509, 95]
c48 = [508, 89]
c49 = [508, 89]
c410 = [508, 89]
c411 = [508, 89]
c412 = [508, 89]

#size of circles around each cepheid
npix_small = 29
npix_big = 151
npix_diff = npix_big - npix_small

#putting each cepheid into an array
cepheids = [c1, c2, c3, c5, c6]
cepheid4 = [c41, c42, c43, c44, c45, c46, c47, c48, c49, c410, c411, c412]

#function to get date out of each file
def startdate(filename):
    header = fits.getheader(filename)
    date = header['EXPSTART']
    return date

#function to import fits file
def import_data(filename):
    data = fits.getdata(filename)
    return data

#importing data and logging
data1 = import_data(file1)
logged_data1 = np.log(data1)

data2 = import_data(file2)
logged_data2 = np.log(data2)

data3 = import_data(file3)
logged_data3 = np.log(data3)

data4 = import_data(file4)
logged_data4 = np.log(data4)

data5 = import_data(file5)
logged_data5 = np.log(data5)

data6 = import_data(file6)
logged_data6 = np.log(data6)

data7 = import_data(file7)
logged_data7 = np.log(data7)

data8 = import_data(file8)
logged_data8 = np.log(data8)

data9 = import_data(file9)
logged_data9 = np.log(data9)

data10 = import_data(file10)
logged_data10 = np.log(data10)

data11 = import_data(file11)
logged_data11 = np.log(data11)

data12 = import_data(file12)
logged_data12 = np.log(data12)


data_array = [data1, data2, data3, data4, data5, data6, data7, data8, data9, data10, data11, data12]


#function to sum intensity of all pixels within circle around a cepheid
def get_intensity(data, plus):
    intensities = np.zeros(8)
    for i in cepheids:
        row1 = data[i[1]+3][i[0]]
        row2 = np.sum(np.fromiter((data[i[1]+2][i[0]+x] for x in np.arange(-2,3)), dtype=float))
        row3 = np.sum(np.fromiter((data[i[1]+1][i[0]+x] for x in np.arange(-2,3)), dtype=float))
        row4 = np.sum(np.fromiter((data[i[1]][i[0]+x] for x in np.arange(-3,4)), dtype=float))
        row5 = np.sum(np.fromiter((data[i[1]-1][i[0]+x] for x in np.arange(-2,3)), dtype=float))
        row6 = np.sum(np.fromiter((data[i[1]-2][i[0]+x] for x in np.arange(-2,3)), dtype=float))
        row7 = data[i[1]-3][i[0]]
        intensity = row1 + row2 + row3 + row4 + row5 + row6 + row7
        intensities[i[2]] = intensity
    return intensities

#function to sum intensity of all pixels within background region around a cepheid
def background(data):
    intensities = np.zeros(8)
    for i in cepheids:
        row1 = data[i[1]+7][i[0]]
        row2 = np.sum(np.fromiter((data[i[1]+6][i[0]+x] for x in np.arange(-3,4)), dtype=float))
        row3 = np.sum(np.fromiter((data[i[1]+5][i[0]+x] for x in np.arange(-4,5)), dtype=float))
        row4 = np.sum(np.fromiter((data[i[1]+4][i[0]+x] for x in np.arange(-5,6)), dtype=float))
        row5 = np.sum(np.fromiter((data[i[1]+3][i[0]+x] for x in np.arange(-6,7)), dtype=float))
        row6 = np.sum(np.fromiter((data[i[1]+2][i[0]+x] for x in np.arange(-6,7)), dtype=float))
        row7 = np.sum(np.fromiter((data[i[1]+1][i[0]+x] for x in np.arange(-6,7)), dtype=float))
        row8 = np.sum(np.fromiter((data[i[1]][i[0]+x] for x in np.arange(-7,8)), dtype=float))
        row9 = np.sum(np.fromiter((data[i[1]-1][i[0]+x] for x in np.arange(-6,7)), dtype=float))
        row10 = np.sum(np.fromiter((data[i[1]-2][i[0]+x] for x in np.arange(-6,7)), dtype=float))
        row11 = np.sum(np.fromiter((data[i[1]-3][i[0]+x] for x in np.arange(-6,7)), dtype=float))
        row12 = np.sum(np.fromiter((data[i[1]-4][i[0]+x] for x in np.arange(-5,6)), dtype=float))
        row13 = np.sum(np.fromiter((data[i[1]-5][i[0]+x] for x in np.arange(-4,5)), dtype=float))
        row14 = np.sum(np.fromiter((data[i[1]-6][i[0]+x] for x in np.arange(-3,4)), dtype=float))
        row15 = data[i[1]-7][i[0]]
        intensity = row1 + row2 + row3 + row4 + row5 +row6 + row7 + row8 + row9 + row10 + row11 + row12 + row13 + row14 + row15
        intensities[i[2]] = intensity
    return intensities

#function to plot sky map for each cepheid and time
def plot(data, date, number):
    plt.style.use(astropy_mpl_style)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.title('Region 1, Julian Date = {:.2f}'.format(date))
    plt.imshow(data, cmap='gist_ncar')
    for i in cepheids:
        ax.plot(i[0], i[1], 'o', ms=3, mec='b', mfc='none', mew=0.5, alpha=0.5)
        ax.plot(i[0], i[1], 'o', ms=7, mec='b', mfc='none', mew=0.5, alpha=0.5)
        ax.text(i[0], i[1]+20, '{}'.format(i[2] + 1), alpha=0.4)
        
    ax.plot(cepheid4[number-1][0], cepheid4[number-1][1], 'o', ms=3, mec='b', mfc='none', mew=0.5, alpha=0.5)
    ax.text(cepheid4[number-1][0], cepheid4[number-1][1]+20, '4', alpha=0.4)
    ax.plot(cepheid4[number-1][0]-10, cepheid4[number-1][1]-10, 'o', ms=7, mec='b', mfc='none', mew=0.5, alpha=0.5)
    
    ax.plot(c7[0], c7[1], 'o', ms=3, mec='b', mfc='none', mew=0.5, alpha=0.5)
    ax.text(c7[0], c7[1]+20, '7', alpha=0.4)
    ax.plot(c7[0], c7[1]+13, 'o', ms=7, mec='b', mfc='none', mew=0.5, alpha=0.5)
    ax.invert_yaxis()
    plt.colorbar(label='Log(Intensity)')
    plt.savefig('/Users/harrytabb/Desktop/Uni/Uni Year 2/Cepheid/cepheid/u0{}_2.png'.format(number), dpi=1000)
    return None

#function to find remove background noise from each cepheid
def final_intensity(main, background):
    main = np.array(main)
    background = np.array(background)
    difference = background - main
    difference_pp = difference / npix_diff
    
    intensity_pp_small = main / npix_small
    
    intensity_pp = intensity_pp_small - difference_pp
    intensity_final = npix_small * intensity_pp
    return intensity_final

#function to export data into a csv
def file(intensity, error, number):
    final = np.dstack((intensity, error))
    np.savetxt('/Users/harrytabb/Desktop/Uni/Uni Year 2/Cepheid/cepheid/Cepheid{0}/Cepheid{1}.csv'.format(number, number), final[0], fmt='%.3f', delimiter=',', header='Intensity, error')

#function to calculate error from removing background
def error(main, background):
    main = np.array(main)
    background = np.array(background)
    error = 29 * np.sqrt((main / 29**2) + ((main + background) / (151 - 29)**2))
    return error

#main code

cepheid_intensities = []
background_intensities = []

for i in np.arange(0,12):
    cepheid_intensities.append(get_intensity(data_array[i]))
    background_intensities.append(background(data_array[i]))
    
cepheid1 = []
cepheid2 = []
cepheid3 = []
cepheid5 = []
cepheid6 = []

background1 = []
background2 = []
background3 = []
background5 = []
background6 = []

for i in np.arange(0,12):
    cepheid1.append(cepheid_intensities[i][0])
    cepheid2.append(cepheid_intensities[i][1])
    cepheid3.append(cepheid_intensities[i][2])
    cepheid5.append(cepheid_intensities[i][3])
    cepheid6.append(cepheid_intensities[i][4])
    
for i in np.arange(0,12):
     background1.append(background_intensities[i][0])
     background2.append(background_intensities[i][1])
     background3.append(background_intensities[i][2])
     background5.append(background_intensities[i][3])
     background6.append(background_intensities[i][4])


cepheid1_final = final_intensity(cepheid1, background1)
cepheid2_final = final_intensity(cepheid2, background2)
cepheid3_final = final_intensity(cepheid3, background3)
cepheid5_final = final_intensity(cepheid5, background5)
cepheid6_final = final_intensity(cepheid6, background6)

cepheid1_error_final = error(cepheid1, background1)
cepheid2_error_final = error(cepheid2, background2)
cepheid3_error_final = error(cepheid3, background3)
cepheid5_error_final = error(cepheid5, background5)
cepheid6_error_final = error(cepheid6, background6)


plot(logged_data1, startdate(file1), 1)
plot(logged_data2, startdate(file2), 2)
plot(logged_data9, startdate(file9), 9)
plot(logged_data10, startdate(file10), 10)
plot(logged_data11, startdate(file11), 11)
plot(logged_data12, startdate(file12), 12)

file(cepheid1_final, cepheid1_error_final, 1)
file(cepheid2_final, cepheid2_error_final, 2)
file(cepheid3_final, cepheid3_error_final, 3)
file(cepheid5_final, cepheid5_error_final, 5)
file(cepheid6_final, cepheid6_error_final, 6)

datai1 = import_data(filei1)
datai2 = import_data(filei2)
datai3 = import_data(filei3)


a_date = startdate(filei1)
b_date = startdate(filei2)
c_date = startdate(filei3)

a = get_intensity(datai1, 0)
b = get_intensity(datai2, 0)
c = get_intensity(datai3, 0)

d = background(datai1)
e = background(datai2)
f = background(datai3)

a_final = final_intensity(a,d)
b_final = final_intensity(b,e)
c_final = final_intensity(c, f)

a_error = error(a, d)
b_error = error(b, e)
c_error = error(c, f)
