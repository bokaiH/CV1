'''
This is the code for project 1 question 1
Question 1: High kurtosis and scale invariance
'''
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from PIL import Image
from scipy.stats import gennorm, fit, norm
from scipy.optimize import curve_fit
import scipy.special
from scipy.special import gamma
from tqdm import tqdm
from math import sqrt

data_repo = "./image_set"
set_repo = ['setA','setB','setC']
img_name_list = []
def read_img_list(set):
    '''
    Read images from the corresponding image set
    '''
    global img_name_list
    img_list = os.listdir(os.path.join(data_repo,set))
    img_list.sort()
    img_name_list.append(img_list)
    img_list = [Image.open(os.path.join(data_repo,set,img)) for img in img_list]
    return img_list

# (a) First convert an image to grey level and re-scale the intensity to [0,31]
def convert_grey(img):
    '''
    Convert an image to grey
    Parameters:
        1. img: original image
    Return:
        1. img_grey: grey image

    '''
    img_grey = img # Need to be changed

    # TODO: Add your code here
    img_grey = img.convert('L')

    return img_grey

def rescale(img_grey):
    '''
    Rescale the intensity to [0,31]
    Parameters:
        1. img_grey: grey image
    Return:
        1. scale_img_grey: scaled grey image

    '''
    scale_img_grey = img_grey # Need to be changed
    # TODO: Add your code here
    img_grey_array = np.array(img_grey)
    scale_img_array = img_grey_array / 255 * 31
    scale_img_grey = Image.fromarray(scale_img_array)
    

    return scale_img_grey


# (b) Convolve the images with a horizontal gradient filter ∇xI
def gradient_filter(img):
    '''
    This function is used to calculate horizontal gradient
    Parameters:
        1. img: img for calculating horizontal gradient 
    Return:
        1. img_dx: an array of horizontal gradient

    >>> img = np.array([[1,2,3],[4,5,6],[7,8,9]])
    >>> gradient_filter(img)
    array([[1, 1],
           [1, 1],
           [1, 1]])
    '''
    img_dx = np.array([2, 1, 3, 1, 1, 1])    # Need to be changed
    # TODO: Add your code here
    img_array = np.array(img)
    height, width = img_array.shape
    img_dx = np.zeros((height, width-1))
    for i in range(width-1):
        img_dx[:, i] = img_array[:, i+1] - img_array[:, i]

    return img_dx


def plot_Hz(img_dx,log = False):
    '''
    This function is used to plot the histogram of horizontal gradient
    '''
    # clear previous plot
    hz, bins_edge = np.histogram(img_dx, bins=list(range(-31, 31)))
    hz = hz/np.sum(hz)
    epsilon = 1e-5
    if log:
        plt.plot(bins_edge[:-1], np.log(hz+epsilon), 'b-',label="log Histogram")
    else:
        plt.plot(bins_edge[:-1], hz, 'b-',label="Histogram")
    return hz, bins_edge

def compute_mean_variance_kurtosis(img_dx):
    '''
    Compute the mean, variance and kurtosis 
    Parameters:
        1. img_dx: an array of horizontal gradient
    Return:
        1. mean: mean of the horizontal gradient
        2. variance: variance of the horizontal gradient
        3. kurtosis: kurtosis of the horizontal gradient

    '''
    mean = 0
    variance = 0
    kurtosis = 0
    # TODO: Add your code here
    mean = np.mean(img_dx)
    variance = np.mean((img_dx - mean)**2)
    kurtosis = np.mean(((img_dx - mean)/np.sqrt(variance))**4)

    return mean, variance, kurtosis


def GGD(x, sigma, gammar):
    ''' 
    pdf of GGD
    Parameters:
        1. x: input
        2. sigma: σ
        3. gammar: γ
    Note: The notation of x,σ,γ is the same as the document
    Return:
        1. y: pdf of GGD

    '''
    y = 0
    # TODO: Add your code here
    y = gennorm.pdf(x, gammar, scale=sigma)
    return y


def fit_GGD(hz, bins_edge):
    '''
    Fit the histogram to a Generalized Gaussian Distribution (GGD), and report the fittest sigma and gamma
    Parameters:
        1. hz: histogram of the horizontal gradient
        2. bins_edge: bins_edge of the histogram
    Return:
        None
    '''
    # fit the histogram to a generalized gaussian distribution

    datax = bins_edge[:-1]
    datay = hz


    # TODO: Add your code here
    popt, pcov = curve_fit(GGD, datax, datay)

    sig = popt[0]
    gam = popt[1]

    print(sig, gam)
    return


def plot_Gaussian(mean,variance):
    ''' 
    Plot the Gaussian distribution using the mean and the variance
    Parameters:
        1. mean: mean of the horizontal gradient
        2. variance: variance of the horizontal gradient
    Return:
        None

    '''
    x = np.linspace(-31,31,500)

    y = np.zeros(x.shape) # Need to be changed

    # TODO: Add your code here
    y = (1/sqrt(2*np.pi*variance))*np.exp((-(x-mean)**2)/(2*variance))
    # y: value of pdf of Gassian distribution corresponding to x

    plt.plot(x, y,'g-', label="Gaussian")
    plt.xlabel('x')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.show()

    return 


def downsample(image):
    ''' 
    Downsample our images
    Parameters:
        1. image: original image
    Return:
        1. processed_image: downsampled image
    '''
    processed_image = image # Need to be changed
    # TODO: Add your code here
    image_array = np.array(image)

    height, width = image_array.shape[:2]

    new_height = height // 2
    new_width = width // 2

    downsampled_image = cv2.resize(image_array, (new_width, new_height), interpolation=cv2.INTER_AREA)

    processed_image = Image.fromarray(downsampled_image)
    return processed_image


def main():
    '''
    This is the main function
    '''
    # read img to img list
    # Notice: img_list is a list of image
    img_list = [read_img_list(set) for set in set_repo]
    # set_repo refers to the three sets we'll handle
    for idx1,set in enumerate(set_repo):
        img_dx_list = []
        img_dx_2_list = []
        img_dx_4_list = []
        for idx2,img in enumerate(img_list[idx1]):
            if set == 'setC':
                img_grey = convert_grey(img)
                img_2_grey = downsample(img_grey)
                img_4_grey = downsample(img_2_grey)
            else:
                img_grey = convert_grey(img)
                img_2_grey = downsample(img_grey)
                img_4_grey = downsample(img_2_grey)

                img_grey = rescale(img_grey)
                img_2_grey = rescale(img_2_grey)
                img_4_grey = rescale(img_4_grey)

            img_dx_list.append(gradient_filter(img_grey).flatten())
            img_dx_2_list.append(gradient_filter(img_2_grey).flatten())
            img_dx_4_list.append(gradient_filter(img_4_grey).flatten())
        img_dx = np.concatenate(img_dx_list)
        img_dx_2 = np.concatenate(img_dx_2_list)
        img_dx_4 = np.concatenate(img_dx_4_list)


        # plot histogram and log histogram
        print('--'*20)

        plt.clf()
        hz, bins_edge = plot_Hz(img_dx)
        # compute mean, variance and kurtosis
        mean, variance, kurtosis = compute_mean_variance_kurtosis(img_dx)
        print(f"set: {set}")
        print(f"mean: {mean}, variance: {variance}, kurtosis: {kurtosis}")

        # fit the histogram to a generalized gaussian distribution
        fit_GGD(hz, bins_edge)

        # plot the Gaussian distribution using the mean and the variance
        plot_Gaussian(mean,variance)

        plt.savefig(f"./pro1_result/histogram/{set}.png")

        # plot log histogram

        plt.clf()
        plot_Hz(img_dx,log=True)
        # save the histograms
        plt.savefig(f"./pro1_result/log_histogram/{set}.png")

        # plot the downsampled images histogram
        plt.clf()
        plot_Hz(img_dx)
        plt.savefig(f"./pro1_result/downsampled_histogram/original_{set}.png")

        plt.clf()
        plot_Hz(img_dx_2)
        plt.savefig(f"./pro1_result/downsampled_histogram/2_{set}.png")

        plt.clf()
        plot_Hz(img_dx_4)
        plt.savefig(f"./pro1_result/downsampled_histogram/4_{set}.png")
if __name__ == '__main__':
    main()
