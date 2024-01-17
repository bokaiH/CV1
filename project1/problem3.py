'''
Thanks Yuran Xiang for the help of this problem
-----------------------------------------------
This is the code for project 1 question 3
A 2D scale invariant world
'''
from math import sqrt
import numpy as np
import cv2
from PIL import Image
r_min = 1
def inverse_cdf(x):
    ''' 
    Parameters:
        1. x: the random number sampled from uniform distribution
    Return:
        1. y: the random number sampled from the cubic law power
    '''
    y = x # Need to be changed
    # TODO: Add your code here
    y = np.sqrt(1/(1 - x))
    return y
def GenLength(N):
    ''' 
    Function for generating the length of the line
    Parameters:
        1. N: the number of lines
    Return:
        1. random_length: N*1 array, the length of the line, sampled from sample_r
    Tips:
        1. Using inverse transform sampling. Google it!
    '''
    # sample a random number from uniform distribution
    U = np.random.random(N)
    random_length = inverse_cdf(U)
    return random_length

def DrawLine(points,rad,length,pixel,N):
    ''' 
    Function for drawing lines on a image
    Parameters:
        1. points: N*2 array, the coordinate of the start points of the lines, range from 0 to pixel
        2. rad: N*1 array, the orientation of the line, range from 0 to 2\pi
        3. length: N*1 array, the length of the line, sampled from sample_r
        4. pixel: the size of the image
        5. N: the number of lines
    Return:
        1. bg: the image with lines
    '''
    # background
    bg = 255*np.ones((pixel,pixel)).astype('uint8')

    # TODO: Add your code here
    for i in range(N):
        x1 = points[i, 0]
        y1 = points[i, 1]
        x2 = int(x1 + length[i] * np.cos(rad[i]))
        y2 = int(y1 + length[i] * np.sin(rad[i]))
        if length[i] > 1:

            x2 = max(0, min(pixel-1, x2))
            y2 = max(0, min(pixel-1, y2))

            cv2.line(bg, (x1, y1), (x2, y2), (0, 0, 0), 1)

    cv2.imwrite('./pro3_result/'+str(pixel)+'.png', bg)
    return bg

def solve_q1(N = 5000,pixel = 1024):
    ''' 
    Code for solving question 1
    Parameters:
        1. N: the number of lines
        2. pixel: the size of the image
    '''
    # Generating length
    length = GenLength(N)

    # Generating starting points uniformly
    points = np.array([[0,0] for i in range(N)]) # Need to be changed
    # TODO: Add your code here
    points[:, 0] = np.random.uniform(0, pixel, N)
    points[:, 1] = np.random.uniform(0, pixel, N)
        
    # Generating orientation, range from 0 to 2\pi
    rad = np.array([0 for i in range(N)]) # Need to be changed
    # TODO: Add your code here
    rad = np.random.uniform(0, 2*np.pi, N)

    image = DrawLine(points,rad,length,pixel,N)
    return image,points,rad,length

def DownSampling(img,points,rad,length,pixel,N,rate):
    ''' 
    Function for down sampling the image
    Parameters:
        1. img: the image with lines
        2. points: N*2 array, the coordinate of the start points of the lines, range from 0 to pixel
        3. rad: N*1 array, the orientation of the line, range from 0 to 2\pi
        4. length: N*1 array, the length of the line
        5. pixel: the size of the image
        6. rate: the rate of down sampling
    Return:
        1. image: the down sampled image
    Tips:
        1. You can use Drawline for drawing lines after downsampling the components
    '''
    image = img # Need to be changed    
    # TODO: Add your code here
    image = DrawLine(points//rate, rad, length/rate, pixel//rate, N)

    return image

def crop(image1,image2,image3):
    ''' 
    Function for cropping the image
    Parameters:
        1. image1, image2, image3: I1, I2, I3
    '''
    
    # TODO: Add your code here
    image1 = Image.fromarray(image1)
    width, height = image1.size
    crop1_x1 = np.random.randint(0, height-128)
    crop1_x2 = crop1_x1 + 128
    crop1_y1 = np.random.randint(0, width-128)
    crop1_y2 = crop1_y1 + 128
    image1_crop1 = image1.crop((crop1_x1, crop1_y1, crop1_x2, crop1_y2))
    image1_crop1.save('./pro3_result/crop/1'+'.png')
    

    crop1_x1_ = np.random.randint(0, height-128)
    crop1_x2_ = crop1_x1_ + 128
    crop1_y1_ = np.random.randint(0, width-1-128)
    crop1_y2_ = crop1_y1_ + 128
    image1_crop2 = image1.crop((crop1_x1_, crop1_y1_, crop1_x2_, crop1_y2_))
    image1_crop2.save('./pro3_result/crop/2'+'.png')

    image2 = Image.fromarray(image2)
    height_, width_ = image2.size
    crop2_x1 = np.random.randint(0, height_-128)
    crop2_x2 = crop2_x1 + 128
    crop2_y1 = np.random.randint(0, width_-128)
    crop2_y2 = crop2_y1 + 128
    image2_crop1 = image2.crop((crop2_x1, crop2_y1, crop2_x2, crop2_y2))
    image2_crop1.save('./pro3_result/crop/3'+'.png')

    crop2_x1_ = np.random.randint(0, height_-128)
    crop2_x2_ = crop2_x1_ + 128
    crop2_y1_ = np.random.randint(0, width_-128)
    crop2_y2_ = crop2_y1_ + 128
    image2_crop2 = image2.crop((crop2_x1_, crop2_y1_, crop2_x2_, crop2_y2_))
    image2_crop2.save('./pro3_result/crop/4'+'.png')

    image3 = Image.fromarray(image3)
    height__, width__ = image3.size
    crop3_x1 = np.random.randint(0, height__-128)
    crop3_x2 = crop3_x1 + 128
    crop3_y1 = np.random.randint(0, width__-128)
    crop3_y2 = crop3_y1 + 128
    image3_crop1 = image3.crop((crop3_x1, crop3_y1, crop3_x2, crop3_y2))
    image3_crop1.save('./pro3_result/crop/5'+'.png')

    crop3_x1_ = np.random.randint(0, height__-128)
    crop3_x2_ = crop3_x1_ + 128
    crop3_y1_ = np.random.randint(0, width__-128)
    crop3_y2_ = crop3_y1_ + 128
    image3_crop2 = image3.crop((crop3_x1_, crop3_y1_, crop3_x2_, crop3_y2_))
    image3_crop2.save('./pro3_result/crop/6'+'.png')

    return


def main():
    N = 10000
    pixel = 1024
    image_1024, points, rad, length = solve_q1(N,pixel)
    # 512 * 512
    image_512 = DownSampling(image_1024,points, rad, length, pixel, N, rate = 2)
    # 256 * 256
    image_256 = DownSampling(image_1024,points, rad, length, pixel, N, rate = 4)
    crop(image_1024,image_512,image_256)
if __name__ == '__main__':
    main()
