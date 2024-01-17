'''
This is the main file for the project 2's Second method PDE
'''
import cv2
import numpy as np
import random
from tqdm import tqdm
import torch
from torch.nn.functional import conv2d, pad
import os
import matplotlib.pyplot as plt




def pde(img, loc, beta):
    ''' 
    The function to perform the pde update for a specific pixel location
    Partial Differential Equations (PDE) in the context of image restoration and inpainting
    Parameters:
        1. img: the image to be processed, numpy array
        2. loc: the location of the pixel, loc[0] is the row number, loc[1] is the column number
        3. beta: learning rate
    Return:
        img: the updated image
    '''

    # TODO
    y, x = loc
    delta_I = img[y-1,x] + img[y,x-1] - 4*img[y,x] + img[y,x+1] + img[y+1,x]
    img[y,x] = img[y,x] + beta*delta_I*0.5

    return img


def main():
    # read the distorted image and mask image
    name = "room"
    size = "big"

    distorted_path = f"../image/{name}/{size}/imposed_{name}_{size}.bmp"
    mask_path = f"../image/mask_{size}.bmp"
    ori_path = f"../image/{name}/{name}_ori.bmp"


    # read the BGR image
    distort = cv2.imread(distorted_path).astype(np.float64)
    mask = cv2.imread(mask_path).astype(np.float64)
    ori = cv2.imread(ori_path).astype(np.float64)



    beta = 1
    img_height, img_width, _ = distort.shape

    sweep = 100
    
    error_list = []
    for s in tqdm(range(sweep)):
        for i in range(img_height):
            for j in range(img_width):
                # only change the channel red
                # TODO
                if mask[i,j,2] == 255:
                    distort[:,:,2] = pde(distort[:,:,2], [i,j], beta)
        # TODO
        beta *= 0.98
        # Plot the per pixel error over the number of sweeps t in for both methods.
        error = np.sum((distort - ori)**2)/img_height/img_width
        error_list.append(error)
        
        if s % 10 == 0:
            save_path = f"./result/{name}/{size}"
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            cv2.imwrite(f"{save_path}/pde_{s}.bmp", distort)

    plt.plot(range(sweep), error_list)
    plt.xlabel("sweep")
    plt.ylabel("error")
    plt.title("error vs sweep")
    plt.savefig(f"./result/{name}/{size}/pde_error.png")
    plt.show()


if __name__ == "__main__":
    main()







        

