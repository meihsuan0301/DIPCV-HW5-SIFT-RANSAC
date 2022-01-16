
import matplotlib.pyplot as plt
import numpy as np
import random
from utils import *

plt.rcParams['figure.figsize'] = [15, 15]

def Do_SIFT(target_path, scene_path, book):
    left_gray, left_origin, left_rgb = read_image(target_path)
    right_gray, right_origin, right_rgb = read_image(scene_path)
    
    # Better result when using gray
    kp_left, des_left = SIFT(left_gray)
    kp_right, des_right = SIFT(right_gray)
    kp_left_img = plot_sift(left_gray, left_rgb, kp_left)
    kp_right_img = plot_sift(right_gray, right_rgb, kp_right)
       
    plt.subplot(1,2,1)
    plt.title('Target Keypoint', fontsize = 15)
    plt.imshow(kp_left_img)
    plt.savefig(book+'_kp.png', bbox_inches='tight', pad_inches=0)
    plt.subplot(1,2,2)
    plt.title('Scene Keypoint', fontsize = 15)
    plt.imshow(kp_right_img)
    plt.savefig('Scene_kp.png', bbox_inches='tight', pad_inches=0)
    
    matches = matcher(kp_left, des_left, left_rgb, kp_right, des_right, right_rgb, 0.5)
    plot_matches(matches, left_rgb, right_rgb, book+'_SIFT') # Good mathces
    
    inliers, H = ransac(matches, 0.5, 2000)
    plot_matches(inliers, left_rgb, right_rgb, book+'_SIFT_use_RANSAC') # show inliers matches

if __name__ == '__main__':
    books = ['book1', 'book2', 'book3']
    target_path = ['./scene/book1.jpg', './scene/book2.jpg', './scene/book3.jpg']
    scene_path = ['./scene/scene.jpg', './scene/scene.jpg', './scene/scene.jpg']
    for book, target_path, scene_path in zip(books, target_path, scene_path):
        Do_SIFT(target_path, scene_path, book)