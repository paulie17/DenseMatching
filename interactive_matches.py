from utils_flow.pixel_wise_mapping import remap_using_flow_fields
import cv2
from model_selection import select_model
from utils_flow.visualization_utils import draw_keypoints

from test_models import pad_to_same_shape
import numpy as np
from validation.test_parser import define_model_parser
from validation.utils import matches_from_flow
import argparse
import torch


import tkinter as tk
from tkinter import filedialog


def interactive_window(source_img, target_img, network):


    # pad both images to the same size, to be processed by network
    query_image_, reference_image_ = pad_to_same_shape(target_img, source_img)
    # convert numpy to torch tensor and put it in right format
    query_image_ = torch.from_numpy(query_image_).permute(2, 0, 1).unsqueeze(0)
    reference_image_ = torch.from_numpy(reference_image_).permute(2, 0, 1).unsqueeze(0)
    estimated_flow = network.estimate_flow(query_image_, reference_image_, mode='channel_first')
    

    def draw_point(event, x, y, flags, param):
        # if left mouse button is clicked, add point to the list
        if event == cv2.EVENT_MOUSEMOVE:
            contact_points = []
            contact_points.append((x,y))
            contact_points = np.array(contact_points)
        

        print('Computing corresponding point...')
        target_pt = match_contact_points(estimated_flow,contact_points)
        print('Done!')

        # update the display
        cv2.imshow('Select source point (press any key to finish)', draw_keypoints(source_img,contact_points))
        cv2.imshow('Target image', draw_keypoints(target_img,target_pt))

    # create a window to display the image and register the mouse event handler
    cv2.namedWindow('Select source point (press any key to finish)')
    cv2.setMouseCallback('Select source point (press any key to finish)', draw_point)

    cv2.imshow('Target image', target_img)

    # display the image and wait for user to select points
    cv2.imshow('Select source point (press any key to finish)', source_img)
    cv2.waitKey(0)
    # close the window
    cv2.destroyAllWindows()

def match_contact_points(estimated_flow,contact_pixels):
    mask = np.zeros(estimated_flow.shape[-2:], dtype=int)[np.newaxis, ...]
    mask_indices = contact_pixels.astype(int)[:, [1, 0]]  # (x,y) -> (row, col)
    mask[:, mask_indices[:, 0], mask_indices[:, 1]] = 1
    mask = torch.tensor(mask, device=estimated_flow.device) == 1
    # print(estimated_flow.shape)
    # print(mask.shape)
    mkpts_q, _ = matches_from_flow(estimated_flow, mask)
    
    return mkpts_q

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test models on a pair of images')
    define_model_parser(parser)  # model parameters
    parser.add_argument('--pre_trained_model', type=str, help='Name of the pre-trained-model', required=True)
    
    args = parser.parse_args()

    torch.cuda.empty_cache()
    torch.set_grad_enabled(False) # make sure to not compute gradients for computational performance
    torch.backends.cudnn.enabled = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # either gpu or cpu

    local_optim_iter = args.optim_iter if not args.local_optim_iter else int(args.local_optim_iter)

    # Create a Tkinter window to prompt the user to select a file
    root = tk.Tk()
    root.withdraw()

    # Ask the user to select a source image
    file_path = filedialog.askopenfilename(title='Select a source image',initialdir='/home/paolo/Documents/Datasets/')
    reference_image = cv2.imread(file_path)

    # Display the uploaded image
    #cv2.imshow('Reference image', reference_image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    # Ask the user to select a target image
    file_path = filedialog.askopenfilename(title='Select a target image',initialdir='/home/paolo/Documents/Datasets/')

    target_image = cv2.imread(file_path)

    # Display the uploaded image
    #cv2.imshow('Uploaded image', target_image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    network, estimate_uncertainty = select_model(
            args.model, args.pre_trained_model, args, args.optim_iter, local_optim_iter,
            path_to_pre_trained_models=args.path_to_pre_trained_models)
     

    interactive_window(reference_image, target_image, network)