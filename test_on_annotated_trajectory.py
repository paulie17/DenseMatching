from utils_flow.pixel_wise_mapping import remap_using_flow_fields
import cv2
from model_selection import select_model
from utils_flow.util_optical_flow import flow_to_image
from utils_flow.visualization_utils import overlay_semantic_mask,make_sparse_matching_plot, draw_keypoints

from test_models import pad_to_same_shape
import numpy as np
from validation.test_parser import define_model_parser
from validation.utils import matches_from_flow
import argparse
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import yaml

import tkinter as tk
from tkinter import filedialog

import matplotlib.cm as cm


def test_model_on_image_pair(args, query_image, reference_image,contact_pixels = None, visualize = True):
    with torch.no_grad():
        network, estimate_uncertainty = select_model(
            args.model, args.pre_trained_model, args, args.optim_iter, local_optim_iter,
            path_to_pre_trained_models=args.path_to_pre_trained_models)

        # save original ref image shape
        ref_image_shape = reference_image.shape[:2]

        # pad both images to the same size, to be processed by network
        query_image_, reference_image_ = pad_to_same_shape(query_image, reference_image)
        # convert numpy to torch tensor and put it in right format
        query_image_ = torch.from_numpy(query_image_).permute(2, 0, 1).unsqueeze(0)
        reference_image_ = torch.from_numpy(reference_image_).permute(2, 0, 1).unsqueeze(0)

        # ATTENTION, here source and target images are Torch tensors of size 1x3xHxW, without further pre-processing
        # specific pre-processing (/255 and rescaling) are done within the function.

        # pass both images to the network, it will pre-process the images and ouput the estimated flow
        # in dimension 1x2xHxW
        if estimate_uncertainty:
            if args.flipping_condition:
                raise NotImplementedError('No flipping condition with PDC-Net for now')

            estimated_flow, uncertainty_components = network.estimate_flow_and_confidence_map(query_image_,
                                                                                              reference_image_,
                                                                                              mode='channel_first')
            confidence_map = uncertainty_components['p_r'].squeeze().detach().cpu().numpy()
            confidence_map = confidence_map[:ref_image_shape[0], :ref_image_shape[1]]
        else:
            if args.flipping_condition and 'GLUNet' in args.model:
                estimated_flow = network.estimate_flow_with_flipping_condition(query_image_, reference_image_,
                                                                               mode='channel_first')
            else:
                estimated_flow = network.estimate_flow(query_image_, reference_image_, mode='channel_first')
        estimated_flow_numpy = estimated_flow.squeeze().permute(1, 2, 0).cpu().numpy()
        estimated_flow_numpy = estimated_flow_numpy[:ref_image_shape[0], :ref_image_shape[1]]
        # removes the padding

        warped_query_image = remap_using_flow_fields(query_image, estimated_flow_numpy[:, :, 0],
                                                     estimated_flow_numpy[:, :, 1]).astype(np.uint8)

        if estimate_uncertainty:
            contact_pts_matching, query_pts = match_contact_points(query_image, reference_image,estimated_flow,contact_pixels)
            color = [255, 102, 51]
            fig, axis = plt.subplots(1, 5, figsize=(30, 30))

            confident_mask = (confidence_map > 0.50).astype(np.uint8)
            confident_warped = overlay_semantic_mask(warped_query_image, ann=255 - confident_mask*255, color=color)
            axis[2].imshow(confident_warped)
            axis[2].set_title('Confident warped query image according to \n estimated flow by {}_{}'
                              .format(args.model, args.pre_trained_model))
            axis[4].imshow(confidence_map, vmin=0.0, vmax=1.0)
            axis[4].set_title('Confident regions')
        else:
            contact_pts_matching, query_pts = match_contact_points(query_image, reference_image,estimated_flow,contact_pixels)
            fig, axis = plt.subplots(1, 5, figsize=(30, 30))
            axis[2].imshow(warped_query_image)
            axis[2].set_title(
                'Warped query image according to estimated flow by {}_{}'.format(args.model, args.pre_trained_model))
        axis[0].imshow(query_image)
        axis[0].set_title('Query image')
        axis[1].imshow(reference_image)
        axis[1].set_title('Reference image')

        axis[3].imshow(flow_to_image(estimated_flow_numpy))
        axis[3].set_title('Estimated flow {}_{}'.format(args.model, args.pre_trained_model))

        axis[-1].imshow(contact_pts_matching)
        axis[-1].set_title('Contact points matching')
        
        if visualize:
            plt.show()
            
        plt.close(fig)
        return estimated_flow,query_pts

def match_contact_points(query_image,reference_image,estimated_flow,contact_pixels):
    mask = np.zeros(estimated_flow.shape[-2:], dtype=int)[np.newaxis, ...]
    mask_indices = contact_pixels.astype(int)[:, [1, 0]]  # (x,y) -> (row, col)
    mask[:, mask_indices[:, 0], mask_indices[:, 1]] = 1
    mask = torch.tensor(mask, device=estimated_flow.device) == 1
    # print(estimated_flow.shape)
    # print(mask.shape)
    mkpts_q, mkpts_r = matches_from_flow(estimated_flow, mask)
    # print(mkpts_q)
    # print(mkpts_r)

    confidence_values = np.ones(mkpts_q.shape[0])
    import matplotlib.cm as cm
    color = cm.jet(confidence_values)
    out = make_sparse_matching_plot(
        query_image, reference_image, mkpts_q, mkpts_r, color, margin=10)

    # plt.figure(figsize=(16, 8))
    # plt.imshow(out)
    # plt.show()
    return out,mkpts_q

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

    with open("/home/paolo/Documents/Datasets/utilities/annotated_trajectory.yaml", "r") as f:
        yaml_data = yaml.load(f, Loader=yaml.FullLoader)

    # Load the reference image
    reference_image_path = yaml_data["image_path"]
    reference_image = cv2.imread(reference_image_path, 1)[:, :, ::- 1]

    # Load the fitted polynomial coefficients
    x_fit = yaml_data["x_fit"]
    y_fit = yaml_data["y_fit"]

    # Load the contact points
    contact_points = []
    for i in range(len(x_fit)):
        point = (x_fit[i], y_fit[i])
        contact_points.append(point)
    contact_pixels = np.array(contact_points)

    # Display the uploaded image
    cv2.imshow('Reference image', reference_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Create a Tkinter window to prompt the user to select a file
    root = tk.Tk()
    root.withdraw()
    # Ask the user to select a depth image file
    file_path = filedialog.askopenfilename(title='Select a target image')

    query_image = cv2.imread(file_path, 1)[:, :, ::- 1]

    # Display the uploaded image
    cv2.imshow('Uploaded image', cv2.cvtColor(query_image, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    estimated_flow,query_pts = test_model_on_image_pair(args, query_image, reference_image,
                                                       contact_pixels = contact_pixels, visualize= False)
    
    confidence_values = np.ones(query_pts.shape[0])
    color = cm.jet(confidence_values)
    matching = make_sparse_matching_plot(
                query_image, reference_image, query_pts, contact_pixels, color, margin=10)
    
    plt.figure(figsize=(16, 8))
    plt.imshow(matching)
    plt.axis('off')

    #plt.savefig(os.path.join(save_dir_new,query_img_name + '_matching.png'))
    plt.show()

    reference_trajectory = draw_keypoints(reference_image, contact_pixels)
    target_trajectory = draw_keypoints(query_image,query_pts)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(reference_trajectory)
    ax1.set_title("Reference image with annotated segment")
    ax2.imshow(target_trajectory)
    ax2.set_title("Target image with transferred segment")
    fig.suptitle("Objects' parts matching with semantic correspondence methods")
    plt.show()