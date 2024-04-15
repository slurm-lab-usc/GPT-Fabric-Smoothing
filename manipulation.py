import os
import os.path as osp
import argparse
import time

import base64
import re
import requests
from openai import OpenAI

import numpy as np
import cv2 as cv
import datetime

from abc import ABC, abstractmethod

from softgym.registered_env import env_arg_dict, SOFTGYM_ENVS
from softgym.utils.normalized_env import normalize
from softgym.utils.visualization import save_numpy_as_gif
from softgym.utils import camera_utils

import pyflex
from matplotlib import pyplot as plt
from PIL import Image


# Set up the API key for GPT-4 communication
with open("GPT-API-Key.txt", "r") as f:
    api_key = f.read().strip()

client = OpenAI(api_key=api_key)

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}


# Function to encode the image to base64 for input to the GPT-4 API
def encode_image(image_path):
    """
    Encode the image to base64 for input to the GPT-4 API.
    Input:
        image_path: the path of the image
    Output:
        the base64 encoding of the image
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    

    
class manipulation():
    """
    The base class for GPT-Fabric smoothing manipulation.
    Attributes:
        env: the softgym environment (normally it will only be "ClothFlattenGPTRGB" in the ./softgym/registered_env.py file)
        env_name: the name of the softgym environment
        obs_dir: the directory to save the observations and all the information
        goal_image: the goal image (flattened fabric image, use softgym to set the fabric to be flatten and then save the image)
        goal_config: whether to use the goal image (True or False) for the manipulation
    """
    def __init__(self,env,env_name,obs_dir,goal_image,goal_config):
        self.env=env
        self.env_name=env_name
        self.obs_dir=obs_dir
        self.goal_image=goal_image
        self.goal_config=goal_config

        
    def save_obs(self):
        return NotImplementedError
    
    def single_step(self):
        return NotImplementedError
    
    def picker_step(self,target_pos,pick=False,record=True):
        """
        The function for moving the picker to the target position.
        Input:
            target_pos: the target position
            pick: whether to pick (hold) the object
            record: whether to record the continuous video
        """
        curr_picker_pos=self.env.action_tool.get_picker_pos().squeeze()
        target_pos=target_pos.squeeze()
        picker_translation=(target_pos-curr_picker_pos)
        picker_action=np.append(picker_translation,1 if pick else 0)
        
        # print(picker_action)
        
        _, _, _, info=self.env.step(picker_action,record_continuous_video=True,img_size=self.img_size)
        if record:
            return info['flex_env_recorded_frames']
        else:
            return None
    
    def get_pick_place_point_manual(self):
        """
        Should be deprecated in the future.
        """
        print("------------------------------------------------------------ \n")
        picker_pos=self.env.action_tool.get_picker_pos().squeeze()
        print("current picker's position is : {} \n".format(str(picker_pos)))            
        print("The desired input is x,z,y")
        user_input=input("List the place to pick using  \',\' as specifier:")
        print("\n")
        pick_point = user_input.split(",")
        pick_point= np.array(pick_point, dtype=float)
        
        user_input=input("List the place to place using  \',\' as specifier:")
        print("\n")
        place_point = user_input.split(",")
        place_point=np.array(place_point, dtype=float)
        
        return pick_point,place_point
        
        
class RGB_manipulation(manipulation):
    """
    Defines the basic.
    Added Attributes:
        img_size: the size of the image (normally it will be 720x720)
        goal_depth: the depth image of the goal image
        goal_height: The height of the fabric in the goal image (in pixel)
        goal_width: The width of the fabric in the goal image (in pixel)
        We use the goal_height and goal_width to help discretize the moving distance.

    """
    def __init__(self,env,env_name,obs_dir,goal_image,goal_config,goal_depth,img_size):
        super().__init__(env,env_name,obs_dir,goal_image,goal_config)
        self.img_size=img_size
        self.goal_depth=goal_depth
        top,bottom,left,right=self.get_bounds(self.goal_depth)
        self.goal_height=top-bottom
        self.goal_width=right-left
    
    def save_obs(self,obs,depth=False,specifier="init"):
        """
        Save the observation (Raw and processed top-down image) to the observation directory.
        Input:
            obs: the observation
            depth: whether to save the depth image
            specifier: the specifier (usually should be the number of step) of the observation.
        Output:
            save_path: the path of the saved image
        """
        save_name = osp.join(self.obs_dir, self.env_name)
        save_image = Image.fromarray(obs)
        save_path=save_name+'_'+specifier+'.png'
        save_image.save(save_path)  
        print('observation save to {} \n'.format(save_name))
        
        return save_path
        
    
    def get_corners_img(self,img_path,depth,specifier,corner_limit=10):
        """
        Get the corners of the fabric in the image and save the image with corners marked.
        Input:
            img_path: the path of the image
            specifier: the specifier (usually should be the number of step) of the observation.
            corner_limit: the limit of the number of corners to be detected.
        Output:
            new_corners: the corners detected in the image (in the format of np array with shape [corner limit, 2]: [x1, y1], [x2, y2], ...)
            img: the image with corners marked
        """
        
        img = cv.imread(img_path)
        img_copy=img.copy()
        
        ########################################################
        # Turn the backside of the fabric into white to find more accurate corners
        lower_pink = np.array([140, 50, 75])
        upper_pink = np.array([170, 255, 255])

        # Convert the image from BGR to HSV color space
        hsv_image = cv.cvtColor(img, cv.COLOR_BGR2HSV)

        # Create a mask for pink color
        mask = cv.inRange(hsv_image, lower_pink, upper_pink)

        # Define the blue color you want to change to
        blue_color = np.array([255, 255, 255], dtype=np.uint8)

        # Change the pink areas to blue
        img[mask != 0] = blue_color
        
        img = cv.cvtColor(img, cv.COLOR_HSV2RGB)
        ############################################################
        
        
        
        gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        
        corners = cv.goodFeaturesToTrack(gray,corner_limit,0.1,20)
        corners = np.int0(corners)# shape: corner_limit, 1 , 2
        
        new_corners=np.squeeze(corners,axis=1)
        
        save_name=osp.join(self.obs_dir,specifier)# specifier should be used for step count e.g.: specifier="step_1_"
        with open(save_name+"_corners.txt", "w+") as p:
            p.write(f"The corners are :{new_corners}")   

            for i in corners:
                x,y = i.ravel()
                cv.circle(img_copy,(x,y),3,255,-1)

            
        cv.imwrite(img_path,img_copy)
        
        
        
        return new_corners,img
    
    def map_depth_to_image(self, depth_array, output_range=(0, 255),RGB=True):
        
        """
        Generate an image from the depth array.
        Input:
            depth_array: the depth array
            output_range: the range of the output image
            RGB: whether to output the image in RGB format
        Output:
            output_image: the image generated from the depth array
        """
        
        non_zero_mask=depth_array!=0
        non_zero_elements = depth_array[non_zero_mask]

        # Find the minimum among the non-zero elements
        max_depth = non_zero_elements.max()
        min_depth = non_zero_elements.min()
                
        min_output, max_output = output_range
        
        # Map the depth array to the output range
        scale = (max_output - min_output) / (max_depth - min_depth)
        output_array = (depth_array - min_depth) * scale + min_output
        
        # Ensure the output is within the bounds [0, 255]
        output_array_clipped = np.clip(output_array, min_output, max_output)
        
        # Convert to unsigned 8-bit integer type as required for image pixel values
        output_image = np.uint8(output_array_clipped)
        
        
        if RGB:
            output_image= np.stack((output_image, output_image, output_image), axis=-1)

        return output_image

    def aug_background(self,img,depth,color=[40,40,40]):
        """
        Turn the background (we use depth image to define) of the fabric into a specific color (default to be black) 
        for better corner detection result.
        Input:
            img: the image of the fabric
            depth: the depth image of the fabric
            color: what color should the background be turned into
        Output:
            img: the image with the background turned into the specific color
        """
        mask = depth == 0    
        for c in range(3):  # Loop through each color channel
            img[:, :, c][mask] = color[c]
                            
                        
        return img
    
    def get_bounds(self,depth):
        """
        Get the bounds of the fabric with the help of the depth image.
        Input:
            depth: the depth image of the fabric
        Output:
            top: the top bound of the fabric
            bottom: the bottom bound of the fabric
            left: the left bound of the fabric
            right: the right bound of the fabric
        """
        locations=np.nonzero(depth)
        top=max(locations[0])
        bottom=min(locations[0])
        
        left=min(locations[1])
        right=max(locations[1])
        
        return top,bottom,left,right
    
    def vis_result(self,place_pixel,pick_pixel=None,img_path=None,img=None):
        """
        Visualize the result of the pick-and-place action.
        If provide both place pixel and pick pixel, draw a circle at the pick pixel and an arrow pointing to the place pixel.
        If only provide place pixel, draw a circle at the place pixel (This is to visualize the action from last step).
        
        Input:
            place_pixel: the pixel coordinate of the placing point
            pick_pixel: the pixel coordinate of the picking point
            img_path: the path of the image (If no image is provided, use the img_path to load the image)
            img: the image to be visualized
        Output:
            img: the image with the pick-and-place action visualized
        """
        
        if img_path:
            img=cv.imread(img_path)
        if pick_pixel is not None:
            # Visualize the pick-and-place action
            cv.circle(img, (int(pick_pixel[0]), int(pick_pixel[1])), 5, (0, 255, 0), 2)
            cv.arrowedLine(img, (int(pick_pixel[0]), int(pick_pixel[1])), (int(place_pixel[0]), int(place_pixel[1])), (0, 255, 0), 2)
            cv.circle(img,(int(place_pixel[0]), int(place_pixel[1])), 5, (128, 0, 128), 2)
        else:
            # Visulize a designated place point
            cv.circle(img, (int(place_pixel[0]), int(place_pixel[1])), 3, (0, 0, 255), 2)
        return img
         
    
    def get_picking_point(self,img, corners, gpt_reasoning=False,headers=None,system_prompt_path=None):
        """
        Get the picking point from the detected corners
        
        We deprecated the gpt_reasoning part in this function as we think for random actions,
        the picking point selection from GPT model is not necessary.
        Input:
            img: the image of the fabric
            corners: the corners detected in the image
            gpt_reasoning: whether the picking point is determined by the gpt model
        Output:
            pick_point: the picking point
        """
        if gpt_reasoning:
            # the picking point is determined by the gpt model
            pass
            return NotImplementedError
        else:
            # the picking point is randomly selected from the corners detected
            pick_point = corners[np.random.choice(corners.shape[0], 1, replace=False)]
            pick_point=pick_point.squeeze() 

        print(pick_point.shape)
        
        return pick_point
    
    def random_step(self,
                    headers=None, 
                    frames=[],
                    system_prompt_path=None,
                    aug_background=True,
                    gpt_reasoning=False,
                    direction_seg=8,
                    distance_seg=4,
                    corner_limit=10,                     
                    specifier="init"):
        """
        This function is to choose a point from the detected corners randomly 
        and then choose a direction and distance randomly. We use it for ablation.
        Input:
            headers: the headers for the GPT model
            frames: the recorded frames of the previous manipulation
            system_prompt_path: the path of the system prompt (deprecated)
            aug_background: whether to augment the background (default to be True)
            gpt_reasoning: whether to use the GPT model to reason the picking point (deprecated, default to be false)
            direction_seg: the number of segments for the direction
            distance_seg: the number of segments for the distance
            corner_limit: the limit of the number of corners to be detected
            specifier: the specifier (usually should be the number of step) of the observation.
        Output:
            frames: the recorded frames of the manipulation (including the frames of the previous manipulation)
            test_improvement: the Normalized improvement of the coverage after the manipulation (comparing to the starting config)
            last_coverage: the coverage before the manipulation
            test_coverage: the coverage after the manipulation
        """
        
        self._specifier=specifier
        default_pos=np.array([0.0,0.2,0.0]).squeeze()
        operation_height=0.1
        
        self.paths={
            
            "raw image":osp.join(self.obs_dir,"raw_image_"+specifier+".png"),
            "raw depth":osp.join(self.obs_dir,"raw_depth_image_"+specifier+".png"),
            "processed image":osp.join(self.obs_dir,"processed_image_"+specifier+".png"),
            "processed depth":osp.join(self.obs_dir,"processed_depth_image_"+specifier+".png"),
            "raw vis image":osp.join(self.obs_dir,"Raw_vis_result_"+specifier+".png"),
            "raw vis depth":osp.join(self.obs_dir,"Raw_vis_result_depth_"+specifier+".png"),
            "processed vis image":osp.join(self.obs_dir,"Vis_result_"+specifier+".png"),
            "processed vis depth":osp.join(self.obs_dir,"Vis_result_depth_"+specifier+".png"),
            
        }
                        
        # step 0.a : get and save obs before interaction:
        obs=self.env.get_rgbd()
        
        image=(obs[:,:,:-1]*255).astype(np.uint8)
        image_raw=Image.fromarray(image)
        image_raw.save(self.paths['raw image'])# Raw image (image 1)
        
        depth=np.round(obs[:,:,3:].squeeze(),3)
        self.depth=depth
        self.depth_image=self.map_depth_to_image(depth_array=depth)
        raw_depth_image=Image.fromarray(self.depth_image)
        raw_depth_image.save(self.paths['raw depth'])# Raw depth (image 2)
        
        self.pixel_coords=camera_utils.get_world_coords(rgb=image,depth=depth,env=self.env)[:,:,:-1]

        
        # step 0.b : to process the image and depth image
        
        if aug_background:
            image=self.aug_background(image,depth)
              
        img_path=self.save_obs(obs=image,specifier=specifier)

        
        # step 0.c: get the corners on the image as well as the corner coordinates
        corners,img=self.get_corners_img(img_path=img_path,depth=depth,specifier=specifier,corner_limit=corner_limit)
        
        # step 1: get the picking point from the detected corners (if gpt_reasoning is True, the picking point is determined by the gpt model)
        # Also, randomly generate the moving direction and distance
        
        picking_pixel=self.get_picking_point(img=img,corners=corners,gpt_reasoning=gpt_reasoning,headers=headers)
        
        moving_direction=np.random.choice(np.arange(0,2,2/direction_seg))
        
        moving_distance=np.random.choice(np.array([0.1,0.25,0.5,0.75,1.0]))
        
        # Transform the pixel into the world coordinates for manipulation
        pick_coords=camera_utils.find_nearest(self.pixel_coords,picking_pixel[1],picking_pixel[0])
        pick_coords=self.pixel_coords[pick_coords[0]][pick_coords[1]]
        
        
        curr_config=self.env.get_current_config()
        dimx,dimy=curr_config['ClothSize']
        size=max(dimx,dimy)*self.env.cloth_particle_radius # Get the fabric side length in the world coordinates

        actual_direction=moving_direction*np.pi
        actual_distance=moving_distance*size

        delta_x=actual_distance*np.sin(actual_direction)
        delta_y=actual_distance*np.cos(actual_direction)

        # Generate the placing coordinates
        place_coords = pick_coords.copy()
        place_coords[0]+=delta_x
        place_coords[2]+=delta_y
        
        
        # Generate the placing pixel (for visualization)
        pixel_size=max(self.goal_height,self.goal_width)
        delta_x_pixel=int(pixel_size*np.cos(actual_direction)*moving_distance)
        delta_y_pixel=int(pixel_size*np.sin(actual_direction)*moving_distance)
        
        place_pixel=[picking_pixel[0]+delta_x_pixel,picking_pixel[1]-delta_y_pixel]
        
        
        # Visualize the pick-and-place action (before the actual interaction)
        img=self.vis_result(place_pixel=place_pixel,pick_pixel=picking_pixel,img=img)
        cv.imwrite(self.paths['processed vis image'],img)
        
        
        # step 2: get the coverage before the interaction:
        info=self.env._get_info()
        improvement=info["normalized_performance"]
        improvement=np.round(improvement,3)
        last_coverage=info["normalized_performance_2"]
        last_coverage=np.round(last_coverage,3)
        
        # step 3: get the action sequence and record the frames:
        # pre_pick_pos: the position before picking (slightly higher than the chosen picking point)
        pre_pick_pos=pick_coords.copy()
        pre_pick_pos[1]+=self.env.action_tool.picker_radius*2
        
        # after_pick_pos: the position after picking (Drag the fabric into the operation height)
        after_pick_pos=pick_coords.copy()
        after_pick_pos[1]=operation_height
        
        # pre_place_pos: the position before placing (the x,y coordiantion is the same as the placing point and is on the operation height)
        pre_place_pos=place_coords.copy()
        pre_place_pos[1]=operation_height
          
        
        action_sequence=[[pre_pick_pos, False],
                                [pick_coords, True],
                                [after_pick_pos, True],
                                [pre_place_pos, True],
                                [pre_place_pos, False],
                                [default_pos, False]]
        
        # Action rollout
        for action in action_sequence:
            frames.extend(self.picker_step(target_pos=action[0],pick=action[1]))
        
        # step 4: get the coverage after the interaction:
        info=self.env._get_info()        
        test_improvement=info["normalized_performance"]
        test_coverage=info["normalized_performance_2"]
        
        
        return frames,np.round(test_improvement,3),np.round(last_coverage,3),np.round(test_coverage,3)
    
