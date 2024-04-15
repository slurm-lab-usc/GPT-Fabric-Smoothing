import os
import os.path as osp
import argparse
import time
import json
import base64
import re
import requests
import csv
from openai import OpenAI

import numpy as np

import datetime

from abc import ABC, abstractmethod

from softgym.registered_env import env_arg_dict, SOFTGYM_ENVS
from softgym.utils.normalized_env import normalize
from softgym.utils import camera_utils
from softgym.utils.visualization import save_numpy_as_gif
import pyflex
from matplotlib import pyplot as plt
from PIL import Image


from RGBD_manipulation import RGBD_manipulation_part_obs
from manipulation import encode_image,RGB_manipulation



with open("GPT-API-Key.txt", "r") as f:
    api_key = f.read().strip()

client = OpenAI(api_key=api_key)

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}



def main():
    """
    This is the main function for running the cached config tests. 
    We use this script to test the performance of different methods on the same set of initial states.
    It's recommened to use this script to test the performance of the methods on the cached initial states to see the performance.
    """
    
    # 0. set the parameters
    parser = argparse.ArgumentParser(description='Process some integers.')
    # ['ClothFlattenGPTRGB','ClothFlattenGPTPC','PassWater', 'PourWater', 'PourWaterAmount', 'RopeFlatten', 'ClothFold', 'ClothFlatten', 'ClothDrop', 'ClothFoldCrumpled', 'ClothFoldDrop', 'RopeConfiguration']
    parser.add_argument('--env_name', type=str, default='ClothFlattenGPTRGB')
    parser.add_argument('--cache_state_path',type=str,default='/cloth_flatten_states_40_test')
    parser.add_argument('--headless', type=int, default=1, help='Whether to run the environment with headless rendering')
    parser.add_argument('--num_variations', type=int, default=10, help='Number of environment variations to be generated')
    parser.add_argument('--save_obs_dir', type=str, default='./10_env_tests', help='Path to the saved observation')    
    parser.add_argument('--save_image_dir', type=str, default='states/images_RGBD', help='Path to the saved video')
    parser.add_argument('--img_size', type=int, default=720, help='Size of the recorded videos')
    
    
    
    parser.add_argument('--method_name',type=str,default='RGBD_simple')
    parser.add_argument('--direction_seg',type=int, default=8, help='The number of discretized directions')
    parser.add_argument('--distance_seg',type=int, default=4, help='The number of discretized distance, which are times of fabric side length')
    parser.add_argument('--trails', type=int, default=5, help='The maximum step the interaction can take')
    parser.add_argument('--gif_speed',type=int, default=4, help="This is the speed of gif file. At least 1")
    parser.add_argument('--goal_config',type=int,default=0,help="This is switch of telling the gpt model whether to use goal configuration as a part of system prompt")
    parser.add_argument('--reps',type=int, default=1, help="how many repetitive results we can get from one config")
    parser.add_argument('--starting_config',type=int,default=0,help="which config to start")

    args = parser.parse_args()
    
    # 0.1 set the method based on the method name
    methods={
        # The method proposed in the paper
        "RGBD_simple":{
            "env_name":"ClothFlattenGPTRGB",
            "need_box":True,
            "goal_config":False,
            "system_prompt_path":"system_prompts/RGBD_prompt.txt",
            "img_size":720,
            "corner_limit":15,            
        },
        # With the depth reasoning, deprecated
        "RGBD_depth_reasoning":{
            "env_name":"ClothFlattenGPTRGB",
            "need_box":True,
            "goal_config":False,
            "depth_reasoning":True,
            "system_prompt_path":"system_prompts/RGBD_prompt_depth_reasoning.txt",
            "img_size":720,
            "corner_limit":15, 
        },
        # Add ICL to the method
        "RGBD_ICL":{
            "env_name":"ClothFlattenGPTRGB",
            "need_box":True,
            "goal_config":False,
            "system_prompt_path":"system_prompts/RGBD_prompt.txt",
            "img_size":720,
            "corner_limit":15,
            "in_context_learning":True,
            "demo_dir":"./demo/Manual_test_14",            
        },
        # Remove the image preprocessing module and the evaluation module
        "RGBD_naive":{
            "env_name":"ClothFlattenGPTRGB",
            "need_box":False,
            "goal_config":False,
            "system_prompt_path":"system_prompts/RGBD_naive_prompt.txt",
            "img_size":720,
            "corner_limit":15,
            "in_context_learning":False,
            "demo_dir":"./demo/Manual_test_14",
            "naive":True, 
            
        },
        # Remove the evaluation module
        "RGBD_no_recon":{
            "env_name":"ClothFlattenGPTRGB",
            "need_box":False,
            "goal_config":False,
            "system_prompt_path":"system_prompts/RGBD_prompt.txt",
            "img_size":720,
            "corner_limit":15,
            "in_context_learning":False,
            "demo_dir":"./demo/Manual_test_14",
            "re_consider" :False,
            
        },
        # Remove the picking point approximity check from the evaluation module
        "RGBD_no_last_point":{
            "env_name":"ClothFlattenGPTRGB",
            "need_box":False,
            "goal_config":False,
            "system_prompt_path":"system_prompts/RGBD_prompt.txt",
            "img_size":720,
            "corner_limit":15,
        },
        # Totally random method. No GPT reasoning
        "RGBD_total_random":{
            "env_name":"ClothFlattenGPTRGB",
            "need_box":False,
            "goal_config":False,
            "gpt_reasoning":False,
            "system_prompt_path":"system_prompts/RGBD_prompt.txt",
            "img_size":720,
            "corner_limit":15,
            
        },
        
    }
    method=methods[args.method_name]
    
    # 0.2 set the environment
    env_kwargs = env_arg_dict[args.env_name]
    # Generate and save the initial states for running this environment for the first time
    env_kwargs['use_cached_states'] = True
    env_kwargs['save_cached_states'] = False
    env_kwargs['num_variations'] = args.num_variations
    env_kwargs['render'] = True
    env_kwargs['headless'] = args.headless

    if not env_kwargs['use_cached_states']:
        print('Waiting to generate environment variations. May take 1 minute for each variation...')
    else:
        print("using cached states")
    
     
    cur_dir = osp.dirname(osp.abspath(__file__))  
    cache_state_path = cur_dir+args.cache_state_path
        
    states_path=osp.join(cache_state_path,"states.pkl")
    env_kwargs['cached_states_path']=states_path
    env = normalize(SOFTGYM_ENVS[args.env_name](**env_kwargs))
    env.reset()
    
    
    # 0.3 set the paramters for manipulation based on method configuration
    save_obs_dir=osp.join(args.save_obs_dir,args.method_name)
    
    need_box=method['need_box'] if 'need_box' in method else False
    depth_reasoning=method['depth_reasoning'] if "depth_reasoning" in method else False
    memory=method['memory'] if "memory" in method else False
    in_context_learning=method['in_context_learning'] if "in_context_learning" in method else False
    goal_config=method['goal_config'] if "goal_config" in method else False
    system_prompt_path=method['system_prompt_path'] if "system_prompt_path" in method else "system_prompts/RGBD_prompt.txt"
    demo_dir=method['demo_dir'] if "demo_dir" in method else None
    img_size=method['img_size'] if "img_size" in method else 720
    fine_tuning=method["fine_tuning"] if "fine_tuning" in method else False
    fine_tuning_model_path=method["fine_tuning_model_path"] if "fine_tuning_model_path" in method else None
    corner_limit=method['corner_limit'] if 'corner_limit' in method else 10
    naive=method['naive'] if 'naive' in method else False
    re_consider=method['re_consider'] if 're_consider' in method else True
    gpt_reasoning=method['gpt_reasoning'] if 'gpt_reasoning' in method else True
    
    # 1. start the test
    for i in range(args.starting_config,env.num_variations):
        # Record both the highest coverage and the final coverage. We report the final coverages.
        highest_coverages=[]
        final_coverages=[]
        for j in range(args.reps):
            # get the goal_config
            env.reset(config_id=i)
            env._set_to_flat()
            env.action_tool.hide()
            goal_image=env.get_image(method["img_size"],method["img_size"])
            
            
            #reset the state to the i_th config  
            env.reset(config_id=i)
            if env_kwargs['save_cached_states']:
                env.action_tool.hide()
                time.sleep(2)
                state_image=env.get_image(img_size,img_size)
                image_path = osp.join(cache_state_path, "state_images/state_")
                image_path=image_path+str(i)+'.png'
                state_image=Image.fromarray(state_image)
                state_image.save(image_path)
                env.action_tool.show()

                   
                    

            save_obs_dir_env_main=osp.join(save_obs_dir,f"state_{str(i)}") # The folder where each rep results of the same starting config are saved
            save_obs_dir_env=osp.join(save_obs_dir_env_main,f"rep_{str(j)}") # The folder where the results of j-th rep are saved
            
            if not os.path.exists(save_obs_dir_env):
                os.makedirs(save_obs_dir_env)
                print(f"Directory created at {save_obs_dir_env}\n")
            else:
                print(f"Directory already exists at {save_obs_dir_env}, content there will be update\n")
            
            # Get the goal depth and goal image
            env.reset(config_id=i)
            env._set_to_flat()
            env.action_tool.hide()
            goal_image=env.get_image(img_size,img_size)
            goal_depth=env.get_rgbd()
            goal_depth=np.round(goal_depth[:,:,3:].squeeze(),3)
            save_path=osp.join(save_obs_dir_env,'flatten.png')
            save_image = Image.fromarray(goal_image)
            save_image.save(save_path)   
            goal_image=encode_image(save_path)
            env.action_tool.show()
            env.reset(config_id=i)
            
            
            
            
            # 1.1 start the manipulation
            frames = [env.get_image(img_size, img_size)]
            coverages=[]
            # 1.2 set the manipulation method
            if "random" in args.method_name:
                manipulation=RGB_manipulation(
                env=env,
                env_name=method["env_name"],
                obs_dir=save_obs_dir_env,
                goal_image=goal_image,
                goal_config=goal_config,
                goal_depth=goal_depth,

                img_size=img_size      
                )
                
                
            else:
                manipulation=RGBD_manipulation_part_obs(
                env=env,
                env_name=method["env_name"],
                obs_dir=save_obs_dir_env,
                goal_image=goal_image,
                goal_config=goal_config,
                goal_depth=goal_depth,
                img_size=img_size,
                in_context_learning=in_context_learning,
                demo_dir=demo_dir,
                re_consider=False if naive else re_consider,
                
            )
                
            # 1.3 start the manipulation with args.trails steps 
            messages=[]
            last_step_info=None
            step=0
            improvement=0
            while step <args.trails and improvement<0.95:
                step+=1
                
                if "random" in args.method_name:
                    frames,improvement,coverage,new_coverage=manipulation.random_step(headers=headers,
                                            frames=frames,
                                            system_prompt_path=system_prompt_path,
                                            aug_background=True,
                                            gpt_reasoning=gpt_reasoning,
                                            corner_limit=corner_limit,
                                            specifier="step"+str(step))
                
                else:
                    frames,messages,last_step_info,improvement,coverage,new_coverage=manipulation.gpt_single_step(headers=headers,
                                                frames=frames,
                                                messages=messages,
                                                system_prompt_path=system_prompt_path,
                                                memory=memory,
                                                need_box=need_box,
                                                aug_background= False if naive else True,
                                                corner_limit=corner_limit,
                                                last_step_info=None if naive else last_step_info ,
                                                depth_reasoning=depth_reasoning,
                                                direction_seg=args.direction_seg,
                                                distance_seg=args.distance_seg,
                                                specifier="step"+str(step))
                
                    
                    json_save_path=osp.join(save_obs_dir_env,"message_step"+str(step)+".jsonl")
                    with open(json_save_path,'w+') as file:
                        # for entry in data:
                        json_string=json.dumps(messages)
                        file.write(json_string+'\n')
                
                # 1.4 save the coverage and improvement of this step
                coverages.append([new_coverage,improvement])
                

                # if save_obs_dir_env is not None:
                #     save_name = osp.join(save_obs_dir_env, args.env_name + '.gif')
                #     save_numpy_as_gif(np.array(frames), save_name)
                    
                print('finish step {}'.format(str(step+1)))
                print(f'current coverage is {new_coverage}, improvement is {improvement}\n')
                print('--------------------------------------------------------\n')

                
            # Record the best and final coverage of this rep (episode)
            best_res=max(coverages, key=lambda x: x[1])
            final_res=coverages[-1]
            
            # Record the result of this rep and concat with the results of other reps under same starting config
            highest_coverages.append(best_res)
            final_coverages.append(final_res)
            
            if save_obs_dir_env is not None:
                save_name = osp.join(save_obs_dir_env, args.env_name + '.gif')
                if args.gif_speed>1:
                    frames=frames[::args.gif_speed]

                save_numpy_as_gif(np.array(frames), save_name)
                print('Video generated and save to {}'.format(save_name))
                
                coverage_message_path=osp.join(save_obs_dir_env,"coverages.csv")
                with open(coverage_message_path,"w+",newline='') as file:
                    writer=csv.writer(file)
                    writer.writerows(coverages)
                print('coverage message generated and save to {}'.format(coverage_message_path))
                
        coverage_message_path_best=osp.join(save_obs_dir_env_main,"coverages_best.csv")

        norm_coverage= [item[0] for item in highest_coverages]
        norm_improvements = [item[1] for item in highest_coverages]

        # Calculate the mean among the reps under same starting config
        mean_value_0 = np.mean(norm_coverage)
        mean_value_1 = np.mean(norm_improvements)
        highest_coverages.append([mean_value_0,mean_value_1])

        with open(coverage_message_path_best,"w+",newline='') as file:
            writer=csv.writer(file)
            writer.writerows(highest_coverages)
        print('coverage message generated and save to {}'.format(coverage_message_path_best))
        
        
        
        coverage_message_path_final=osp.join(save_obs_dir_env_main,"coverages_final.csv")
        
        norm_coverage= [item[0] for item in final_coverages]
        norm_improvements = [item[1] for item in final_coverages]

        # Calculate the mean and standard deviation
        mean_value_0 = np.mean(norm_coverage)
        mean_value_1 = np.mean(norm_improvements)
        final_coverages.append([mean_value_0,mean_value_1])

        with open(coverage_message_path_final,"w+",newline='') as file:
            writer=csv.writer(file)
            writer.writerows(final_coverages)
        print('coverage message generated and save to {}'.format(coverage_message_path_final))

if __name__ == '__main__':
    main()

