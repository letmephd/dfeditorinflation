from src.demo.download import download_all
#download_all()

from src.demo.demo import create_demo_move, create_demo_appearance, create_demo_drag, create_demo_face_drag, create_demo_paste
from src.demo.model import DragonModels
from src.demo.model_3d import DragonModels_3d
from src.demo.utils import get_point, store_img, get_point_move, store_img_move, clear_points, upload_image_move, segment_with_points, segment_with_points_paste, fun_clear, paste_with_mask_and_offset, get_point_move_new,segment_with_points_new
from PIL import Image
import numpy as np
import tqdm
import os
import glob

import cv2
import os
os.environ["no_proxy"] = "localhost,127.0.0.1,::1"
import gradio as gr

# main demo
pretrained_model_path = "/data/yinzijin/checkpoints/stable-diffusion/stable-diffusion-v1-5"
model = DragonModels_3d(pretrained_model_path=pretrained_model_path)

DESCRIPTION = '# 🐉🐉[DragonDiffusion V1.0](https://github.com/MC-E/DragonDiffusion)🐉🐉'

DESCRIPTION += f'<p>Gradio demo for [DragonDiffusion](https://arxiv.org/abs/2307.02421) and [DiffEditor](https://arxiv.org/abs/2307.02421). If it is helpful, please help to recommend [[GitHub Repo]](https://github.com/MC-E/DragonDiffusion) to your friends 😊 </p>'

seed = 42
resize_scale = 1
w_edit = 4
w_content = 6
w_contrast = 0.2
w_inpaint = 0.8
SDE_strength = 0.4
ip_scale = 0.1
guidance_scale = 1
energy_scale = 0.5
# max_resolution=768
max_resolution=512

def save_images_from_list(image_list, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for i, image_array in enumerate(image_list):
        image = Image.fromarray(image_array)
        image_name = f"{i+0:05d}.png"
        image_path = os.path.join(output_path, image_name)
        image.save(image_path)
        # print(f"Saved image {image_name}")


def find_center_of_mask(mask):
    """
    找到二维 NumPy 数组中所有值为 255 的位置的中心点。
    
    :param mask: 维度为 H*W 的 NumPy 数组
    :return: 中心点的 (y, x) 坐标
    """
    # 获取所有值为 255 的位置
    positions = np.argwhere(mask == 255)
    
    # 如果没有值为 255 的位置，返回 None
    if positions.size == 0:
        return None
    
    # 计算中心点
    center_y = np.mean(positions[:, 0])
    center_x = np.mean(positions[:, 1])
    
    return (int(center_y), int(center_x))

def find_bounding_box(mask):
    """
    找到二维 NumPy 数组中所有值为 255 的位置的 bounding box 的左上点和右下点。
    
    :param mask: 维度为 H*W 的 NumPy 数组
    :return: 左上点 (top_left_y, top_left_x) 和 右下点 (bottom_right_y, bottom_right_x) 的坐标
    """
    # 获取所有值为 255 的位置
    positions = np.argwhere(mask == 255)
    
    # 如果没有值为 255 的位置，返回 None
    if positions.size == 0:
        return None, None
    
    # 分别计算行和列的最小值和最大值
    top_left_y, top_left_x = np.min(positions, axis=0)
    bottom_right_y, bottom_right_x = np.max(positions, axis=0)
    
    return (top_left_y, top_left_x), (bottom_right_y, bottom_right_x)

import json
# translate_dict = json.load(open('./annotations_bbox.json'))
# image_dir = "/home/yinzijin/experiments/gaojiayi/DragonDiffusion_inflation_ldm/data/C19_swan_small/raw_video/"
# image_list = os.listdir(image_dir)
# # image_list = list(translate_dict.keys())
# save_dir = '/home/yinzijin/experiments/gaojiayi/DragonDiffusion_inflation_ldm/data/output_video_swan'+str(resize_scale)+'_replace_inflation_temporalguidance_prev&first_new_originprompt/'
# if not os.path.exists(save_dir):
#     os.mkdir(save_dir)
# mask_dir = "/home/yinzijin/experiments/gaojiayi/DragonDiffusion_inflation_ldm/data/C19_swan_small/raw_object_mask/"
# ref_dir = "/home/yinzijin/experiments/gaojiayi/DragonDiffusion_inflation_ldm/erase_swan_output"
# # if not os.path.exists(save_dir):
# #     os.mkdir(save_dir)
# prompt = "a swan with a red beak swimming in a river near a wall and bushes"

image_dir = "/home/yinzijin/experiments/gaojiayi/DragonDiffusion_inflation_ldm/data/jeep_turn/"
image_list = os.listdir(image_dir)
# image_list = list(translate_dict.keys())
# save_dir = '/home/yinzijin/experiments/gaojiayi/DragonDiffusion_inflation_ldm/data_new/output_video_dog_replace_inflation_wtemporalguidance_12_nofm/'
save_dir = '/home/yinzijin/experiments/gaojiayi/DragonDiffusion_inflation_ldm/result/output_video_jeep2blacktruck_notemp/'

if not os.path.exists(save_dir):
    os.mkdir(save_dir)
mask_dir = "/home/yinzijin/experiments/gaojiayi/DragonDiffusion_inflation_ldm/data/jeep_turn_result_mask/"
ref_dir = "/home/yinzijin/experiments/gaojiayi/DragonDiffusion_inflation_ldm/output_copypaste_jeep2blacktruckorigin/"
maskref_dir = "/home/yinzijin/experiments/gaojiayi/DragonDiffusion_inflation_ldm/output_copypaste_dog_jeep2blacktruckorigin_mask/"
# if not os.path.exists(save_dir):
#     os.mkdir(save_dir)
prompt = " a black truck is moving on the road"


def process_files_in_order(directory):
    # 获取目录中所有的 .png 文件，并按文件名排序
    file_pattern = os.path.join(directory, '*.png')
    files = sorted(glob.glob(file_pattern))

    return files

image_list_sorted = process_files_in_order(image_dir)
img_input_list = []
imgref_input_list = []
mask_input_list = []
mask_ref_list = []

# from IPython import embed;embed()

for image_name in tqdm.tqdm(image_list_sorted[:16]):
    if 'png' not in image_name:
        continue
    image_name = image_name.split("/")[-1]
    
    # from IPython import embed;embed()
    image = Image.open(image_dir +image_name)
    img = np.array(image)
    original_image = None
    mask_name = image_name.replace(".png","_mask.png")
    mask_image = Image.open(mask_dir+mask_name)
    maskref_image = Image.open(maskref_dir+mask_name)
    # from IPython import embed; embed()
    mask_img = np.array(mask_image)
    maskref_image = np.array(maskref_image)
    mask_y,mask_x = find_center_of_mask(mask_img)
    (top_left_y, top_left_x), (bottom_right_y, bottom_right_x) = find_bounding_box(mask_img)
    
    global_points = [[top_left_x,top_left_y],[bottom_right_x,bottom_right_y]]
    selected_points = [[mask_x,mask_y],[mask_x,mask_y]]
    global_point_label = [2]
    img, original_image, selected_points = get_point_move_new(original_image, img, selected_points) 
    img_draw_box, original_image, mask, global_points, global_point_label, img, img_ref = segment_with_points_new(None, original_image[:,:,:3], global_points, global_point_label, img)
    mask_rgb = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
    image_ref = Image.open(os.path.join(ref_dir,image_name))
    target_size =(original_image.shape[1], original_image.shape[0]) 
    image_ref = np.array(image_ref)
    image_ref = cv2.resize( image_ref , target_size)
    img_input_list.append(original_image[:,:,:3])
    imgref_input_list.append(image_ref[:,:,:3])
    mask_input_list.append(mask_rgb)
    mask_ref =np.stack([maskref_image] * 3, axis=-1)
    mask_ref_list.append(mask_ref)
    # from IPython import embed;embed()


# from IPython import embed;embed()
img_input = np.stack(img_input_list,axis=0)
mask_input = np.stack(mask_input_list,axis=0)
imgref_input = np.stack(imgref_input_list,axis=0)
mask_ref_input =  np.stack(mask_ref_list,axis=0)

# output = model.run_move_batch(img_input, mask_input,imgref_input, None, prompt, resize_scale, w_edit, w_content, w_contrast, w_inpaint, seed, selected_points, guidance_scale, energy_scale, max_resolution, SDE_strength, ip_scale)
output = model.run_move_batch_replace(img_input, mask_input,imgref_input, mask_ref_input, prompt, resize_scale, w_edit, w_content, w_contrast, w_inpaint, seed, selected_points, guidance_scale, energy_scale, max_resolution, SDE_strength, ip_scale)

save_images_from_list(output, save_dir)

