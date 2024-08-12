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
resize_scale = 0.7
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
translate_dict = json.load(open('./annotations_bbox.json'))
image_dir = "/home/yinzijin/experiments/gaojiayi/DragonDiffusion_inflation_ldm/data/C19_swan_small/raw_video/"
image_list = os.listdir(image_dir)
# image_list = list(translate_dict.keys())
save_dir = '/home/yinzijin/experiments/gaojiayi/DragonDiffusion_inflation_ldm/data/output_video_swan'+str(resize_scale)+'_replace_inflation_temporalguidance_prev&first_new_originprompt/'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)
mask_dir = "/home/yinzijin/experiments/gaojiayi/DragonDiffusion_inflation_ldm/data/C19_swan_small/raw_object_mask/"
ref_dir = "/home/yinzijin/experiments/gaojiayi/DragonDiffusion_inflation_ldm/erase_swan_output"
# if not os.path.exists(save_dir):
#     os.mkdir(save_dir)
prompt = "a swan with a red beak swimming in a river near a wall and bushes"

def process_files_in_order(directory):
    # 获取目录中所有的 .png 文件，并按文件名排序
    file_pattern = os.path.join(directory, '*.png')
    files = sorted(glob.glob(file_pattern))

    return files

image_list_sorted = process_files_in_order(image_dir)
img_input_list = []
imgref_input_list = []
mask_input_list = []

for image_name in tqdm.tqdm(image_list_sorted):
    if 'png' not in image_name:
        continue
    image_name = image_name.split("/")[-1]
    
    # from IPython import embed;embed()
    image = Image.open(image_dir +image_name)
    img = np.array(image)
    original_image = None
    mask_name = image_name.replace(".png","_mask.png")
    mask_image = Image.open(mask_dir+mask_name)
    # from IPython import embed; embed()
    mask_img = np.array(mask_image)
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

# from IPython import embed;embed()
img_input = np.stack(img_input_list,axis=0)
mask_input = np.stack(mask_input_list,axis=0)
imgref_input = np.stack(imgref_input_list,axis=0)


output = model.run_move_batch(img_input, mask_input,imgref_input, None, prompt, resize_scale, w_edit, w_content, w_contrast, w_inpaint, seed, selected_points, guidance_scale, energy_scale, max_resolution, SDE_strength, ip_scale)

save_images_from_list(output, save_dir)

# print("final")
# from IPython import embed;embed()
# for image_name in tqdm.tqdm(image_list):
#     if 'png' not in image_name:
#         continue
    
#     image = Image.open(image_dir +image_name)
#     img = np.array(image)
#     original_image = None

#     mask_name = image_name.replace(".png","_mask.png")
#     mask_image = Image.open(mask_dir+mask_name)
#     # from IPython import embed; embed()
#     mask_img = np.array(mask_image)
#     mask_y,mask_x = find_center_of_mask(mask_img)
#     (top_left_y, top_left_x), (bottom_right_y, bottom_right_x) = find_bounding_box(mask_img)
    
#     global_points = [[top_left_x,top_left_y],[bottom_right_x,bottom_right_y]]
#     selected_points = [[mask_x,mask_y],[mask_x,mask_y]]

#     # bbox_xywh = translate_dict[image_name]["original"]["boundingbox"]["xywh"]
#     # global_points = [[bbox_xywh[0],bbox_xywh[1]],[bbox_xywh[0]+bbox_xywh[2] ,bbox_xywh[1]+bbox_xywh[3]]]
#     # cx_ori,cy_ori = translate_dict[image_name]["original"]["boundingbox"]["cxcywh"][:2]
#     # cx_trans,cy_trans = translate_dict[image_name]["translate"]["boundingbox"]["cxcywh"][:2]
#     # selected_points = [[cx_ori,cy_ori],[cx_ori,cy_ori]]
#     global_point_label = [2]
    


#     img, original_image, selected_points = get_point_move_new(original_image, img, selected_points) 
#     img_draw_box, original_image, mask, global_points, global_point_label, img, img_ref = segment_with_points_new(None, original_image[:,:,:3], global_points, global_point_label, img)
#     mask_rgb = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
#     image_ref = Image.open(os.path.join(ref_dir,image_name))
#     target_size =(original_image.shape[1], original_image.shape[0]) 
#     image_ref = np.array(image_ref)
#     image_ref = cv2.resize( image_ref , target_size)
    
#     output = model.run_move_(original_image[:,:,:3], mask_rgb,image_ref[:,:,:3], None, prompt, resize_scale, w_edit, w_content, w_contrast, w_inpaint, seed, selected_points, guidance_scale, energy_scale, max_resolution, SDE_strength, ip_scale)
#     output_image = Image.fromarray(np.uint8(output[0]))
#     # image.save('./p_1.png')
#     # output_image.save('./p_1_2.png')
#     # from IPython import embed; embed()
    
    
    
#     output_image.save(save_dir +image_name)
    
# selected_points = gr.State([])
# global_points = [[x1,y1],[x2,y2]]
# global_point_label = []
# # img_draw_box = ""
# # img = ""
# # img_ref = ""

# # mask = ""
# # mask_ref = ""

# seed = 42
# resize_scale = 1
# w_edit = 4
# w_content = 6
# w_contrast = 0.2
# w_inpaint = 0.8
# SDE_strength = 0.4
# ip_scale = 0.1
# guidance_scale = 1
# energy_scale = 0.5
# max_resolution=768
# prompt = "headphones"
# image = Image.open('p_23.png')

# # 转换为numpy数组
# img = np.array(image)
# original_image = None

# img, original_image, selected_points = get_point_move(original_image, img, selected_points) #我不太理解这里为什么要用他的index,但应该可以确保输出的selected_point是两个坐标构成的数组,然后这里的origin image是原图,img是画了两个点的图
# img_draw_box, original_image, mask, global_points, global_point_label, img, img_ref = segment_with_points(img_draw_box, original_image, global_points, global_point_label, img)

# output = model.run_move(original_image, mask, None, prompt, resize_scale, w_edit, w_content, w_contrast, w_inpaint, seed, selected_points, guidance_scale, energy_scale, max_resolution, SDE_strength, ip_scale)

# with gr.Blocks(css='style.css') as demo:
#     gr.Markdown(DESCRIPTION)
#     with gr.Tabs():
#         with gr.TabItem('Appearance Modulation'):
#             create_demo_appearance(model.run_appearance)
#         with gr.TabItem('Object Moving & Resizing'):
#             create_demo_move(model.run_move)
#         with gr.TabItem('Face Modulation'):
#             create_demo_face_drag(model.run_drag_face)
#         with gr.TabItem('Content Dragging'):
#             create_demo_drag(model.run_drag)
#         with gr.TabItem('Object Pasting'):
#             create_demo_paste(model.run_paste)

# demo.queue(concurrency_count=3, max_size=20)
# demo.launch(server_name="0.0.0.0")
