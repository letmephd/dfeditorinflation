from src.models.dragondiff import DragonPipeline
from src.models.dragondiff_3d import DragonPipeline_3d
from src.utils.utils import resize_batch_numpy_images,resize_numpy_image, split_ldm, process_replace_batch,process_move_batch, process_move, process_drag_face, process_drag, process_appearance, process_paste

import torch
import cv2
from pytorch_lightning import seed_everything
from PIL import Image
from torchvision.transforms import PILToTensor
import numpy as np
import torch.nn.functional as F
from basicsr.utils import img2tensor
from src.utils.alignment import align_face, get_landmark
import dlib

NUM_DDIM_STEPS = 50
SIZES = {
    0:4,
    1:2,
    2:1,
    3:1,
}

def tensor_to_mask_image(tensor):
    """
    将形状为1x1x64x64的tensor转换为黑白mask图像。

    参数:
    tensor (torch.Tensor): 由0和1组成的矩阵，大小为(1, 1, 64, 64)。

    返回:
    PIL.Image: 黑白mask图像。
    """
    # 去除多余的维度
    tensor_2d = tensor.squeeze().byte() * 255  # 变成64x64，并将1转为255, 0保持为0
    
    # 转换为numpy数组
    np_image = tensor_2d.numpy()
    
    # 创建PIL图像
    mask_image = Image.fromarray(np_image, mode='L')  # 'L'模式表示灰度图像
    
    return mask_image

class DragonModels_3d():
    def __init__(self, pretrained_model_path):
        self.ip_scale = 0.1
        self.precision = torch.float16
        self.editor = DragonPipeline_3d(sd_id=pretrained_model_path, NUM_DDIM_STEPS=NUM_DDIM_STEPS, precision=self.precision, ip_scale=self.ip_scale)
        self.up_ft_index = [1,2] # fixed in gradio demo
        self.up_scale = 2        # fixed in gradio demo
        self.device = 'cuda'     # fixed in gradio demo
        # face editing
        SHAPE_PREDICTOR_PATH = 'models/shape_predictor_68_face_landmarks.dat'
        self.face_predictor = dlib.shape_predictor(SHAPE_PREDICTOR_PATH)
    
    def run_move_batch_replace(self, original_image, mask, image_replace , mask_ref, prompt, resize_scale, w_edit, w_content, w_contrast, w_inpaint, seed, selected_points, guidance_scale, energy_scale, max_resolution, SDE_strength, ip_scale=None):
        #ref 和 mask ref 应该都是不需要的
        # 只有一张original_image
        # 关键就是搞清mask 和
        seed_everything(seed)
        energy_scale = energy_scale*1e3
        img = original_image # (4640, 6960, 3)
        img_batch, input_scale = resize_batch_numpy_images(img, max_resolution*max_resolution) # 按照最大像素数768*768 对大小进行放缩,后一个值是变放缩的倍数
        # h, w = img_batch.shape[2], img_batch.shape[1] 
        
        n, h, w, c = img_batch.shape
        
        processed_images_prompt = []
        processed_images_tensor = []
        processed_replacements_tensor = []
        
        for i in range(n):
            img = Image.fromarray(img_batch[i])
            img_prompt = img.resize((256, 256))
            img_tensor = (PILToTensor()(img) / 255.0 - 0.5) * 2
            img_tensor = img_tensor.to(self.device, dtype=self.precision).unsqueeze(0)
            # from IPython import embed;embed()
            img_replace = Image.fromarray(image_replace[i])
            img_replace = img_replace.resize((img_tensor.shape[-1], img_tensor.shape[-2]))
            img_replace_tensor = (PILToTensor()(img_replace) / 255.0 - 0.5) * 2
            img_replace_tensor = img_replace_tensor.to(self.device, dtype=self.precision).unsqueeze(0)
            
            processed_images_prompt.append(img_prompt)
            processed_images_tensor.append(img_tensor)
            processed_replacements_tensor.append(img_replace_tensor)

        # from IPython import embed;embed()

        # processed_images_prompt = torch.cat(processed_images_prompt, dim=0)
        processed_images_tensor = torch.cat(processed_images_tensor, dim=0)
        processed_replacements_tensor = torch.cat(processed_replacements_tensor, dim=0)
        
        
        
        # img = Image.fromarray(img)
        # img_prompt = img.resize((256, 256))
        # img_tensor = (PILToTensor()(img) / 255.0 - 0.5) * 2
        # img_tensor = img_tensor.to(self.device, dtype=self.precision).unsqueeze(0)
        
        # img_replace = Image.fromarray(image_replace)
        # img_prompt_replace = img_replace.resize((256, 256))
        # img_replace = img_replace.resize((img_tensor.shape[-1], img_tensor.shape[-2]))
        # img_replace_tensor = (PILToTensor()(img_replace) / 255.0 - 0.5) * 2
        # img_replace_tensor = img_replace_tensor.to(self.device, dtype=self.precision).unsqueeze(0)

        # if mask_ref is not None and np.sum(mask_ref)!=0:
        #     mask_ref = np.repeat(mask_ref[:,:,None], 3, 2)
        # else:
        #     mask_ref = None

        
        emb_im, emb_im_uncond = self.editor.get_image_embeds(processed_images_prompt)
        # emb_im_replace, emb_im_uncond_replace = self.editor.get_image_embeds(img_prompt_replace)
        # emb_im = torch.cat([emb_im, emb_im_replace], dim=1)
        # emb_im_uncond = torch.cat([emb_im_uncond, emb_im_uncond_replace], dim=1)
        if ip_scale is not None and ip_scale != self.ip_scale: # ?
            self.ip_scale = ip_scale
            self.editor.load_adapter(self.editor.ip_id, self.ip_scale)
        # print("test_size")
        # from IPython import embed;embed()
        latent = self.editor.image2latent(processed_images_tensor)
        latent_replace = self.editor.image2latent(processed_replacements_tensor)
        # latent = self.editor.image2latent(img_tensor) ([9, 4, 48, 88])
        # latent_replace = self.editor.image2latent(img_replace_tensor)
        # from IPython import embed;embed()
        prompt1 = "a person and a dog"
        prompt2 = "a person and a white tiger"
        ddim_latents = self.editor.ddim_inv_3d(latent=torch.cat([latent.transpose(0,1).unsqueeze(0),latent_replace.transpose(0,1).unsqueeze(0)]), prompt=[prompt1,prompt2])
        # 51 [2, 4, 9, 48, 88]
        # torch.save(ddim_latents,"ddim_latents.pth")
        
        
        # ddim_latents = self.editor.ddim_inv_3d(latent=torch.cat([latent,latent_replace]), prompt=[prompt,prompt])
            
        # ddim_latents  = torch.load("ddim_latents.pth")
        # latent_in = ddim_latents[-1][:1].squeeze(2)
        # latent_in_replace = ddim_latents[-1][1:].squeeze(2)
        latent_in = ddim_latents[-1][:1]
        latent_in_replace = ddim_latents[-1][1:]
        # torch.save(ddim_latents,"ddim_latents.pth")
        
        scale = 8*SIZES[max(self.up_ft_index)]/self.up_scale #? 这块的话应该只适应于不移动的情况
        x=[]
        y=[]
        x_cur = []
        y_cur = []
        for idx, point in enumerate(selected_points):
            if idx%2 == 0:
                y.append(point[1])
                x.append(point[0])
            else:
                y_cur.append(point[1])
                x_cur.append(point[0])
        dx = x_cur[0]-x[0]
        dy = y_cur[0]-y[0]
        
        # print("test_model1")
        # # # # 将mask保存在本地        
        # from IPython import embed; embed()
        
        edit_kwargs = process_replace_batch( # 这块后面可以再检查下
            path_mask_batch=mask, #就是物体单独的mask
            h=h, #960   
            w=w, #640
            dx=dx, #位置变化情况
            dy=dy, #位置变化情况
            scale=scale, #
            input_scale=input_scale, #图像边的放缩倍数
            resize_scale=resize_scale, 
            up_scale=self.up_scale, 
            up_ft_index=self.up_ft_index, 
            w_edit=w_edit, 
            w_content=w_content, 
            w_contrast=w_contrast, 
            w_inpaint=w_inpaint,  
            precision=self.precision, 
            path_mask_ref_batch=mask_ref
        )
        
        # print("edit test")
        # from IPython import embed;embed()
        
        # resize_scale=1
        #
        device = latent_in.device
        dtype = latent_in.dtype
        latent_tensors = []
        # from IPython import embed;embed()
        batch_size, c, n, h, w = latent_in.shape
        
        resize_scale=1

        for i in range(n):
            mask_tmp = (F.interpolate(img2tensor(mask[i])[0].unsqueeze(0).unsqueeze(0), 
                                    (int(h * resize_scale), int(w * resize_scale))) > 0).float().to(device, dtype=dtype)
            
            mask_ref_tmp = (F.interpolate(img2tensor(mask_ref[i])[0].unsqueeze(0).unsqueeze(0), 
                                    (int(h * resize_scale), int(w * resize_scale))) > 0).float().to(device, dtype=dtype)
            
            # from IPython import embed; embed()
            
            latent_tmp = F.interpolate(latent_in[:, :, i, :, :], 
                                    (int(h * resize_scale), int(w * resize_scale)))
            
            mask_tmp = torch.roll(mask_tmp, (int(dy / (w / h) * resize_scale), 
                                            int(dx / (w / h) * resize_scale)), (-2, -1))
            
            mask_ref_tmp = torch.roll(mask_ref_tmp, (int(dy / (w / h) * resize_scale), 
                                            int(dx / (w / h) * resize_scale)), (-2, -1))
            
            latent_tmp = torch.roll(latent_tmp, (int(dy / (w / h) * resize_scale), 
                                                int(dx / (w / h) * resize_scale)), (-2, -1))
            
            pad_size_x = abs(mask_tmp.shape[-1] - w) // 2
            pad_size_y = abs(mask_tmp.shape[-2] - h) // 2
            # print("in_loop")
            # from IPython import embed;embed()
            # if resize_scale > 1:
            #     sum_before = torch.sum(mask_tmp)
            #     mask_tmp = mask_tmp[:, :, pad_size_y:pad_size_y + h, pad_size_x:pad_size_x + w]
            #     latent_tmp = latent_tmp[:, :, pad_size_y:pad_size_y + h, pad_size_x:pad_size_x + w]
            #     sum_after = torch.sum(mask_tmp)
            #     if sum_after != sum_before:
            #         raise ValueError('Resize out of bounds.')
            
            # elif resize_scale < 1:
            #     temp_mask = torch.zeros(1, 1, h, w).to(device, dtype=dtype)
            #     temp_mask[:, :, pad_size_y:pad_size_y + mask_tmp.shape[-2], pad_size_x:pad_size_x + mask_tmp.shape[-1]] = mask_tmp
            #     mask_tmp = (temp_mask > 0.5).float()

            #     temp_latent = torch.zeros(1, c, h, w).to(device, dtype=dtype)
            #     temp_latent[:, :, pad_size_y:pad_size_y + latent_tmp.shape[-2], pad_size_x:pad_size_x + latent_tmp.shape[-1]] = latent_tmp
            #     latent_tmp = temp_latent
            
            mask_union = torch.logical_or(mask_tmp,mask_ref_tmp).float()
            # from IPython import embed;embed()
            latent_tensors.append((latent_in[:, :, i, :, :] * (1 - mask_union) + latent_in_replace[:, :, i, :, :] * mask_union).to(dtype=dtype))
        # from IPython import embed;embed()
        latent_in = torch.cat(latent_tensors, dim=0).unsqueeze(0).permute(0, 2, 1, 3, 4)
        
        # 从154行到上面的函数可能会导致原图失真
       
        # latent_in = latent[None].transpose(1,2)
        # from IPython import embed;embed()
        # pre-process zT
        # mask_tmp = (F.interpolate(img2tensor(mask)[0].unsqueeze(0).unsqueeze(0), (int(latent_in.shape[-2]*resize_scale), int(latent_in.shape[-1]*resize_scale)))>0).float().to('cuda', dtype=latent_in.dtype)
        # latent_tmp = F.interpolate(latent_in, (int(latent_in.shape[-2]*resize_scale), int(latent_in.shape[-1]*resize_scale)))
        # mask_tmp = torch.roll(mask_tmp, (int(dy/(w/latent_in.shape[-2])*resize_scale), int(dx/(w/latent_in.shape[-2])*resize_scale)), (-2,-1))
        # latent_tmp = torch.roll(latent_tmp, (int(dy/(w/latent_in.shape[-2])*resize_scale), int(dx/(w/latent_in.shape[-2])*resize_scale)), (-2,-1))
        # # 感觉这块和上面有点对不上呢
        # pad_size_x = abs(mask_tmp.shape[-1]-latent_in.shape[-1])//2
        # pad_size_y = abs(mask_tmp.shape[-2]-latent_in.shape[-2])//2
        # if resize_scale>1:
        #     sum_before = torch.sum(mask_tmp)
        #     mask_tmp = mask_tmp[:,:,pad_size_y:pad_size_y+latent_in.shape[-2],pad_size_x:pad_size_x+latent_in.shape[-1]]
        #     latent_tmp = latent_tmp[:,:,pad_size_y:pad_size_y+latent_in.shape[-2],pad_size_x:pad_size_x+latent_in.shape[-1]]
        #     sum_after = torch.sum(mask_tmp)
        #     if sum_after != sum_before:
        #         raise ValueError('Resize out of bounds.')
        #         exit(0)
        # elif resize_scale<1:
        #     temp = torch.zeros(1,1,latent_in.shape[-2], latent_in.shape[-1]).to(latent_in.device, dtype=latent_in.dtype)
        #     temp[:,:,pad_size_y:pad_size_y+mask_tmp.shape[-2],pad_size_x:pad_size_x+mask_tmp.shape[-1]]=mask_tmp
        #     mask_tmp =(temp>0.5).float()
        #     temp = torch.zeros_like(latent_in)
        #     temp[:,:,pad_size_y:pad_size_y+latent_tmp.shape[-2],pad_size_x:pad_size_x+latent_tmp.shape[-1]]=latent_tmp
        #     latent_tmp = temp
        # latent_in = (latent_in*(1-mask_tmp)+latent_tmp*mask_tmp).to(dtype=latent_in.dtype) # torch.Size([1, 4, 80, 120])
        
        #self.pipe src.models.dragondiff.DragonPipeline
        #
        
        latent_rec = self.editor.pipe.edit(
            mode = 'move_batch_replace',
            emb_im=emb_im,
            emb_im_uncond=emb_im_uncond, # [1, 64, 768]
            latent=latent_in, # torch.Size([1, 4, 80, 120])
            prompt=prompt, 
            guidance_scale=guidance_scale, 
            energy_scale=energy_scale,  
            latent_noise_ref = ddim_latents, #51*[1, 4, 1, 80, 120]
            SDE_strength=SDE_strength,
            edit_kwargs=edit_kwargs,
        )
        # img_rec = self.editor.decode_latents(latent_rec)[:,:,::-1]
        # print("decode")
        # from IPython import embed;embed()
        img_rec = self.editor.decode_latents_video(latent_rec)
        # print("decode_finish")
        # from IPython import embed;embed()
        torch.cuda.empty_cache()

        return img_rec
    
    def run_move_(self, original_image, mask, image_replace , mask_ref, prompt, resize_scale, w_edit, w_content, w_contrast, w_inpaint, seed, selected_points, guidance_scale, energy_scale, max_resolution, SDE_strength, ip_scale=None):
        #ref 和 mask ref 应该都是不需要的
        # 只有一张original_image
        # 关键就是搞清mask 和
        seed_everything(seed)
        energy_scale = energy_scale*1e3
        img = original_image # (4640, 6960, 3)
        img, input_scale = resize_numpy_image(img, max_resolution*max_resolution) # 按照最大像素数768*768 对大小进行放缩,后一个值是变放缩的倍数
        h, w = img.shape[1], img.shape[0] 
        img = Image.fromarray(img)
        img_prompt = img.resize((256, 256))
        img_tensor = (PILToTensor()(img) / 255.0 - 0.5) * 2
        img_tensor = img_tensor.to(self.device, dtype=self.precision).unsqueeze(0)
        
        img_replace = Image.fromarray(image_replace)
        img_prompt_replace = img_replace.resize((256, 256))
        img_replace = img_replace.resize((img_tensor.shape[-1], img_tensor.shape[-2]))
        img_replace_tensor = (PILToTensor()(img_replace) / 255.0 - 0.5) * 2
        img_replace_tensor = img_replace_tensor.to(self.device, dtype=self.precision).unsqueeze(0)

        if mask_ref is not None and np.sum(mask_ref)!=0:
            mask_ref = np.repeat(mask_ref[:,:,None], 3, 2)
        else:
            mask_ref = None

        
        emb_im, emb_im_uncond = self.editor.get_image_embeds(img_prompt)
        # emb_im_replace, emb_im_uncond_replace = self.editor.get_image_embeds(img_prompt_replace)
        # emb_im = torch.cat([emb_im, emb_im_replace], dim=1)
        # emb_im_uncond = torch.cat([emb_im_uncond, emb_im_uncond_replace], dim=1)
        if ip_scale is not None and ip_scale != self.ip_scale:
            self.ip_scale = ip_scale
            self.editor.load_adapter(self.editor.ip_id, self.ip_scale)
        # from IPython import embed;embed()
        latent = self.editor.image2latent(img_tensor)
        latent_replace = self.editor.image2latent(img_replace_tensor)
        
        ddim_latents = self.editor.ddim_inv(latent=torch.cat([latent,latent_replace]), prompt=[prompt,prompt])
            
        # ddim_latents  = torch.load("ddim_latents.pth")
        latent_in = ddim_latents[-1][:1].squeeze(2)
        # torch.save(ddim_latents,"ddim_latents.pth")
        
        scale = 8*SIZES[max(self.up_ft_index)]/self.up_scale #?
        x=[]
        y=[]
        x_cur = []
        y_cur = []
        for idx, point in enumerate(selected_points):
            if idx%2 == 0:
                y.append(point[1])
                x.append(point[0])
            else:
                y_cur.append(point[1])
                x_cur.append(point[0])
        dx = x_cur[0]-x[0]
        dy = y_cur[0]-y[0]
        
        
        edit_kwargs = process_move(
            path_mask=mask, #就是物体单独的mask
            h=h, #960   
            w=w, #640
            dx=dx, #位置变化情况
            dy=dy, #位置变化情况
            scale=scale, #
            input_scale=input_scale, #图像边的放缩倍数
            resize_scale=resize_scale, 
            up_scale=self.up_scale, 
            up_ft_index=self.up_ft_index, 
            w_edit=w_edit, 
            w_content=w_content, 
            w_contrast=w_contrast, 
            w_inpaint=w_inpaint,  
            precision=self.precision, 
            path_mask_ref=mask_ref
        )
        # pre-process zT
       
        mask_tmp = (F.interpolate(img2tensor(mask)[0].unsqueeze(0).unsqueeze(0), (int(latent_in.shape[-2]*resize_scale), int(latent_in.shape[-1]*resize_scale)))>0).float().to('cuda', dtype=latent_in.dtype)
        latent_tmp = F.interpolate(latent_in, (int(latent_in.shape[-2]*resize_scale), int(latent_in.shape[-1]*resize_scale)))
        mask_tmp = torch.roll(mask_tmp, (int(dy/(w/latent_in.shape[-2])*resize_scale), int(dx/(w/latent_in.shape[-2])*resize_scale)), (-2,-1))
        latent_tmp = torch.roll(latent_tmp, (int(dy/(w/latent_in.shape[-2])*resize_scale), int(dx/(w/latent_in.shape[-2])*resize_scale)), (-2,-1))
        pad_size_x = abs(mask_tmp.shape[-1]-latent_in.shape[-1])//2
        pad_size_y = abs(mask_tmp.shape[-2]-latent_in.shape[-2])//2
        if resize_scale>1:
            sum_before = torch.sum(mask_tmp)
            mask_tmp = mask_tmp[:,:,pad_size_y:pad_size_y+latent_in.shape[-2],pad_size_x:pad_size_x+latent_in.shape[-1]]
            latent_tmp = latent_tmp[:,:,pad_size_y:pad_size_y+latent_in.shape[-2],pad_size_x:pad_size_x+latent_in.shape[-1]]
            sum_after = torch.sum(mask_tmp)
            if sum_after != sum_before:
                raise ValueError('Resize out of bounds.')
                exit(0)
        elif resize_scale<1:
            
            temp = torch.zeros(1,1,latent_in.shape[-2], latent_in.shape[-1]).to(latent_in.device, dtype=latent_in.dtype)
            temp[:,:,pad_size_y:pad_size_y+mask_tmp.shape[-2],pad_size_x:pad_size_x+mask_tmp.shape[-1]]=mask_tmp
            mask_tmp =(temp>0.5).float()
            temp = torch.zeros_like(latent_in)
            temp[:,:,pad_size_y:pad_size_y+latent_tmp.shape[-2],pad_size_x:pad_size_x+latent_tmp.shape[-1]]=latent_tmp
            latent_tmp = temp
        latent_in = (latent_in*(1-mask_tmp)+latent_tmp*mask_tmp).to(dtype=latent_in.dtype) # torch.Size([1, 4, 80, 120])
        # from IPython import embed; embed()
        #self.pipe src.models.dragondiff.DragonPipeline
        #
        # print("test_model1")
        # # 将mask保存在本地        
        # from IPython import embed; embed()
        latent_rec = self.editor.pipe.edit(
            mode = 'move_',
            emb_im=emb_im,
            emb_im_uncond=emb_im_uncond, # [1, 64, 768]
            latent=latent_in, # torch.Size([1, 4, 80, 120])
            prompt=prompt, 
            guidance_scale=guidance_scale, 
            energy_scale=energy_scale,  
            latent_noise_ref = ddim_latents, #51*[1, 4, 1, 80, 120]
            SDE_strength=SDE_strength,
            edit_kwargs=edit_kwargs,
        )
        img_rec = self.editor.decode_latents(latent_rec)[:,:,::-1]
        torch.cuda.empty_cache()

        return [img_rec]

    def run_move(self, original_image, mask, mask_ref, prompt, resize_scale, w_edit, w_content, w_contrast, w_inpaint, seed, selected_points, guidance_scale, energy_scale, max_resolution, SDE_strength, ip_scale=None):
        #ref 和 mask ref 应该都是不需要的
        # 只有一张original_image
        # 关键就是搞清mask 和
        seed_everything(seed)
        energy_scale = energy_scale*1e3
        img = original_image # (4640, 6960, 3)
        img, input_scale = resize_numpy_image(img, max_resolution*max_resolution) # 按照最大像素数768*768 对大小进行放缩,后一个值是变放缩的倍数
        h, w = img.shape[1], img.shape[0] 
        img = Image.fromarray(img)
        img_prompt = img.resize((256, 256))
        img_tensor = (PILToTensor()(img) / 255.0 - 0.5) * 2
        img_tensor = img_tensor.to(self.device, dtype=self.precision).unsqueeze(0)

        if mask_ref is not None and np.sum(mask_ref)!=0:
            mask_ref = np.repeat(mask_ref[:,:,None], 3, 2)
        else:
            mask_ref = None

        emb_im, emb_im_uncond = self.editor.get_image_embeds(img_prompt)
        if ip_scale is not None and ip_scale != self.ip_scale:
            self.ip_scale = ip_scale
            self.editor.load_adapter(self.editor.ip_id, self.ip_scale)
        latent = self.editor.image2latent(img_tensor)
        ddim_latents = self.editor.ddim_inv(latent=latent, prompt=prompt)
        latent_in = ddim_latents[-1].squeeze(2)
        
        scale = 8*SIZES[max(self.up_ft_index)]/self.up_scale #?
        x=[]
        y=[]
        x_cur = []
        y_cur = []
        for idx, point in enumerate(selected_points):
            if idx%2 == 0:
                y.append(point[1])
                x.append(point[0])
            else:
                y_cur.append(point[1])
                x_cur.append(point[0])
        dx = x_cur[0]-x[0]
        dy = y_cur[0]-y[0]
        
        edit_kwargs = process_move(
            path_mask=mask, #就是物体单独的mask
            h=h, #960   
            w=w, #640
            dx=dx, #位置变化情况
            dy=dy, #位置变化情况
            scale=scale, #
            input_scale=input_scale, #图像边的放缩倍数
            resize_scale=resize_scale, 
            up_scale=self.up_scale, 
            up_ft_index=self.up_ft_index, 
            w_edit=w_edit, 
            w_content=w_content, 
            w_contrast=w_contrast, 
            w_inpaint=w_inpaint,  
            precision=self.precision, 
            path_mask_ref=mask_ref
        )
        # pre-process zT
        mask_tmp = (F.interpolate(img2tensor(mask)[0].unsqueeze(0).unsqueeze(0), (int(latent_in.shape[-2]*resize_scale), int(latent_in.shape[-1]*resize_scale)))>0).float().to('cuda', dtype=latent_in.dtype)
        latent_tmp = F.interpolate(latent_in, (int(latent_in.shape[-2]*resize_scale), int(latent_in.shape[-1]*resize_scale)))
        mask_tmp = torch.roll(mask_tmp, (int(dy/(w/latent_in.shape[-2])*resize_scale), int(dx/(w/latent_in.shape[-2])*resize_scale)), (-2,-1))
        latent_tmp = torch.roll(latent_tmp, (int(dy/(w/latent_in.shape[-2])*resize_scale), int(dx/(w/latent_in.shape[-2])*resize_scale)), (-2,-1))
        pad_size_x = abs(mask_tmp.shape[-1]-latent_in.shape[-1])//2
        pad_size_y = abs(mask_tmp.shape[-2]-latent_in.shape[-2])//2
        if resize_scale>1:
            sum_before = torch.sum(mask_tmp)
            mask_tmp = mask_tmp[:,:,pad_size_y:pad_size_y+latent_in.shape[-2],pad_size_x:pad_size_x+latent_in.shape[-1]]
            latent_tmp = latent_tmp[:,:,pad_size_y:pad_size_y+latent_in.shape[-2],pad_size_x:pad_size_x+latent_in.shape[-1]]
            sum_after = torch.sum(mask_tmp)
            if sum_after != sum_before:
                raise ValueError('Resize out of bounds.')
                exit(0)
        elif resize_scale<1:
            temp = torch.zeros(1,1,latent_in.shape[-2], latent_in.shape[-1]).to(latent_in.device, dtype=latent_in.dtype)
            temp[:,:,pad_size_y:pad_size_y+mask_tmp.shape[-2],pad_size_x:pad_size_x+mask_tmp.shape[-1]]=mask_tmp
            mask_tmp =(temp>0.5).float()
            temp = torch.zeros_like(latent_in)
            temp[:,:,pad_size_y:pad_size_y+latent_tmp.shape[-2],pad_size_x:pad_size_x+latent_tmp.shape[-1]]=latent_tmp
            latent_tmp = temp
        latent_in = (latent_in*(1-mask_tmp)+latent_tmp*mask_tmp).to(dtype=latent_in.dtype) # torch.Size([1, 4, 80, 120])
        # from IPython import embed; embed()
        #self.pipe src.models.dragondiff.DragonPipeline
        #
        # print("test_model1")
        # # 将mask保存在本地        
        # from IPython import embed; embed()
        latent_rec = self.editor.pipe.edit(
            mode = 'move',
            emb_im=emb_im,
            emb_im_uncond=emb_im_uncond, # [1, 64, 768]
            latent=latent_in, # torch.Size([1, 4, 80, 120])
            prompt=prompt, 
            guidance_scale=guidance_scale, 
            energy_scale=energy_scale,  
            latent_noise_ref = ddim_latents, #51*[1, 4, 1, 80, 120]
            SDE_strength=SDE_strength,
            edit_kwargs=edit_kwargs,
        )
        img_rec = self.editor.decode_latents(latent_rec)[:,:,::-1]
        torch.cuda.empty_cache()

        return [img_rec]

    def run_appearance(self, img_base, mask_base, img_replace, mask_replace, prompt, prompt_replace, w_edit, w_content, seed, guidance_scale, energy_scale, max_resolution, SDE_strength, ip_scale=None):
        seed_everything(seed)
        energy_scale = energy_scale*1e3
        img_base, input_scale = resize_numpy_image(img_base, max_resolution*max_resolution)
        h, w = img_base.shape[1], img_base.shape[0] 
        img_base = Image.fromarray(img_base)
        img_prompt_base = img_base.resize((256, 256))
        img_base_tensor = (PILToTensor()(img_base) / 255.0 - 0.5) * 2
        img_base_tensor = img_base_tensor.to(self.device, dtype=self.precision).unsqueeze(0)

        img_replace = Image.fromarray(img_replace)
        img_prompt_replace = img_replace.resize((256, 256))
        img_replace = img_replace.resize((img_base_tensor.shape[-1], img_base_tensor.shape[-2]))
        img_replace_tensor = (PILToTensor()(img_replace) / 255.0 - 0.5) * 2
        img_replace_tensor = img_replace_tensor.to(self.device, dtype=self.precision).unsqueeze(0)

        mask_replace = np.repeat(mask_replace[:,:,None], 3, 2) if len(mask_replace.shape)==2 else mask_replace
        mask_base = np.repeat(mask_base[:,:,None], 3, 2) if len(mask_base.shape)==2 else mask_base

        emb_im_base, emb_im_uncond_base = self.editor.get_image_embeds(img_prompt_base)
        emb_im_replace, emb_im_uncond_replace = self.editor.get_image_embeds(img_prompt_replace)
        emb_im = torch.cat([emb_im_base, emb_im_replace], dim=1)
        emb_im_uncond = torch.cat([emb_im_uncond_base, emb_im_uncond_replace], dim=1)

        if ip_scale is not None and ip_scale != self.ip_scale:
            self.ip_scale = ip_scale
            self.editor.load_adapter(self.editor.ip_id, self.ip_scale)
        latent_base = self.editor.image2latent(img_base_tensor)
        latent_replace = self.editor.image2latent(img_replace_tensor)
        ddim_latents = self.editor.ddim_inv(latent=torch.cat([latent_base, latent_replace]), prompt=[prompt, prompt_replace])
        latent_in = ddim_latents[-1][:1].squeeze(2)
        
        scale = 8*SIZES[max(self.up_ft_index)]/self.up_scale
        edit_kwargs = process_appearance(
            path_mask = mask_base, 
            path_mask_replace = mask_replace, 
            h = h, 
            w = w, 
            scale = scale, 
            input_scale = input_scale, 
            up_scale = self.up_scale, 
            up_ft_index = self.up_ft_index, 
            w_edit = w_edit, 
            w_content = w_content, 
            precision = self.precision
        )
        latent_rec = self.editor.pipe.edit(
            mode = 'appearance',
            emb_im=emb_im,
            emb_im_uncond=emb_im_uncond,
            latent=latent_in, 
            prompt=prompt, 
            guidance_scale=guidance_scale, 
            energy_scale=energy_scale,  
            latent_noise_ref = ddim_latents,
            SDE_strength=SDE_strength, 
            edit_kwargs=edit_kwargs,
        )
        img_rec = self.editor.decode_latents(latent_rec)[:,:,::-1]
        torch.cuda.empty_cache()

        return [img_rec]

    def run_drag_face(self, original_image, reference_image, w_edit, w_inpaint, seed, guidance_scale, energy_scale, max_resolution, SDE_strength, ip_scale=0.05):
        seed_everything(seed)
        prompt = 'a photo of a human face'
        energy_scale = energy_scale*1e3
        original_image = np.array(align_face(original_image, self.face_predictor, 1024))
        reference_image = np.array(align_face(reference_image, self.face_predictor, 1024))
        ldm = get_landmark(original_image, self.face_predictor)
        ldm_ref = get_landmark(reference_image, self.face_predictor)
        x_cur, y_cur = split_ldm(ldm_ref)
        x, y = split_ldm(ldm)
        original_image, input_scale = resize_numpy_image(original_image, max_resolution*max_resolution)
        reference_image, _ = resize_numpy_image(reference_image, max_resolution*max_resolution)
        img = original_image
        h, w = img.shape[1], img.shape[0] 
        img = Image.fromarray(img)
        img_prompt = img.resize((256, 256))
        img_tensor = (PILToTensor()(img) / 255.0 - 0.5) * 2
        img_tensor = img_tensor.to(self.device, dtype=self.precision).unsqueeze(0)

        emb_im, emb_im_uncond = self.editor.get_image_embeds(img_prompt)
        if ip_scale is not None and ip_scale != self.ip_scale:
            self.ip_scale = ip_scale
            self.editor.load_adapter(self.editor.ip_id, self.ip_scale)

        latent = self.editor.image2latent(img_tensor)
        ddim_latents = self.editor.ddim_inv(latent=latent, prompt=prompt)
        latent_in = ddim_latents[-1].squeeze(2)
        
        scale = 8*SIZES[max(self.up_ft_index)]/self.up_scale
        edit_kwargs = process_drag_face(
            h=h, 
            w=w, 
            x=x,
            y=y,
            x_cur=x_cur,
            y_cur=y_cur,
            scale=scale, 
            input_scale=input_scale, 
            up_scale=self.up_scale, 
            up_ft_index=self.up_ft_index, 
            w_edit=w_edit, 
            w_inpaint=w_inpaint,  
            precision=self.precision, 
        )
        latent_rec = self.editor.pipe.edit(
            mode = 'landmark',
            emb_im=emb_im,
            emb_im_uncond=emb_im_uncond,
            latent=latent_in, 
            prompt=prompt, 
            guidance_scale=guidance_scale, 
            energy_scale=energy_scale, 
            latent_noise_ref = ddim_latents, 
            edit_kwargs=edit_kwargs,
            SDE_strength_un=SDE_strength,
            SDE_strength = SDE_strength,
        )
        img_rec = self.editor.decode_latents(latent_rec)[:,:,::-1]
        torch.cuda.empty_cache()
        # draw editing direction
        for x_cur_i, y_cur_i in zip(x_cur, y_cur):
            reference_image = cv2.circle(reference_image, (x_cur_i, y_cur_i), 8,(255,0,0),-1)
        for x_i, y_i, x_cur_i, y_cur_i in zip(x, y, x_cur, y_cur):
            cv2.arrowedLine(original_image, (x_i, y_i), (x_cur_i, y_cur_i), (255, 255, 255), 4, tipLength=0.2)
            original_image = cv2.circle(original_image, (x_i, y_i), 6,(0,0,255),-1)
            original_image = cv2.circle(original_image, (x_cur_i, y_cur_i), 6,(255,0,0),-1)

        return [img_rec, reference_image, original_image]

    def run_drag(self, original_image, mask, prompt, w_edit, w_content, w_inpaint, seed, selected_points, guidance_scale, energy_scale, max_resolution, SDE_strength, ip_scale=None):
        seed_everything(seed)
        energy_scale = energy_scale*1e3
        img = original_image
        img, input_scale = resize_numpy_image(img, max_resolution*max_resolution)
        h, w = img.shape[1], img.shape[0] 
        img = Image.fromarray(img)
        img_prompt = img.resize((256, 256))
        img_tensor = (PILToTensor()(img) / 255.0 - 0.5) * 2
        img_tensor = img_tensor.to(self.device, dtype=self.precision).unsqueeze(0)
        mask = np.repeat(mask[:,:,None], 3, 2) if len(mask.shape)==2 else mask

        emb_im, emb_im_uncond = self.editor.get_image_embeds(img_prompt)
        if ip_scale is not None and ip_scale != self.ip_scale:
            self.ip_scale = ip_scale
            self.editor.load_adapter(self.editor.ip_id, self.ip_scale)

        latent = self.editor.image2latent(img_tensor)
        ddim_latents = self.editor.ddim_inv(latent=latent, prompt=prompt)
        latent_in = ddim_latents[-1].squeeze(2)

        x=[]
        y=[]
        x_cur = []
        y_cur = []
        for idx, point in enumerate(selected_points):
            if idx%2 == 0:
                y.append(point[1]*input_scale)
                x.append(point[0]*input_scale)
            else:
                y_cur.append(point[1]*input_scale)
                x_cur.append(point[0]*input_scale)
        
        scale = 8*SIZES[max(self.up_ft_index)]/self.up_scale
        edit_kwargs = process_drag(
            latent_in = latent_in,
            path_mask=mask, 
            h=h, 
            w=w, 
            x=x,
            y=y,
            x_cur=x_cur,
            y_cur=y_cur,
            scale=scale, 
            input_scale=input_scale, 
            up_scale=self.up_scale, 
            up_ft_index=self.up_ft_index, 
            w_edit=w_edit, 
            w_content=w_content, 
            w_inpaint=w_inpaint,  
            precision=self.precision, 
        )
        latent_in = edit_kwargs.pop('latent_in')
        latent_rec = self.editor.pipe.edit(
            mode = 'drag',
            emb_im=emb_im,
            emb_im_uncond=emb_im_uncond,
            latent=latent_in, 
            prompt=prompt, 
            guidance_scale=guidance_scale, 
            energy_scale=energy_scale, 
            latent_noise_ref = ddim_latents,
            SDE_strength=SDE_strength, 
            edit_kwargs=edit_kwargs,
        )
        img_rec = self.editor.decode_latents(latent_rec)[:,:,::-1]
        torch.cuda.empty_cache()

        return [img_rec]

    def run_paste(self, img_base, mask_base, img_replace, prompt, prompt_replace, w_edit, w_content, seed, guidance_scale, energy_scale, dx, dy, resize_scale, max_resolution, SDE_strength, ip_scale=None):
        seed_everything(seed)
        energy_scale = energy_scale*1e3
        img_base, input_scale = resize_numpy_image(img_base, max_resolution*max_resolution)
        h, w = img_base.shape[1], img_base.shape[0] 
        img_base = Image.fromarray(img_base)
        img_prompt_base = img_base.resize((256, 256))
        img_base_tensor = (PILToTensor()(img_base) / 255.0 - 0.5) * 2
        img_base_tensor = img_base_tensor.to(self.device, dtype=self.precision).unsqueeze(0)

        img_replace = Image.fromarray(img_replace)
        img_prompt_replace = img_replace.resize((256, 256))
        img_replace = img_replace.resize((img_base_tensor.shape[-1], img_base_tensor.shape[-2]))
        img_replace_tensor = (PILToTensor()(img_replace) / 255.0 - 0.5) * 2
        img_replace_tensor = img_replace_tensor.to(self.device, dtype=self.precision).unsqueeze(0)

        mask_base = np.repeat(mask_base[:,:,None], 3, 2) if len(mask_base.shape)==2 else mask_base

        emb_im_base, emb_im_uncond_base = self.editor.get_image_embeds(img_prompt_base)
        emb_im_replace, emb_im_uncond_replace = self.editor.get_image_embeds(img_prompt_replace)
        emb_im = torch.cat([emb_im_base, emb_im_replace], dim=1)
        emb_im_uncond = torch.cat([emb_im_uncond_base, emb_im_uncond_replace], dim=1)

        if ip_scale is not None and ip_scale != self.ip_scale:
            self.ip_scale = ip_scale
            self.editor.load_adapter(self.editor.ip_id, self.ip_scale)
        latent_base = self.editor.image2latent(img_base_tensor)
        if resize_scale != 1:
            hr, wr = img_replace_tensor.shape[-2], img_replace_tensor.shape[-1]
            img_replace_tensor = F.interpolate(img_replace_tensor, (int(hr*resize_scale), int(wr*resize_scale)))
            pad_size_x = abs(img_replace_tensor.shape[-1]-wr)//2
            pad_size_y = abs(img_replace_tensor.shape[-2]-hr)//2
            if resize_scale>1:
                img_replace_tensor = img_replace_tensor[:,:,pad_size_y:pad_size_y+hr,pad_size_x:pad_size_x+wr]
            else:
                temp = torch.zeros(1,3,hr, wr).to(self.device, dtype=self.precision)
                temp[:,:,pad_size_y:pad_size_y+img_replace_tensor.shape[-2],pad_size_x:pad_size_x+img_replace_tensor.shape[-1]]=img_replace_tensor
                img_replace_tensor = temp

        latent_replace = self.editor.image2latent(img_replace_tensor)
        ddim_latents = self.editor.ddim_inv(latent=torch.cat([latent_base, latent_replace]), prompt=[prompt, prompt_replace])
        latent_in = ddim_latents[-1][:1].squeeze(2)
        
        scale = 8*SIZES[max(self.up_ft_index)]/self.up_scale
        edit_kwargs = process_paste(
            path_mask=mask_base, 
            h=h, 
            w=w, 
            dx=dx, 
            dy=dy, 
            scale=scale, 
            input_scale=input_scale, 
            up_scale=self.up_scale, 
            up_ft_index=self.up_ft_index, 
            w_edit = w_edit, 
            w_content = w_content, 
            precision = self.precision,
            resize_scale=resize_scale
        )
        mask_tmp = (F.interpolate(edit_kwargs['mask_base_cur'].float(), (latent_in.shape[-2], latent_in.shape[-1]))>0).float()
        latent_tmp = torch.roll(ddim_latents[-1][1:].squeeze(2), (int(dy/(w/latent_in.shape[-2])), int(dx/(w/latent_in.shape[-2]))), (-2,-1))
        latent_in = (latent_in*(1-mask_tmp)+latent_tmp*mask_tmp).to(dtype=latent_in.dtype)

        latent_rec = self.editor.pipe.edit(
            mode = 'paste',
            emb_im=emb_im,
            emb_im_uncond=emb_im_uncond,
            latent=latent_in, 
            prompt=prompt, 
            guidance_scale=guidance_scale, 
            energy_scale=energy_scale,  
            latent_noise_ref = ddim_latents, 
            SDE_strength=SDE_strength,
            edit_kwargs=edit_kwargs,
        )
        img_rec = self.editor.decode_latents(latent_rec)[:,:,::-1]
        torch.cuda.empty_cache()

        return [img_rec]
