
from diffusers import StableDiffusionPipeline
from typing import Any, Callable, Dict, List, Optional, Union
import torch.nn.functional as F
import torch
from basicsr.utils import img2tensor
from tqdm import tqdm
import torch.nn as nn
import copy
import numpy as np
from PIL import Image
from einops import rearrange, repeat
from src.utils.temp_func import *
# from src.demo.model_3d import tensor_to_mask_image
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

class Sampler_3d(StableDiffusionPipeline):

    def edit(
        self,
        prompt:  List[str],
        mode,
        emb_im,
        emb_im_uncond,
        edit_kwargs,
        num_inference_steps: int = 50,
        guidance_scale: Optional[float] = 7.5,
        latent: Optional[torch.FloatTensor] = None,
        start_time=50,
        energy_scale = 0,
        SDE_strength = 0.4,
        SDE_strength_un = 0,
        latent_noise_ref = None,
        alg='D+'
    ):
        print('Start Editing:')
        
        self.alg=alg
        # generate source text embedding
        text_input = self.tokenizer(
            [prompt],
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]
        max_length = text_input.input_ids.shape[-1]
        uncond_input = self.tokenizer(
                [""], padding="max_length", max_length=max_length, return_tensors="pt"
            )
        uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]
        # image prompt
        
        # choice one 变成bs维
        if emb_im is not None and emb_im_uncond is not None:
            # choice 1 变成bs维
            # bs = emb_im.shape[0]
            # uncond_embeddings = torch.cat([uncond_embeddings.repeat(bs,1,1), emb_im_uncond],dim=1)
            text_embeddings_org = text_embeddings
            # text_embeddings = torch.cat([text_embeddings.repeat(bs,1,1), emb_im],dim=1)
            # context = torch.cat([uncond_embeddings.expand(*text_embeddings.shape), text_embeddings])
            # context_ = torch.cat([uncond_embeddings.expand(*text_embeddings.shape)[None], text_embeddings[None]])
            # choice 2 放弃image prompt
            context = torch.cat([uncond_embeddings.expand(*text_embeddings.shape), text_embeddings])
        
        print("test_energy_scale")
        print(energy_scale )
        print("test pipeline")
        # from IPython import embed; embed()

        self.scheduler.set_timesteps(num_inference_steps) 
        dict_mask = edit_kwargs['dict_mask'] if 'dict_mask' in edit_kwargs else None
        
        for i, t in enumerate(tqdm(self.scheduler.timesteps[-start_time:])):
            next_timestep = min(t - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999)
            next_timestep = max(next_timestep, 0)
            if energy_scale==0 or alg=='D':
                repeat=1
            elif 20<i<30 and i%2==0 : 
                repeat = 20
            else:
                repeat = 1
            stack = []
            for ri in range(repeat):
                latent_in = torch.cat([latent] * 2) # torch.Size([2, 4, 1, 96, 96])
                with torch.no_grad():
                    # noise_pred = self.unet(latent_in, t, encoder_hidden_states=context, mask=dict_mask, save_kv=False, mode=mode, iter_cur=i)["sample"].squeeze(2)
                    # mode,'move_'; iter_cur=i
                    # print("try sampling")
                    # from IPython import embed; embed()
                    noise_pred = self.unet(latent_in, t, encoder_hidden_states=context,inversion = False, samplestep=i,prompt = [prompt])["sample"]
                    # noise_pred = self.unet(latent_in, t, encoder_hidden_states=context, save_kv=False, mode=mode, iter_cur=i)["sample"].squeeze(2)
                    # save_kv啥的看下dragondiffusion的attention2d
                noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
                # torch.Size([1, 4, 8, 64, 64])
                noise_pred_org=None
                # from IPython import embed; embed()
                if energy_scale!=0 and i<30 and (alg=='D' or i%2==0 or i<10):
                    # editing guidance
                    # print(i)
                    noise_pred_org = noise_pred
                    if mode == 'move_batch_replace':
                        # print("test_move_batch_replace")
                        # from IPython import embed; embed()
                        guidance = 0.15*self.guidance_move_perframe_replace_changed0827(latent=latent, latent_noise_ref=latent_noise_ref[-(i+1)], t=t, text_embeddings=text_embeddings_org, energy_scale=energy_scale, **edit_kwargs)
                        guidance2 = 1.5*self.guidance_move_perframe_residual2(latent=latent, latent_noise_ref=latent_noise_ref[-(i+1)], t=t, text_embeddings=text_embeddings_org, energy_scale=energy_scale, **edit_kwargs)
                        guidance = guidance+guidance2
                        # print("test_move_batch_replace")
                        # from IPython import embed; embed()
                        # guidance = guidance+ 0.3*self.guidance_move_perframe_residual(latent=latent, latent_noise_ref=latent_noise_ref[-(i+1)], t=t, text_embeddings=text_embeddings_org, energy_scale=energy_scale, **edit_kwargs)
                        # guidance = guidance
                    elif mode == 'move_batch':
                        guidance = self.guidance_move_perframe(latent=latent, latent_noise_ref=latent_noise_ref[-(i+1)], t=t, text_embeddings=text_embeddings_org, energy_scale=energy_scale, **edit_kwargs)
                        guidance = guidance+ 10*self.guidance_temporal(latent=latent, latent_noise_ref=latent_noise_ref[-(i+1)], t=t, text_embeddings=text_embeddings_org, energy_scale=energy_scale, **edit_kwargs)
                    
                    noise_pred = noise_pred + guidance
                    # noise_pred = noise_pred 
                else:
                    noise_pred_org=None
                
                # zt->zt-1
                prev_timestep = t - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
                alpha_prod_t = self.scheduler.alphas_cumprod[t]
                alpha_prod_t_prev = self.scheduler.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.scheduler.final_alpha_cumprod
                beta_prod_t = 1 - alpha_prod_t
                pred_original_sample = (latent - beta_prod_t ** (0.5) * noise_pred) / alpha_prod_t ** (0.5)

                if 10<i<20:
                    eta, eta_rd = 0,0
                else:
                    eta, eta_rd = 0., 0.
                
                variance = self.scheduler._get_variance(t, prev_timestep)
                std_dev_t = eta * variance ** (0.5)
                std_dev_t_rd = eta_rd * variance ** (0.5)
                # if energy_scale!=0 and i<30 and (alg=='D' or i%2==0 or i<10):
                #     from IPython import embed; embed()
                if noise_pred_org is not None:
                    pred_sample_direction_rd = (1 - alpha_prod_t_prev - std_dev_t_rd**2) ** (0.5) * noise_pred_org
                    pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * noise_pred_org
                else:
                    pred_sample_direction_rd = (1 - alpha_prod_t_prev - std_dev_t_rd**2) ** (0.5) * noise_pred
                    pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * noise_pred

                latent_prev = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
                latent_prev_rd = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction_rd
                
                # from IPython import embed;embed()
                if repeat>1:# 这些步骤都是在中间几个step才会用到的
                    with torch.no_grad():
                        # from IPython import embed; embed()
                        alpha_prod_t = self.scheduler.alphas_cumprod[next_timestep]
                        alpha_prod_t_next = self.scheduler.alphas_cumprod[t]
                        beta_prod_t = 1 - alpha_prod_t
                        model_output = self.unet(latent_prev, next_timestep, encoder_hidden_states=text_embeddings, samplestep=i,no_save = True)["sample"]
                        # model_output = self.unet(latent_prev.unsqueeze(2), next_timestep, encoder_hidden_states=text_embeddings, mask=dict_mask, save_kv=False, mode=mode, iter_cur=-2)["sample"].squeeze(2)
                        next_original_sample = (latent_prev - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
                        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
                        latent = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
            
            latent = latent_prev
            
        return latent
    
    def guidance_move_perframe_replace_changed0827(
        self, 
        mask_x0, 
        mask_x0_ref, 
        mask_tar, 
        mask_cur, 
        mask_other, 
        mask_overlap, 
        mask_non_overlap,
        latent, 
        latent_noise_ref, 
        t, 
        # estimator,
        up_ft_index, 
        text_embeddings, 
        up_scale, 
        resize_scale, 
        energy_scale,
        w_edit,
        w_content,
        w_contrast,
        w_inpaint, 
    ):
        cos = nn.CosineSimilarity(dim=1)
        w_inpaint=10
        w_edit = 10
        w_contrast=1 # 原来是0.2
        loss_scale = [0.5, 0.5]
        bs,c,f,h,w = latent.shape
        with torch.no_grad():
            up_ft_tar = self.estimator( # src.unet.estimator.MyUNet2DConditionModel
                        sample=latent_noise_ref[0::2],
                        timestep=t,
                        up_ft_indices=up_ft_index,
                        encoder_hidden_states=text_embeddings)['up_ft'] #[[8, 1280, 32, 32,],[8, 640, 64, 64]]
            up_ft_tar_org = copy.deepcopy(up_ft_tar)
            for f_id in range(len(up_ft_tar_org)):
                up_ft_tar_org[f_id] = F.interpolate(up_ft_tar_org[f_id], (up_ft_tar_org[-1].shape[-3],up_ft_tar_org[-1].shape[-2]*up_scale, up_ft_tar_org[-1].shape[-1]*up_scale))
            
        with torch.no_grad():
            up_ft_tar_replace = self.estimator(
                        sample=latent_noise_ref[1::2] ,
                        timestep=t,
                        up_ft_indices=up_ft_index,
                        encoder_hidden_states=text_embeddings)['up_ft']
            for f_id in range(len(up_ft_tar_replace)):
                up_ft_tar_replace[f_id] = F.interpolate(up_ft_tar_replace[f_id], (up_ft_tar_replace[-1].shape[-3],up_ft_tar_replace[-1].shape[-2]*up_scale, up_ft_tar_replace[-1].shape[-1]*up_scale))
        
        
        
        
        
        latent = latent.detach().requires_grad_(True) # [1, 4, 8, 64, 64]
        for f_id in range(len(up_ft_tar)):
            up_ft_tar[f_id] = F.interpolate(up_ft_tar[f_id], (int(up_ft_tar[-1].shape[-3]),int(up_ft_tar[-1].shape[-2]*up_scale), int(up_ft_tar[-1].shape[-1]*up_scale)))
  
            

        up_ft_cur = self.estimator(
                    sample = latent,
                    timestep=t,
                    up_ft_indices=up_ft_index,
                    encoder_hidden_states=text_embeddings)['up_ft']# [[8, 1280, 32, 32],[8, 640, 64, 64]
        for f_id in range(len(up_ft_cur)):
            up_ft_cur[f_id] = F.interpolate(up_ft_cur[f_id], (up_ft_cur[-1].shape[-3],up_ft_cur[-1].shape[-2]*up_scale, up_ft_cur[-1].shape[-1]*up_scale))
        
        
        loss_edit = 0
        
        mask_cur = torch.stack(mask_cur).transpose(0,2) # [8, 1, 128, 128]
        mask_tar = torch.stack(mask_tar).transpose(0,2) # [8, 1, 128, 128]
        
        
        for f_id in range(len(up_ft_tar)):
            # print("test")
            # from IPython import embed;embed()
            up_ft_cur_vec = up_ft_cur[f_id][mask_cur.repeat(1,up_ft_cur[f_id].shape[1],1,1,1)].view(up_ft_cur[f_id].shape[1], -1).permute(1,0)
            up_ft_tar_vec = up_ft_tar_replace[f_id][mask_tar.repeat(1,up_ft_tar[f_id].shape[1],1,1,1)].view(up_ft_tar[f_id].shape[1], -1).permute(1,0)
            sim = cos(up_ft_cur_vec, up_ft_tar_vec)
            sim_global = cos(up_ft_cur_vec.mean(0, keepdim=True), up_ft_tar_vec.mean(0, keepdim=True))
            loss_edit = loss_edit + (w_edit/(1+4*sim.mean()))*loss_scale[f_id] 
        
        loss_con = 0
        
        mask_other = torch.stack(mask_other) # 8, 1, 1, 128, 128
        # 这里没问题
        for f_id in range(len(up_ft_tar_org)):
            # sim_other = cos(up_ft_tar_org[f_id], up_ft_cur[f_id])[0][mask_other[0,0]]
            sim_other = cos(up_ft_tar_org[f_id], up_ft_cur[f_id])[mask_other.transpose(0,2)[0]]
            loss_con = loss_con+w_content/(1+4*sim_other.mean())*loss_scale[f_id]
        
        
        # mask_non_overlap = rearrange(torch.stack(mask_non_overlap),"b c f h w -> (b f) c h w")
        mask_non_overlap = torch.stack(mask_non_overlap).transpose(0,2)

        # print("test1")
        # from IPython import embed;embed()
        cond_grad_edit = torch.autograd.grad(loss_edit*energy_scale, latent, retain_graph=True)[0]
        cond_grad_con = torch.autograd.grad(loss_con*energy_scale, latent)[0]
        
        mask_x0 = torch.stack(mask_x0) # 8, 128, 128
        mask_edit2 = (F.interpolate(mask_x0[None], (mask_cur.shape[-2], mask_cur.shape[-1]))>0.5).float() # torch.Size([1, 8, 128, 128])
        mask_edit1 = (mask_cur>0.5).float().transpose(0,1) # # torch.Size([1, 8, 128, 128])
        # from IPython import embed;embed()
        mask_cur = mask_cur.transpose(0,1)
        mask = ((mask_cur+mask_edit2)>0.5).float()
        mask_edit1 = (F.interpolate(mask_edit1, (latent.shape[-3],latent.shape[-2], latent.shape[-1]))>0).to(dtype=latent.dtype)
        
        # guidance = cond_grad_edit.detach()*2e-1*mask_edit1 + cond_grad_con.detach()*8e-2*(1-mask_edit1)
        guidance = cond_grad_edit.detach()*8e-2*mask_edit1 + cond_grad_con.detach()*8e-2*(1-mask_edit1)
        self.estimator.zero_grad()

        return guidance
    
    def guidance_move_perframe_residual2(
        self, 
        mask_x0, 
        mask_x0_ref, 
        mask_tar, 
        mask_cur, 
        mask_other, 
        mask_overlap, 
        mask_non_overlap,
        latent, 
        latent_noise_ref, 
        t, 
        # estimator,
        up_ft_index, 
        text_embeddings, 
        up_scale, 
        resize_scale, 
        energy_scale,
        w_edit,
        w_content,
        w_contrast,
        w_inpaint, 
    ):
        cos = nn.CosineSimilarity(dim=1)
        w_inpaint=10
        w_edit = 10
        w_contrast=1 # 原来是0.2
        loss_scale = [0.5, 0.5]
        bs,c,f,h,w = latent.shape
        with torch.no_grad():
            up_ft_tar = self.estimator( # src.unet.estimator.MyUNet2DConditionModel
                        sample=latent_noise_ref[0::2],
                        timestep=t,
                        up_ft_indices=up_ft_index,
                        encoder_hidden_states=text_embeddings)['up_ft'] #[[8, 1280, 32, 32,],[8, 640, 64, 64]]
            up_ft_tar_org = copy.deepcopy(up_ft_tar)
            # for f_id in range(len(up_ft_tar_org)):
            #     up_ft_tar_org[f_id] = F.interpolate(up_ft_tar_org[f_id], (up_ft_tar_org[-1].shape[-3],up_ft_tar_org[-1].shape[-2]*up_scale, up_ft_tar_org[-1].shape[-1]*up_scale))
            
        with torch.no_grad():
            up_ft_tar_replace = self.estimator(
                        sample=latent_noise_ref[1::2] ,
                        timestep=t,
                        up_ft_indices=up_ft_index,
                        encoder_hidden_states=text_embeddings)['up_ft']
            # for f_id in range(len(up_ft_tar_replace)):
            #     up_ft_tar_replace[f_id] = F.interpolate(up_ft_tar_replace[f_id], (up_ft_tar_replace[-1].shape[-3],up_ft_tar_replace[-1].shape[-2]*up_scale, up_ft_tar_replace[-1].shape[-1]*up_scale))
        
        latent = latent.detach().requires_grad_(True) # [1, 4, 8, 64, 64]
        # for f_id in range(len(up_ft_tar)):
        #     up_ft_tar[f_id] = F.interpolate(up_ft_tar[f_id], (int(up_ft_tar[-1].shape[-3]),int(up_ft_tar[-1].shape[-2]*up_scale), int(up_ft_tar[-1].shape[-1]*up_scale)))

        up_ft_cur = self.estimator(
                    sample = latent,
                    timestep=t,
                    up_ft_indices=up_ft_index,
                    encoder_hidden_states=text_embeddings)['up_ft']# [[8, 1280, 32, 32],[8, 640, 64, 64]
        # for f_id in range(len(up_ft_cur)):
        #     up_ft_cur[f_id] = F.interpolate(up_ft_cur[f_id], (up_ft_cur[-1].shape[-3],up_ft_cur[-1].shape[-2]*up_scale, up_ft_cur[-1].shape[-1]*up_scale))
        
        
        mask_cur = torch.stack(mask_cur).transpose(0,2) # [8, 1, 128, 128]
        mask_tar = torch.stack(mask_tar).transpose(0,2) # [8, 1, 128, 128]    
        loss_edit_t = 0
        loss_consistency_t = 0
    
        for f_id in range(len(up_ft_tar)):
            
            mask_cur_fid = F.interpolate(mask_cur.float(),(up_ft_cur[f_id].shape[-3],up_ft_cur[f_id].shape[-2],up_ft_cur[f_id].shape[-1]))>0.5
            # mask_ref = F.interpolate(mask_ref_origin.float().unsqueeze(0).unsqueeze(0),(up_ft_cur[f_id].shape[-3],up_ft_cur[f_id].shape[-2],up_ft_cur[f_id].shape[-1]))>0.5
            # up_ft_tar
            frame_maskmean_cur = []
            frame_maskmean_cur_back = []
            frame_maskmean_tar = []
            frame_maskmean_tar_replace = []
            # mask_ref_temp = mask_ref.repeat(1,up_ft_cur[f_id].shape[1],1,1,1) # [1, 1280, 16, 128, 128]
            # mask_ref_temp_changed = rearrange(mask_ref_temp,"b c f h w -> (b f) c h w") # [16, 1280, 128, 128]
            mask_cur_fid = mask_cur_fid.repeat(1,up_ft_cur[f_id].shape[1],1,1,1)
            mask_cur_fid = rearrange(mask_cur_fid,"b c f h w -> (b f) c h w") 
            feature_cur = rearrange(up_ft_cur[f_id],"b c f h w -> (b f) c h w")
            feature_tar = rearrange(up_ft_tar[f_id],"b c f h w -> (b f) c h w")
            feature_tar_replace = rearrange(up_ft_tar_replace[f_id],"b c f h w -> (b f) c h w")
            # from IPython import embed;embed()
            for frame_id in range(bs*f):
                frame_maskmean_cur.append(feature_cur[frame_id][mask_cur_fid[frame_id]].view(up_ft_cur[f_id].shape[1],-1).transpose(0,1))   
                frame_maskmean_tar_replace.append(feature_tar_replace[frame_id][mask_cur_fid[frame_id]].view(up_ft_tar_replace[f_id].shape[1],-1).transpose(0,1))
                frame_maskmean_tar.append(feature_tar[frame_id][~mask_cur_fid[frame_id]].view(up_ft_tar[f_id].shape[1],-1).transpose(0,1))
                frame_maskmean_cur_back.append(feature_cur[frame_id][~mask_cur_fid[frame_id]].view(up_ft_cur[f_id].shape[1],-1).transpose(0,1))
                # frame_maskmean_tar_replace.append(feature_tar_replace[frame_id][mask_cur_fid[frame_id]].view(up_ft_tar[f_id].shape[1],-1))
                
            

            
            # frame_maskmean_cur = torch.stack(frame_maskmean_cur,dim=0)
            
            # frame_maskmean_tar = torch.stack(frame_maskmean_tar,dim=0)
            
            temporal_relation_matrix_cur = compute_normalized_correlation(frame_maskmean_cur)
            temporal_relation_matrix_tar_replace = compute_normalized_correlation(frame_maskmean_tar_replace)
            temporal_loss_edit = compute_list_similarity(temporal_relation_matrix_cur,temporal_relation_matrix_tar_replace)
            
            
            temporal_relation_matrix_cur_back = compute_normalized_correlation(frame_maskmean_cur_back)
            temporal_relation_matrix_tar = compute_normalized_correlation(frame_maskmean_tar)
            temporal_loss_consistency = compute_list_similarity(temporal_relation_matrix_cur_back,temporal_relation_matrix_tar)
            
            
            loss_temp_edit = sum(temporal_loss_edit)/len(temporal_loss_edit)
            loss_edit_t = loss_edit_t + (w_edit/(1+4*(loss_temp_edit)))*loss_scale[f_id] #temporal_loss
            
            loss_temp_consistency = sum(temporal_loss_consistency)/len(temporal_loss_consistency)
            loss_consistency_t= loss_consistency_t + (w_content/(1+4*(loss_temp_consistency)))*loss_scale[f_id] #temporal_loss
            
            # for
        # print("test temporal")
        # from IPython import embed;embed()
        cond_grad_edit = torch.autograd.grad(loss_edit_t*energy_scale, latent, retain_graph=True)[0]
        cond_grad_con = torch.autograd.grad(loss_consistency_t*energy_scale, latent)[0]
        mask_edit1 = (mask_cur>0.5).float().transpose(0,1) # # torch.Size([1, 8, 128, 128])
        # from IPython import embed;embed()
        # mask_cur = mask_cur.transpose(0,1)
        # mask = ((mask_cur+mask_edit2)>0.5).float()
        mask_edit1 = (F.interpolate(mask_edit1, (latent.shape[-3],latent.shape[-2], latent.shape[-1]))>0).to(dtype=latent.dtype)
        
        guidance = cond_grad_edit.detach()*8e-2*mask_edit1 + cond_grad_con.detach()*8e-2*(1-mask_edit1)
        self.estimator.zero_grad()

        return guidance
    
    
    def guidance_move_perframe_replace(
        self, 
        mask_x0, 
        mask_x0_ref, 
        mask_tar, 
        mask_cur, 
        mask_other, 
        mask_overlap, 
        mask_non_overlap,
        latent, 
        latent_noise_ref, 
        t, 
        # estimator,
        up_ft_index, 
        text_embeddings, 
        up_scale, 
        resize_scale, 
        energy_scale,
        w_edit,
        w_content,
        w_contrast,
        w_inpaint, 
    ):
        cos = nn.CosineSimilarity(dim=1)
        w_inpaint=10
        w_edit = 10
        w_contrast=1 # 原来是0.2
        loss_scale = [0.5, 0.5]
        bs,c,f,h,w = latent.shape
        with torch.no_grad():
            up_ft_tar = self.estimator( # src.unet.estimator.MyUNet2DConditionModel
                        sample=rearrange(latent_noise_ref.squeeze(2)[1::2], "b c f h w -> (b f) c h w") ,
                        # sample=latent_noise_ref.squeeze(2)[::2].transpose(0,2).squeeze(2),
                        timestep=t,
                        up_ft_indices=up_ft_index,
                        encoder_hidden_states=text_embeddings.repeat(bs*f,1,1))['up_ft'] #[[8, 1280, 32, 32,],[8, 640, 64, 64]]
            up_ft_tar_org = copy.deepcopy(up_ft_tar)
            for f_id in range(len(up_ft_tar_org)):
                up_ft_tar_org[f_id] = F.interpolate(up_ft_tar_org[f_id], (up_ft_tar_org[-1].shape[-2]*up_scale, up_ft_tar_org[-1].shape[-1]*up_scale))
            
        with torch.no_grad():
            up_ft_tar_replace = self.estimator(
                        sample=rearrange(latent_noise_ref.squeeze(2)[1::2], "b c f h w -> (b f) c h w") ,
                        # sample=latent_noise_ref.squeeze(2)[1::2],
                        timestep=t,
                        up_ft_indices=up_ft_index,
                        encoder_hidden_states=text_embeddings.repeat(bs*f,1,1))['up_ft']
            for f_id in range(len(up_ft_tar_replace)):
                up_ft_tar_replace[f_id] = F.interpolate(up_ft_tar_replace[f_id], (up_ft_tar_replace[-1].shape[-2]*up_scale, up_ft_tar_replace[-1].shape[-1]*up_scale))
        
        
        
        
        latent = latent.detach().requires_grad_(True) # [1, 4, 8, 64, 64]
        for f_id in range(len(up_ft_tar)):
            up_ft_tar[f_id] = F.interpolate(up_ft_tar[f_id], (int(up_ft_tar[-1].shape[-2]*up_scale), int(up_ft_tar[-1].shape[-1]*up_scale)))
  
            

        up_ft_cur = self.estimator(
                    sample=rearrange(latent,"b c f h w -> (b f) c h w"),
                    timestep=t,
                    up_ft_indices=up_ft_index,
                    encoder_hidden_states=text_embeddings.repeat(bs*f,1,1))['up_ft']# [[8, 1280, 32, 32],[8, 640, 64, 64]
        for f_id in range(len(up_ft_cur)):
            up_ft_cur[f_id] = F.interpolate(up_ft_cur[f_id], (up_ft_cur[-1].shape[-2]*up_scale, up_ft_cur[-1].shape[-1]*up_scale))
        
        
        # editing energy
        loss_edit = 0
        mask_cur = torch.stack(mask_cur).squeeze(1) # [8, 1, 128, 128]
        mask_tar = torch.stack(mask_tar).squeeze(1) # [8, 1, 128, 128]
        
        # print("test")
        # from IPython import embed;embed()
        
        for f_id in range(len(up_ft_tar)):
            up_ft_cur_vec = up_ft_cur[f_id][mask_cur.repeat(1,up_ft_cur[f_id].shape[1],1,1)].view(up_ft_cur[f_id].shape[1], -1).permute(1,0)
            up_ft_tar_vec = up_ft_tar[f_id][mask_tar.repeat(1,up_ft_tar[f_id].shape[1],1,1)].view(up_ft_tar[f_id].shape[1], -1).permute(1,0)
            sim = cos(up_ft_cur_vec, up_ft_tar_vec)
            sim_global = cos(up_ft_cur_vec.mean(0, keepdim=True), up_ft_tar_vec.mean(0, keepdim=True))
            loss_edit = loss_edit + (w_edit/(1+4*sim.mean()))*loss_scale[f_id] 
        # print("test")
        # from IPython import embed;embed()
        # content energy
        loss_con = 0
        # if mask_x0_ref is not None and mask_x0_ref[0] is not None:
        #     mask_x0_ref_cur = F.interpolate(mask_x0_ref[None,None], (mask_other.shape[-2], mask_other.shape[-1]))>0.5
        # else:
        #     mask_x0_ref_cur = mask_other
        
        mask_other = torch.stack(mask_other) # 8, 1, 1, 128, 128
        # 这里没问题
        for f_id in range(len(up_ft_tar_org)):
            # sim_other = cos(up_ft_tar_org[f_id], up_ft_cur[f_id])[0][mask_other[0,0]]
            sim_other = cos(up_ft_tar_org[f_id], up_ft_cur[f_id])[mask_other[:,0,0]]
            loss_con = loss_con+w_content/(1+4*sim_other.mean())*loss_scale[f_id]
        
        
        mask_non_overlap = rearrange(torch.stack(mask_non_overlap),"b c f h w -> (b f) c h w")
        
        print("test")
        from IPython import embed;embed()
        
        #opt部分
        for f_id in range(len(up_ft_tar)):
            up_ft_cur_non_overlap = up_ft_cur[f_id][mask_non_overlap.repeat(1,up_ft_cur[f_id].shape[1],1,1)].view(up_ft_cur[f_id].shape[1], -1).permute(1,0)
            up_ft_tar_non_overlap = up_ft_tar_org[f_id][mask_non_overlap.repeat(1,up_ft_tar_org[f_id].shape[1],1,1)].view(up_ft_tar_org[f_id].shape[1], -1).permute(1,0)
            sim_non_overlap = (cos(up_ft_cur_non_overlap, up_ft_tar_non_overlap)+1.)/2.
            loss_con = loss_con + w_contrast*sim_non_overlap.mean()*loss_scale[f_id]

            up_ft_cur_non_overlap = up_ft_cur[f_id][mask_non_overlap.repeat(1,up_ft_cur[f_id].shape[1],1,1)].view(up_ft_cur[f_id].shape[1],-1).permute(1,0).mean(0, keepdim=True)
            up_ft_tar_non_overlap = up_ft_tar_replace[f_id][mask_non_overlap.repeat(1,up_ft_tar_org[f_id].shape[1],1,1)].view(up_ft_tar_org[f_id].shape[1],-1).permute(1,0).mean(0, keepdim=True)
            sim_inpaint = ((cos(up_ft_cur_non_overlap, up_ft_tar_non_overlap)+1.)/2.)
            loss_con = loss_con + w_inpaint/(1+4*sim_inpaint.mean())

        
        cond_grad_edit = torch.autograd.grad(loss_edit*energy_scale, latent, retain_graph=True)[0]
        cond_grad_con = torch.autograd.grad(loss_con*energy_scale, latent)[0]
        
        mask_x0 = torch.stack(mask_x0) # 8, 128, 128
        mask_edit2 = (F.interpolate(mask_x0[None], (mask_cur.shape[-2], mask_cur.shape[-1]))>0.5).float() # torch.Size([1, 8, 128, 128])
        mask_edit1 = (mask_cur>0.5).float().transpose(0,1) # # torch.Size([1, 8, 128, 128])
        # from IPython import embed;embed()
        mask_cur = mask_cur.transpose(0,1)
        mask = ((mask_cur+mask_edit2)>0.5).float()
        mask_edit1 = (F.interpolate(mask_edit1, (latent.shape[-2], latent.shape[-1]))>0).to(dtype=latent.dtype)
        
        guidance = cond_grad_edit.detach()*2e-1*mask_edit1 + cond_grad_con.detach()*8e-2*(1-mask_edit1)
        self.estimator.zero_grad()

        return guidance
    
    
    def guidance_move_perframe_residual(
        self, 
        mask_x0, 
        mask_x0_ref, 
        mask_tar, 
        mask_cur, 
        mask_other, 
        mask_overlap, 
        mask_non_overlap,
        latent, 
        latent_noise_ref, 
        t, 
        # estimator,
        up_ft_index, 
        text_embeddings, 
        up_scale, 
        resize_scale, 
        energy_scale,
        w_edit,
        w_content,
        w_contrast,
        w_inpaint, 
    ):
        cos = nn.CosineSimilarity(dim=1)
        w_inpaint=10
        w_edit = 10
        w_contrast=1 # 原来是0.2
        loss_scale = [0.5, 0.5]
        bs,c,f,h,w = latent.shape
        with torch.no_grad():
            up_ft_tar = self.estimator( # src.unet.estimator.MyUNet2DConditionModel
                        sample=latent_noise_ref[0::2] ,
                        # sample=latent_noise_ref.squeeze(2)[::2].transpose(0,2).squeeze(2),
                        timestep=t,
                        up_ft_indices=up_ft_index,
                        encoder_hidden_states=text_embeddings)['up_ft'] #[[8, 1280, 32, 32,],[8, 640, 64, 64]]
            up_ft_tar_org = copy.deepcopy(up_ft_tar)
            for f_id in range(len(up_ft_tar_org)):
                up_ft_tar_org[f_id] = F.interpolate(up_ft_tar_org[f_id], (up_ft_tar_org[-1].shape[-3],up_ft_tar_org[-1].shape[-2]*up_scale, up_ft_tar_org[-1].shape[-1]*up_scale))
            
        
        
        # latent = latent.detach().requires_grad_(True) # [1, 4, 8, 64, 64]
        # for f_id in range(len(up_ft_tar)):
        #     up_ft_tar[f_id] = F.interpolate(up_ft_tar[f_id], (int(up_ft_tar[-1].shape[-2]*up_scale), int(up_ft_tar[-1].shape[-1]*up_scale)))

        latent = latent.detach().requires_grad_(True) # [1, 4, 8, 64, 64]
        for f_id in range(len(up_ft_tar)):
            up_ft_tar[f_id] = F.interpolate(up_ft_tar[f_id], (int(up_ft_tar[-1].shape[-3]),int(up_ft_tar[-1].shape[-2]*up_scale), int(up_ft_tar[-1].shape[-1]*up_scale)))
  
            
            

        up_ft_cur = self.estimator(
                    sample=latent,
                    timestep=t,
                    up_ft_indices=up_ft_index,
                    encoder_hidden_states=text_embeddings)['up_ft']# [[8, 1280, 32, 32],[8, 640, 64, 64]
        for f_id in range(len(up_ft_cur)):
            up_ft_cur[f_id] = F.interpolate(up_ft_cur[f_id], (up_ft_cur[-1].shape[-3],up_ft_cur[-1].shape[-2]*up_scale, up_ft_cur[-1].shape[-1]*up_scale))
        
        
        
        mask_ref_origin = torch.stack(mask_x0_ref)
        # mask_ref = F.interpolate(mask_ref.float().unsqueeze(1),(up_ft_cur[-1].shape[-2]*up_scale, up_ft_cur[-1].shape[-1]*up_scale))
        
        mask_cur = torch.stack(mask_cur).transpose(0,2) # [8, 1, 128, 128]
        mask_tar = torch.stack(mask_tar).transpose(0,2) # [8, 1, 128, 128]
        
        loss_temp = 0
        # print("test_temporal_residual")
        # from IPython import embed;embed()
        eps= 1e-8
        features_diff_loss = 0
        
        for f_id in range(len(up_ft_tar)):
            
            mask_ref = F.interpolate(mask_ref_origin.float().unsqueeze(0).unsqueeze(0),(up_ft_cur[f_id].shape[-3],up_ft_cur[f_id].shape[-2],up_ft_cur[f_id].shape[-1]))>0.5
            frame_maskmean_cur = []
            frame_maskmean_tar = []
            # rearrange(latent_noise_ref.squeeze(2)[0::2], "b c f h w -> (b f) c h w")
            # print("test1")
            # from IPython import embed;embed()
            mask_ref_temp = mask_ref.repeat(1,up_ft_cur[f_id].shape[1],1,1,1) # [1, 1280, 16, 128, 128]
            mask_ref_temp_changed = rearrange(mask_ref_temp,"b c f h w -> (b f) c h w") # [16, 1280, 128, 128]
            feature_cur = rearrange(up_ft_cur[f_id],"b c f h w -> (b f) c h w")
            feature_tar = rearrange(up_ft_tar[f_id],"b c f h w -> (b f) c h w")
            for frame_id in range(bs*f):
                # print("test_temporal_residual_1")
                # from IPython import embed;embed()
                # up_ft_cur[f_id]  torch.Size([1, 1280, 16, 128, 128])
                frame_maskmean_cur.append(feature_cur[frame_id][mask_ref_temp_changed[frame_id]].view(up_ft_cur[f_id].shape[1],-1).mean(-1))   
                frame_maskmean_tar.append(feature_tar[frame_id][mask_ref_temp_changed[frame_id]].view(up_ft_tar[f_id].shape[1],-1).mean(-1))

                # frame_maskmean_cur.append(up_ft_cur[f_id][frame_id].view(up_ft_cur[f_id].shape[1],-1).mean(-1))
                # frame_maskmean_tar.append(up_ft_tar[f_id][frame_id].view(up_ft_tar[f_id].shape[1],-1).mean(-1))
            # print("test_temporal_residual")
            from IPython import embed;embed()
                
            # frame_maskmean_cur_before = torch.stack([frame_maskmean_cur[0]]+frame_maskmean_cur[:-1],dim=0)
            frame_maskmean_cur = torch.stack(frame_maskmean_cur,dim=0)
            
            # frame_diff_cur =  frame_maskmean_cur-frame_maskmean_cur_before
            
            # frame_maskmean_tar_before = torch.stack([frame_maskmean_tar[0]]+frame_maskmean_tar[:-1],dim=0)
            frame_maskmean_tar = torch.stack(frame_maskmean_tar,dim=0)
            
            
            for i in range(len(frame_maskmean_cur)):
                
                frame_diff_tar = frame_maskmean_tar-frame_maskmean_tar[i]
                frame_diff_cur = frame_maskmean_cur-frame_maskmean_cur[i]
                # from IPython import embed;embed()
                sim =F.cosine_similarity(frame_diff_cur,frame_diff_tar.detach(), dim=1,eps=1e-3).mean()
                features_diff_loss+=sim
                # sim_filtered = torch.cat((sim[:i], sim[i+1:]))
                # # sim[i]=
                # sim = sim_filtered.mean()
                
                # F.cosine_similarity(frame_diff_cur,frame_diff_tar.detach(), dim=0)
                
            
            #     features_diff_loss+=sim
            # print("test_temporal_residual")
            # from IPython import embed;embed() 
            
            # frame_diff_tar =  frame_maskmean_tar-frame_maskmean_tar_before
            # sim = cos(frame_diff_cur,frame_diff_tar)[1:]
            
            
            
            # up_ft_cur_vec = up_ft_cur[f_id][mask_.repeat(1,up_ft_cur[f_id].shape[1],1,1)].view(up_ft_cur[f_id].shape[1], -1).permute(1,0)
            
            # up_ft_cur_vec = up_ft_cur[f_id][mask_ref.repeat(1,up_ft_cur[f_id].shape[1],1,1)].view(f,up_ft_cur[f_id].shape[1], -1).permute(1,0)
            # up_ft_tar_vec = up_ft_tar[f_id][mask_tar.repeat(1,up_ft_tar[f_id].shape[1],1,1)].view(up_ft_tar[f_id].shape[1], -1).permute(1,0)
            # sim = cos(up_ft_cur_vec, up_ft_tar_vec)
            # sim_global = cos(up_ft_cur_vec.mean(0, keepdim=True), up_ft_tar_vec.mean(0, keepdim=True))
            loss_temp = loss_temp + (w_edit/(1+4*(features_diff_loss/len(frame_maskmean_cur))))*loss_scale[f_id] 
        
            # with torch.autograd.detect_anomaly():
        cond_grad_temp = torch.autograd.grad(loss_temp*energy_scale, latent, retain_graph=True)[0]
        
        # print("test_temporal_residual_2")
        # from IPython import embed;embed()
        
        # mask_other = torch.stack(mask_other) # 8, 1, 1, 128, 128
        # 这里没问题
        loss_con = 0
        #mask_ref.shape 1, 1, 16, 128, 128 bool
        
        for f_id in range(len(up_ft_tar_org)):
            # sim_other = cos(up_ft_tar_org[f_id], up_ft_cur[f_id])[0][mask_other[0,0]]
            # cos(up_ft_tar_org[f_id], up_ft_cur[f_id])torch.Size([1, 16, 128, 128])
            sim_other = cos(up_ft_tar_org[f_id], up_ft_cur[f_id])[(~mask_ref)[0]]
            loss_con = loss_con+w_content/(1+4*sim_other.mean())*loss_scale[f_id]
        
        cond_grad_con = torch.autograd.grad(loss_con*energy_scale, latent)[0]
        
        
        
        # print("test_temporal_residual_2")
        # from IPython import embed;embed()
        mask_ref =  F.interpolate(mask_ref_origin.float().unsqueeze(1),(cond_grad_temp.shape[-2],cond_grad_temp.shape[-1]))>0.5
        # torch.Size([16, 1, 64, 64])
        mask_ref = mask_ref.transpose(0,1).unsqueeze(0).repeat(1,4,1,1,1) # 1, 4, 16, 64, 64
        # print("test_temporal_residual_2")
        # from IPython import embed;embed()
        # guidance = cond_grad_temp.detach()*8e-2*mask_ref+cond_grad_con.detach()*8e-2*(~mask_ref)
        guidance = cond_grad_temp.detach()*8e-2
        # guidance = guidance
        
        # editing energy
        # loss_edit = 0
        # mask_cur = torch.stack(mask_cur).squeeze(1) # [8, 1, 128, 128]
        # mask_tar = torch.stack(mask_tar).squeeze(1) # [8, 1, 128, 128]
        
        # guidance  =  None
        
        self.estimator.zero_grad()

        return guidance
    
    
    def guidance_temporal(
        self, 
        mask_x0, 
        mask_x0_ref, 
        mask_tar, 
        mask_cur, 
        mask_other, 
        mask_overlap, 
        mask_non_overlap,
        latent, 
        latent_noise_ref, 
        t, 
        # estimator,
        up_ft_index, 
        text_embeddings, 
        up_scale, 
        resize_scale, 
        energy_scale,
        w_edit,
        w_content,
        w_contrast,
        w_inpaint, 
    ):
        
        cos = nn.CosineSimilarity(dim=0)
        latent = latent.detach().requires_grad_(True)
        prior_latent = latent.clone()
        for i in range(1, latent.size(2)):
            prior_latent[:, :, i, :, :] = latent[:, :, i - 1, :, :]
        
        mask_non_overlap = rearrange(torch.stack(mask_non_overlap),"b c f h w -> (b f) c h w")
        mask_other = rearrange(mask_other,"f c b h w -> (b c) f h w")
        mask_edit = (1 - (mask_other > 0.5).float()) > 0.5
        b, c, f, h, w =latent.shape
        mask_edit = (F.interpolate(mask_edit.float(), (latent.shape[-2], latent.shape[-1]))>0.5)# torch.Size([1, 8, 128, 128])
        mask_edit_new = mask_edit.clone()
        
        w_temp = 10 
        loss_temp = 0
        
        
        for i in range(f-1):
            mask_edit_tmp1,mask_edit_tmp2 = main(mask_edit[0][i].unsqueeze(0).unsqueeze(0),mask_edit[0][i+1].unsqueeze(0).unsqueeze(0))
            # print("test temporal")
            # from IPython import embed;embed()
            mask_edit_tmp0,mask_edit_tmp2_ = main(mask_edit[0][0].unsqueeze(0).unsqueeze(0),mask_edit[0][i+1].unsqueeze(0).unsqueeze(0))
            mask_edit_tmp0 = (mask_edit_tmp0>0).repeat(1,4,1,1)
            mask_edit_tmp1 = (mask_edit_tmp1>0).repeat(1,4,1,1)
            mask_edit_tmp2 = (mask_edit_tmp2>0).repeat(1,4,1,1)
            mask_edit_tmp2_ = (mask_edit_tmp2_>0).repeat(1,4,1,1)
            
            latent1 = latent[:,:,i][mask_edit_tmp1]
            latent2 = latent[:,:,i+1][mask_edit_tmp2]
            latent0 = latent[:,:,0][mask_edit_tmp0]
            latent2_ = latent[:,:,0][mask_edit_tmp2_]
            loss_temp = loss_temp+cos(latent1, latent2)+cos(latent0, latent2_)
            # loss_temp = loss_temp+cos(latent1, latent2)+cos(latent0, latent2_)
            
        
        loss_temp = w_temp/(1+4*loss_temp/(2*(f-1)))
        cond_grad_temp = torch.autograd.grad(loss_temp*energy_scale, latent, retain_graph=True)[0]
        guidance = cond_grad_temp.detach()*mask_edit*8e-2
        
            

        return guidance
    
    def guidance_temporal_residual(
        self, 
        mask_x0, 
        mask_x0_ref, 
        mask_tar, 
        mask_cur, 
        mask_other, 
        mask_overlap, 
        mask_non_overlap,
        latent, 
        latent_noise_ref, 
        t, 
        # estimator,
        up_ft_index, 
        text_embeddings, 
        up_scale, 
        resize_scale, 
        energy_scale,
        w_edit,
        w_content,
        w_contrast,
        w_inpaint, 
    ):
        
        cos = nn.CosineSimilarity(dim=0)
        latent = latent.detach().requires_grad_(True)
        prior_latent = latent.clone()
        for i in range(1, latent.size(2)):
            prior_latent[:, :, i, :, :] = latent[:, :, i - 1, :, :]
        
        mask_non_overlap = rearrange(torch.stack(mask_non_overlap),"b c f h w -> (b f) c h w")
        mask_other = rearrange(mask_other,"f c b h w -> (b c) f h w")
        mask_edit = (1 - (mask_other > 0.5).float()) > 0.5
        b, c, f, h, w =latent.shape
        mask_edit = (F.interpolate(mask_edit.float(), (latent.shape[-2], latent.shape[-1]))>0.5)# torch.Size([1, 8, 128, 128])
        mask_edit_new = mask_edit.clone()
        
        mask_ref = torch.stack(mask_x0_ref)
        mask_ref = F.interpolate(mask_ref.float().unsqueeze(1),(latent.shape[-2], latent.shape[-1]))
        
        w_temp = 10 
        loss_temp = 0
        
        
        for i in range(f-1):
            # latent1 = latent[:,:,i][mask_edit[0][i]]
            latent_origin1 =  latent[:,:,i].view(1,4,-1)
            mask_ref1 = mask_ref[i].view(1,-1)>0.5
            latent1 = latent_origin1[:,:,mask_ref1[0]].mean(-1)
            latent_origin2 =  latent[:,:,i+1].view(1,4,-1)
            mask_ref2 = mask_ref[i+1].view(1,-1)>0.5
            latent2 = latent_origin2[:,:,mask_ref2[0]].mean(-1)
            
        
        loss_temp = w_temp/(1+4*loss_temp/(2*(f-1)))
        cond_grad_temp = torch.autograd.grad(loss_temp*energy_scale, latent, retain_graph=True)[0]
        guidance = cond_grad_temp.detach()*mask_edit*8e-2
        
            

        return guidance
    
    # v1
    def guidance_move_perframe(
        self, 
        mask_x0, 
        mask_x0_ref, 
        mask_tar, 
        mask_cur, 
        mask_other, 
        mask_overlap, 
        mask_non_overlap,
        latent, 
        latent_noise_ref, 
        t, 
        # estimator,
        up_ft_index, 
        text_embeddings, 
        up_scale, 
        resize_scale, 
        energy_scale,
        w_edit,
        w_content,
        w_contrast,
        w_inpaint, 
    ):
        cos = nn.CosineSimilarity(dim=1)
        w_inpaint=10
        w_edit = 10
        w_contrast=1 # 原来是0.2
        loss_scale = [0.5, 0.5]
        bs,c,f,h,w = latent.shape
        with torch.no_grad():
            up_ft_tar = self.estimator( # src.unet.estimator.MyUNet2DConditionModel
                        sample=rearrange(latent_noise_ref.squeeze(2)[::2], "b c f h w -> (b f) c h w") ,
                        # sample=latent_noise_ref.squeeze(2)[::2].transpose(0,2).squeeze(2),
                        timestep=t,
                        up_ft_indices=up_ft_index,
                        encoder_hidden_states=text_embeddings.repeat(bs*f,1,1))['up_ft'] #[[8, 1280, 32, 32,],[8, 640, 64, 64]]
            up_ft_tar_org = copy.deepcopy(up_ft_tar)
            for f_id in range(len(up_ft_tar_org)):
                up_ft_tar_org[f_id] = F.interpolate(up_ft_tar_org[f_id], (up_ft_tar_org[-1].shape[-2]*up_scale, up_ft_tar_org[-1].shape[-1]*up_scale))
            # up_ft_tar_org ([8, 1280, 128, 128],[8, 640, 128, 128])
            # for f_id in range(len(up_ft_tar_org)):
            #     # up_ft_tar_org[f_id] = F.interpolate(up_ft_tar_org[f_id], (up_ft_tar_org[-1].shape[-2]*up_scale, up_ft_tar_org[-1].shape[-1]*up_scale))
            #     d1,d2,d3,d4,d5 = up_ft_tar_org[f_id].shape
            #     up_ft_tar_org[f_id] = up_ft_tar_org[f_id].view(d1*d2,d3,d4,d5)
            #     up_ft_tar_org[f_id] = F.interpolate(up_ft_tar_org[f_id], (up_ft_tar_org[-1].shape[-2]*up_scale, up_ft_tar_org[-1].shape[-1]*up_scale)) #1280, 8, 32, 32->1280, 8, 128, 128
            #     _,d3,d4,d5 = up_ft_tar_org[f_id].shape
            #     up_ft_tar_org[f_id] = up_ft_tar_org[f_id].view(d1,d2,d3,d4,d5)
        # from IPython import embed;embed()
        with torch.no_grad():
            up_ft_tar_replace = self.estimator(
                        sample=rearrange(latent_noise_ref.squeeze(2)[1::2], "b c f h w -> (b f) c h w") ,
                        # sample=latent_noise_ref.squeeze(2)[1::2],
                        timestep=t,
                        up_ft_indices=up_ft_index,
                        encoder_hidden_states=text_embeddings.repeat(bs*f,1,1))['up_ft']
            for f_id in range(len(up_ft_tar_replace)):
                up_ft_tar_replace[f_id] = F.interpolate(up_ft_tar_replace[f_id], (up_ft_tar_replace[-1].shape[-2]*up_scale, up_ft_tar_replace[-1].shape[-1]*up_scale))
             # up_ft_tar_replace ([8, 1280, 128, 128],[8, 640, 128, 128])
                # d1,d2,d3,d4,d5 = up_ft_tar_replace[f_id].shape
                # up_ft_tar_replace[f_id] = up_ft_tar_replace[f_id].view(d1*d2,d3,d4,d5)
                # up_ft_tar_replace[f_id] = F.interpolate(up_ft_tar_replace[f_id], (up_ft_tar_replace[-1].shape[-2]*up_scale, up_ft_tar_replace[-1].shape[-1]*up_scale)) #1280, 8, 32, 32->1280, 8, 128, 128
                # _,d3,d4,d5 = up_ft_tar_replace[f_id].shape
                # up_ft_tar_replace[f_id] = up_ft_tar_replace[f_id].view(d1,d2,d3,d4,d5)
        
        
        
        latent = latent.detach().requires_grad_(True) # [1, 4, 8, 64, 64]
        for f_id in range(len(up_ft_tar)):
            up_ft_tar[f_id] = F.interpolate(up_ft_tar[f_id], (int(up_ft_tar[-1].shape[-2]*up_scale), int(up_ft_tar[-1].shape[-1]*up_scale)))
        
        # for f_id in range(len(up_ft_tar)):
        #     up_ft_tar[f_id] = F.interpolate(up_ft_tar[f_id], (int(up_ft_tar[-1].shape[-2]*resize_scale*up_scale), int(up_ft_tar[-1].shape[-1]*resize_scale*up_scale)))
        #up_ft_tar ([8, 1280, 128, 128],[8, 640, 128, 128])
        # for f_id in range(len(up_ft_tar)):
        #     d1,d2,d3,d4,d5 = up_ft_tar[f_id].shape
        #     up_ft_tar[f_id] = up_ft_tar[f_id].view(d1*d2,d3,d4,d5)
        #     up_ft_tar[f_id] = F.interpolate(up_ft_tar[f_id], (int(up_ft_tar[-1].shape[-2]*resize_scale*up_scale), int(up_ft_tar[-1].shape[-1]*resize_scale*up_scale)))
        #     _,d3,d4,d5 = up_ft_tar[f_id].shape
        #     up_ft_tar[f_id] = up_ft_tar[f_id].view(d1,d2,d3,d4,d5)
            

        up_ft_cur = self.estimator(
                    sample=rearrange(latent,"b c f h w -> (b f) c h w"),
                    timestep=t,
                    up_ft_indices=up_ft_index,
                    encoder_hidden_states=text_embeddings.repeat(bs*f,1,1))['up_ft']# [[8, 1280, 32, 32],[8, 640, 64, 64]
        for f_id in range(len(up_ft_cur)):
            up_ft_cur[f_id] = F.interpolate(up_ft_cur[f_id], (up_ft_cur[-1].shape[-2]*up_scale, up_ft_cur[-1].shape[-1]*up_scale))
        
        
        # for f_id in range(len(up_ft_cur)):
        #     d1,d2,d3,d4,d5 = up_ft_cur[f_id].shape
        #     up_ft_cur[f_id] = up_ft_cur[f_id].view(d1*d2,d3,d4,d5)
        #     up_ft_cur[f_id] = F.interpolate(up_ft_cur[f_id], (up_ft_cur[-1].shape[-2]*up_scale, up_ft_cur[-1].shape[-1]*up_scale))
        #     _,d3,d4,d5 = up_ft_cur[f_id].shape
        #     up_ft_cur[f_id] = up_ft_cur[f_id].view(d1,d2,d3,d4,d5)
        # editing energy
        loss_edit = 0
        mask_cur = torch.stack(mask_cur).squeeze(1) # [8, 1, 128, 128]
        mask_tar = torch.stack(mask_tar).squeeze(1) # [8, 1, 128, 128]
        
        for f_id in range(len(up_ft_tar)):
            up_ft_cur_vec = up_ft_cur[f_id][mask_cur.repeat(1,up_ft_cur[f_id].shape[1],1,1)].view(up_ft_cur[f_id].shape[1], -1).permute(1,0)
            up_ft_tar_vec = up_ft_tar[f_id][mask_tar.repeat(1,up_ft_tar[f_id].shape[1],1,1)].view(up_ft_tar[f_id].shape[1], -1).permute(1,0)
            sim = cos(up_ft_cur_vec, up_ft_tar_vec)
            sim_global = cos(up_ft_cur_vec.mean(0, keepdim=True), up_ft_tar_vec.mean(0, keepdim=True))
            loss_edit = loss_edit + (w_edit/(1+4*sim.mean()))*loss_scale[f_id] 

        # for f_id in range(len(up_ft_tar)):
        #     # print("test_move")
        #     # from IPython import embed; embed()
        #     up_ft_cur_vec = up_ft_cur[f_id][mask_cur.repeat(1,1,up_ft_cur[f_id].shape[1],1,1).transpose(1,2)].view(up_ft_cur[f_id].shape[1], -1).permute(1,0)
        #     up_ft_tar_vec = up_ft_tar[f_id][mask_tar.repeat(1,1,up_ft_tar[f_id].shape[1],1,1).transpose(1,2)].view(up_ft_tar[f_id].shape[1], -1).permute(1,0)
        #     sim = cos(up_ft_cur_vec, up_ft_tar_vec)
        #     sim_global = cos(up_ft_cur_vec.mean(0, keepdim=True), up_ft_tar_vec.mean(0, keepdim=True))
        #     loss_edit = loss_edit + (w_edit/(1+4*sim.mean()))*loss_scale[f_id] 

        # content energy
        print("test_move")
        from IPython import embed; embed()
        loss_con = 0
        if mask_x0_ref is not None and mask_x0_ref[0] is not None:
            mask_x0_ref_cur = F.interpolate(mask_x0_ref[None,None], (mask_other.shape[-2], mask_other.shape[-1]))>0.5
        else:
            mask_x0_ref_cur = mask_other
        
        mask_other = torch.stack(mask_other) # 8, 1, 1, 128, 128
        
        for f_id in range(len(up_ft_tar_org)):
            # sim_other = cos(up_ft_tar_org[f_id], up_ft_cur[f_id])[0][mask_other[0,0]]
            sim_other = cos(up_ft_tar_org[f_id], up_ft_cur[f_id])[mask_other[:,0,0]]
            loss_con = loss_con+w_content/(1+4*sim_other.mean())*loss_scale[f_id]
        
        
        mask_non_overlap = rearrange(torch.stack(mask_non_overlap),"b c f h w -> (b f) c h w")
        
        #opt部分
        for f_id in range(len(up_ft_tar)):
            up_ft_cur_non_overlap = up_ft_cur[f_id][mask_non_overlap.repeat(1,up_ft_cur[f_id].shape[1],1,1)].view(up_ft_cur[f_id].shape[1], -1).permute(1,0)
            up_ft_tar_non_overlap = up_ft_tar_org[f_id][mask_non_overlap.repeat(1,up_ft_tar_org[f_id].shape[1],1,1)].view(up_ft_tar_org[f_id].shape[1], -1).permute(1,0)
            sim_non_overlap = (cos(up_ft_cur_non_overlap, up_ft_tar_non_overlap)+1.)/2.
            loss_con = loss_con + w_contrast*sim_non_overlap.mean()*loss_scale[f_id]

            up_ft_cur_non_overlap = up_ft_cur[f_id][mask_non_overlap.repeat(1,up_ft_cur[f_id].shape[1],1,1)].view(up_ft_cur[f_id].shape[1],-1).permute(1,0).mean(0, keepdim=True)
            up_ft_tar_non_overlap = up_ft_tar_replace[f_id][mask_non_overlap.repeat(1,up_ft_tar_org[f_id].shape[1],1,1)].view(up_ft_tar_org[f_id].shape[1],-1).permute(1,0).mean(0, keepdim=True)
            sim_inpaint = ((cos(up_ft_cur_non_overlap, up_ft_tar_non_overlap)+1.)/2.)
            loss_con = loss_con + w_inpaint/(1+4*sim_inpaint.mean())

        # for f_id in range(len(up_ft_tar)):
        #     up_ft_cur_non_overlap = up_ft_cur[f_id][mask_non_overlap.repeat(1,1,up_ft_cur[f_id].shape[1],1,1).transpose(1,2)].view(up_ft_cur[f_id].shape[1], -1).permute(1,0)
        #     up_ft_tar_non_overlap = up_ft_tar_org[f_id][mask_non_overlap.repeat(1,1,up_ft_tar_org[f_id].shape[1],1,1).transpose(1,2)].view(up_ft_tar_org[f_id].shape[1], -1).permute(1,0)
        #     sim_non_overlap = (cos(up_ft_cur_non_overlap, up_ft_tar_non_overlap)+1.)/2.
        #     loss_con = loss_con + w_contrast*sim_non_overlap.mean()*loss_scale[f_id]

        #     up_ft_cur_non_overlap = up_ft_cur[f_id][mask_non_overlap.repeat(1,1,up_ft_cur[f_id].shape[1],1,1).transpose(1,2)].view(up_ft_cur[f_id].shape[1],-1).permute(1,0).mean(0, keepdim=True)
        #     up_ft_tar_non_overlap = up_ft_tar_replace[f_id][mask_non_overlap.repeat(1,1,up_ft_tar_org[f_id].shape[1],1,1).transpose(1,2)].view(up_ft_tar_org[f_id].shape[1],-1).permute(1,0).mean(0, keepdim=True)
        #     sim_inpaint = ((cos(up_ft_cur_non_overlap, up_ft_tar_non_overlap)+1.)/2.)
        #     loss_con = loss_con + w_inpaint/(1+4*sim_inpaint.mean())
        # print("in guidance")
        # from IPython import embed;embed()
        cond_grad_edit = torch.autograd.grad(loss_edit*energy_scale, latent, retain_graph=True)[0]
        cond_grad_con = torch.autograd.grad(loss_con*energy_scale, latent)[0]
        # mask_edit2 = (F.interpolate(mask_x0[None,None], (mask_cur.shape[-2], mask_cur.shape[-1]))>0.5).float()
        # print("check estimator_")
        # from IPython import embed;embed()
        mask_x0 = torch.stack(mask_x0) # 8, 128, 128
        mask_edit2 = (F.interpolate(mask_x0[None], (mask_cur.shape[-2], mask_cur.shape[-1]))>0.5).float() # torch.Size([1, 8, 128, 128])
        mask_edit1 = (mask_cur>0.5).float().transpose(0,1) # # torch.Size([1, 8, 128, 128])
        # from IPython import embed;embed()
        mask_cur = mask_cur.transpose(0,1)
        mask = ((mask_cur+mask_edit2)>0.5).float()
        mask_edit1 = (F.interpolate(mask_edit1, (latent.shape[-2], latent.shape[-1]))>0).to(dtype=latent.dtype)
        # print("in guidance")
        # from IPython import embed;embed()
        guidance = cond_grad_edit.detach()*8e-2*mask_edit1 + cond_grad_con.detach()*8e-2*(1-mask_edit1)
        self.estimator.zero_grad()

        return guidance
    
    
    
def compute_normalized_correlation(tensor_list):
    # 初始化用于存储相关性矩阵的列表
    correlation_matrices = []
    
    # 对列表中的每个 tensor 进行归一化，同时确保支持梯度计算
    normalized_tensors = [F.normalize(tensor, p=2, dim=1) for tensor in tensor_list]
    
    # 遍历列表中的每一对tensor，计算它们之间的相关性矩阵
    for i in range(len(normalized_tensors)):
        for j in range(i + 1, len(normalized_tensors)):
            tensor_i = normalized_tensors[i]
            tensor_j = normalized_tensors[j]
            
            # 计算相关性矩阵, tensor_i 的形状为 (n_i, c)，tensor_j 的形状为 (n_j, c)
            correlation_matrix = torch.matmul(tensor_i, tensor_j.T)
            
            # 将每个相关性矩阵加入结果列表
            correlation_matrices.append(correlation_matrix)
    
    return correlation_matrices

def compute_list_similarity(list_cur, list_tar, method='cosine'):
    """
    计算两个列表中每个对应 tensor 的相似度。
    
    参数:
    list_cur (list of torch.Tensor): 第一个 tensor 列表。
    list_tar (list of torch.Tensor): 第二个 tensor 列表。
    method (str): 相似度度量方法，可以是 'cosine' 或 'mse'。
    
    返回:
    similarities (list of float): 列表中每对 tensor 的相似度值。
    """
    similarities = []
    
    # 遍历两个列表中对应的tensor，计算相似度
    for tensor_cur, tensor_tar in zip(list_cur, list_tar):
        if method == 'cosine':
            # 计算余弦相似度
            tensor_cur_flat = tensor_cur.flatten()
            tensor_tar_flat = tensor_tar.flatten()
            similarity = F.cosine_similarity(tensor_cur_flat, tensor_tar_flat, dim=0)
        elif method == 'mse':
            # 计算均方误差
            similarity = F.mse_loss(tensor_cur, tensor_tar)
        else:
            raise ValueError("Unknown method. Use 'cosine' or 'mse'.")
        
        # print("test_compute")
        # from IPython import embed;embed()
        
        similarities.append(similarity)
    
    return similarities

