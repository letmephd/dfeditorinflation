
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
        # print("test pipeline")
        # from IPython import embed; embed()

        self.scheduler.set_timesteps(num_inference_steps) 
        dict_mask = edit_kwargs['dict_mask'] if 'dict_mask' in edit_kwargs else None
        
        for i, t in enumerate(tqdm(self.scheduler.timesteps[-start_time:])):
            next_timestep = min(t - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999)
            next_timestep = max(next_timestep, 0)
            if energy_scale==0 or alg=='D':
                repeat=1
            elif 20<i<30 and i%2==0 : 
                repeat = 3
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
                    noise_pred = self.unet(latent_in, t, encoder_hidden_states=context,inversion = False, samplestep=i,prompt = [prompt])["sample"].squeeze(2)
                    # noise_pred = self.unet(latent_in, t, encoder_hidden_states=context, save_kv=False, mode=mode, iter_cur=i)["sample"].squeeze(2)
                    # save_kv啥的看下dragondiffusion的attention2d
                noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
                # torch.Size([1, 4, 8, 64, 64])
                noise_pred_org=None
                # from IPython import embed; embed()
                if energy_scale!=0 and i<30 and (alg=='D' or i%2==0 or i<10):
                    # editing guidance
                    noise_pred_org = noise_pred
                    if mode == 'move_batch_replace':
                        guidance = self.guidance_move_perframe_replace(latent=latent, latent_noise_ref=latent_noise_ref[-(i+1)], t=t, text_embeddings=text_embeddings_org, energy_scale=energy_scale, **edit_kwargs)
                        guidance = guidance+ self.guidance_temporal(latent=latent, latent_noise_ref=latent_noise_ref[-(i+1)], t=t, text_embeddings=text_embeddings_org, energy_scale=energy_scale, **edit_kwargs)
                   
                    elif mode == 'move_batch':
                        guidance = self.guidance_move_perframe(latent=latent, latent_noise_ref=latent_noise_ref[-(i+1)], t=t, text_embeddings=text_embeddings_org, energy_scale=energy_scale, **edit_kwargs)
                        guidance = guidance+ 10*self.guidance_temporal(latent=latent, latent_noise_ref=latent_noise_ref[-(i+1)], t=t, text_embeddings=text_embeddings_org, energy_scale=energy_scale, **edit_kwargs)
                    elif mode == 'move':
                        guidance = self.guidance_move(latent=latent, latent_noise_ref=latent_noise_ref[-(i+1)], t=t, text_embeddings=text_embeddings_org, energy_scale=energy_scale, **edit_kwargs)
                    elif mode == 'drag':
                        guidance = self.guidance_drag(latent=latent, latent_noise_ref=latent_noise_ref[-(i+1)], t=t, text_embeddings=text_embeddings_org, energy_scale=energy_scale, **edit_kwargs)
                    elif mode == 'landmark':
                        guidance = self.guidance_landmark(latent=latent, latent_noise_ref=latent_noise_ref[-(i+1)], t=t, text_embeddings=text_embeddings_org, energy_scale=energy_scale, **edit_kwargs)
                    elif mode == 'appearance':
                        guidance = self.guidance_appearance(latent=latent, latent_noise_ref=latent_noise_ref[-(i+1)], t=t, text_embeddings=text_embeddings_org, energy_scale=energy_scale, **edit_kwargs)
                    elif mode == 'paste':
                        guidance = self.guidance_paste(latent=latent, latent_noise_ref=latent_noise_ref[-(i+1)], t=t, text_embeddings=text_embeddings_org, energy_scale=energy_scale, **edit_kwargs)

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

                # if 10<i<20:
                #     eta, eta_rd = SDE_strength_un, SDE_strength
                # else:
                #     eta, eta_rd = 0., 0.
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
                # print("test_guidance")
                # from IPython import embed; embed()
                # Regional SDE 这一部分是diffeditor里面的内容
                # if (eta_rd > 0 or eta>0) and alg=='D+': # 这一步没有用到
                #     variance_noise = torch.randn_like(latent_prev)
                #     variance_rd = std_dev_t_rd * variance_noise
                #     variance = std_dev_t * variance_noise
                #     # from IPython import embed;embed()
                    
                #     if mode == 'move_batch_replace' :
                        
                #         mask_x0 = torch.stack(edit_kwargs["mask_x0"]) # 8, 512, 512
                #         mask_cur = torch.stack(edit_kwargs["mask_cur"]).transpose(0,2).squeeze(0) # 1,1,8,128,128
                #         mask = (F.interpolate(mask_x0[None], (mask_cur.shape[-2], mask_cur.shape[-1]))>0.5).float()
                #         mask = ((mask_cur+mask)>0.5).float()
                #         mask = (F.interpolate(mask, (latent_prev.shape[-2], latent_prev.shape[-1]))>0.5).to(dtype=latent.dtype)
                    
                #     elif mode == 'move_batch' :
                        
                #         mask_x0 = torch.stack(edit_kwargs["mask_x0"]) # 8, 512, 512
                #         mask_cur = torch.stack(edit_kwargs["mask_cur"]).transpose(0,2).squeeze(0) # 1,1,8,128,128
                #         mask = (F.interpolate(mask_x0[None], (mask_cur.shape[-2], mask_cur.shape[-1]))>0.5).float()
                #         mask = ((mask_cur+mask)>0.5).float()
                #         mask = (F.interpolate(mask, (latent_prev.shape[-2], latent_prev.shape[-1]))>0.5).to(dtype=latent.dtype)
                    
                #     if mode == 'move_':
                #         mask = (F.interpolate(edit_kwargs["mask_x0"][None,None], (edit_kwargs["mask_cur"].shape[-2], edit_kwargs["mask_cur"].shape[-1]))>0.5).float()
                #         mask = ((edit_kwargs["mask_cur"]+mask)>0.5).float()
                #         mask = (F.interpolate(mask, (latent_prev.shape[-2], latent_prev.shape[-1]))>0.5).to(dtype=latent.dtype)
                #     elif mode == 'move':
                #         mask = (F.interpolate(edit_kwargs["mask_x0"][None,None], (edit_kwargs["mask_cur"].shape[-2], edit_kwargs["mask_cur"].shape[-1]))>0.5).float()
                #         mask = ((edit_kwargs["mask_cur"]+mask)>0.5).float()
                #         mask = (F.interpolate(mask, (latent_prev.shape[-2], latent_prev.shape[-1]))>0.5).to(dtype=latent.dtype)
                #     elif mode == 'drag':
                #         mask = F.interpolate(edit_kwargs["mask_x0"][None,None], (latent_prev[-1].shape[-2], latent_prev[-1].shape[-1]))
                #         mask = (mask>0).to(dtype=latent.dtype)
                #     elif mode == 'landmark':
                #         mask = torch.ones_like(latent_prev)
                #     elif mode == 'appearance' or mode == 'paste':
                #         mask = F.interpolate(edit_kwargs["mask_base_cur"].float(), (latent_prev[-1].shape[-2], latent_prev[-1].shape[-1]))
                #         mask = (mask>0).to(dtype=latent.dtype)
                #     latent_prev = (latent_prev+variance)*(1-mask) + (latent_prev_rd+variance_rd)*mask
                # print("move_batch")
                # from IPython import embed;embed()
                if repeat>1:# 这些步骤都是在中间几个step才会用到的
                    with torch.no_grad():
                        # from IPython import embed; embed()
                        alpha_prod_t = self.scheduler.alphas_cumprod[next_timestep]
                        alpha_prod_t_next = self.scheduler.alphas_cumprod[t]
                        beta_prod_t = 1 - alpha_prod_t
                        model_output = self.unet(latent_prev, next_timestep, encoder_hidden_states=text_embeddings, samplestep=i,no_save = True)["sample"].squeeze(2)
                        # model_output = self.unet(latent_prev.unsqueeze(2), next_timestep, encoder_hidden_states=text_embeddings, mask=dict_mask, save_kv=False, mode=mode, iter_cur=-2)["sample"].squeeze(2)
                        next_original_sample = (latent_prev - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
                        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
                        latent = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
            
            latent = latent_prev
            
        return latent
    
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
        
        for f_id in range(len(up_ft_tar_org)):
            # sim_other = cos(up_ft_tar_org[f_id], up_ft_cur[f_id])[0][mask_other[0,0]]
            sim_other = cos(up_ft_tar_org[f_id], up_ft_cur[f_id])[mask_other[:,0,0]]
            loss_con = loss_con+w_content/(1+4*sim_other.mean())*loss_scale[f_id]
        
        
        mask_non_overlap = rearrange(torch.stack(mask_non_overlap),"b c f h w -> (b f) c h w")
        
        # print("test")
        # from IPython import embed;embed()
        
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
        
        guidance = cond_grad_edit.detach()*8e-2*mask_edit1 + cond_grad_con.detach()*8e-2*(1-mask_edit1)
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
            
        
        # for i in range(f-1):
        #     mask_edit_tmp1,mask_edit_tmp2 = main(mask_edit[0][i].unsqueeze(0).unsqueeze(0),mask_edit[0][i+1].unsqueeze(0).unsqueeze(0))
        #     # print("test temporal")
        #     # from IPython import embed;embed()
        #     mask_edit_tmp0,mask_edit_tmp2_ = main(mask_edit[0][0].unsqueeze(0).unsqueeze(0),mask_edit[0][i+1].unsqueeze(0).unsqueeze(0))
        #     mask_edit_tmp0 = (mask_edit_tmp0>0).repeat(1,4,1,1)
        #     mask_edit_tmp1 = (mask_edit_tmp1>0).repeat(1,4,1,1)
        #     mask_edit_tmp2 = (mask_edit_tmp2>0).repeat(1,4,1,1)
        #     mask_edit_tmp2_ = (mask_edit_tmp2_>0).repeat(1,4,1,1)
            
        #     latent1 = latent[:,:,i][mask_edit_tmp1]
        #     latent2 = latent[:,:,i+1][mask_edit_tmp2]
        #     latent0 = latent[:,:,0][mask_edit_tmp0]
        #     latent2_ = latent[:,:,0][mask_edit_tmp2_]
        #     loss_temp = loss_temp+cos(latent1, latent2)+cos(latent0, latent2_)
            # loss_temp = loss_temp+cos(latent1, latent2)
        # from IPython import embed;embed()
        
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
    
    
    
    def guidance_move_(
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
        # from IPython import embed;embed()
        w_inpaint=10
        w_edit = 10
        # from IPython import embed;embed()
        w_contrast=1 # 原来是0.2
        loss_scale = [0.5, 0.5]
        
        
        with torch.no_grad():
            # print("guidance")
            # from IPython import embed;embed()
            up_ft_tar = self.estimator( # src.unet.estimator.MyUNet2DConditionModel
                        sample=latent_noise_ref.squeeze(2)[::2],
                        timestep=t,
                        up_ft_indices=up_ft_index,
                        encoder_hidden_states=text_embeddings)['up_ft']
            up_ft_tar_org = copy.deepcopy(up_ft_tar)
            for f_id in range(len(up_ft_tar_org)):
                up_ft_tar_org[f_id] = F.interpolate(up_ft_tar_org[f_id], (up_ft_tar_org[-1].shape[-2]*up_scale, up_ft_tar_org[-1].shape[-1]*up_scale))
        with torch.no_grad():
            up_ft_tar_replace = self.estimator(
                        sample=latent_noise_ref.squeeze(2)[1::2],
                        timestep=t,
                        up_ft_indices=up_ft_index,
                        encoder_hidden_states=text_embeddings)['up_ft']
            for f_id in range(len(up_ft_tar_replace)):
                up_ft_tar_replace[f_id] = F.interpolate(up_ft_tar_replace[f_id], (up_ft_tar_replace[-1].shape[-2]*up_scale, up_ft_tar_replace[-1].shape[-1]*up_scale))
        
        # print("test_move")
        # from IPython import embed; embed()
        
        latent = latent.detach().requires_grad_(True)
        for f_id in range(len(up_ft_tar)):
            up_ft_tar[f_id] = F.interpolate(up_ft_tar[f_id], (int(up_ft_tar[-1].shape[-2]*resize_scale*up_scale), int(up_ft_tar[-1].shape[-1]*resize_scale*up_scale)))

        up_ft_cur = self.estimator(
                    sample=latent,
                    timestep=t,
                    up_ft_indices=up_ft_index,
                    encoder_hidden_states=text_embeddings)['up_ft']
        for f_id in range(len(up_ft_cur)):
            up_ft_cur[f_id] = F.interpolate(up_ft_cur[f_id], (up_ft_cur[-1].shape[-2]*up_scale, up_ft_cur[-1].shape[-1]*up_scale))
        # editing energy
        loss_edit = 0
        for f_id in range(len(up_ft_tar)):
            up_ft_cur_vec = up_ft_cur[f_id][mask_cur.repeat(1,up_ft_cur[f_id].shape[1],1,1)].view(up_ft_cur[f_id].shape[1], -1).permute(1,0)
            up_ft_tar_vec = up_ft_tar[f_id][mask_tar.repeat(1,up_ft_tar[f_id].shape[1],1,1)].view(up_ft_tar[f_id].shape[1], -1).permute(1,0)
            sim = cos(up_ft_cur_vec, up_ft_tar_vec)
            sim_global = cos(up_ft_cur_vec.mean(0, keepdim=True), up_ft_tar_vec.mean(0, keepdim=True))
            loss_edit = loss_edit + (w_edit/(1+4*sim.mean()))*loss_scale[f_id] 

        # content energy
        loss_con = 0
        if mask_x0_ref is not None:
            mask_x0_ref_cur = F.interpolate(mask_x0_ref[None,None], (mask_other.shape[-2], mask_other.shape[-1]))>0.5
        else:
            mask_x0_ref_cur = mask_other
        for f_id in range(len(up_ft_tar_org)):
            sim_other = cos(up_ft_tar_org[f_id], up_ft_cur[f_id])[0][mask_other[0,0]]
            loss_con = loss_con+w_content/(1+4*sim_other.mean())*loss_scale[f_id]
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
        mask_edit2 = (F.interpolate(mask_x0[None,None], (mask_cur.shape[-2], mask_cur.shape[-1]))>0.5).float()
        mask_edit1 = (mask_cur>0.5).float()
        # from IPython import embed;embed()
        mask = ((mask_cur+mask_edit2)>0.5).float()
        mask_edit1 = (F.interpolate(mask_edit1, (latent.shape[-2], latent.shape[-1]))>0).to(dtype=latent.dtype)
        guidance = cond_grad_edit.detach()*8e-2*mask_edit1 + cond_grad_con.detach()*8e-2*(1-mask_edit1)
        self.estimator.zero_grad()

        return guidance
    
    def guidance_move(
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
        loss_scale = [0.5, 0.5]
        with torch.no_grad():
            up_ft_tar = self.estimator( # src.unet.estimator.MyUNet2DConditionModel
                        sample=latent_noise_ref.squeeze(2),
                        timestep=t,
                        up_ft_indices=up_ft_index,
                        encoder_hidden_states=text_embeddings)['up_ft']
            up_ft_tar_org = copy.deepcopy(up_ft_tar)
            for f_id in range(len(up_ft_tar_org)):
                up_ft_tar_org[f_id] = F.interpolate(up_ft_tar_org[f_id], (up_ft_tar_org[-1].shape[-2]*up_scale, up_ft_tar_org[-1].shape[-1]*up_scale))

        # print("test_move")
        # from IPython import embed; embed()
        
        latent = latent.detach().requires_grad_(True)
        for f_id in range(len(up_ft_tar)):
            up_ft_tar[f_id] = F.interpolate(up_ft_tar[f_id], (int(up_ft_tar[-1].shape[-2]*resize_scale*up_scale), int(up_ft_tar[-1].shape[-1]*resize_scale*up_scale)))

        up_ft_cur = self.estimator(
                    sample=latent,
                    timestep=t,
                    up_ft_indices=up_ft_index,
                    encoder_hidden_states=text_embeddings)['up_ft']
        for f_id in range(len(up_ft_cur)):
            up_ft_cur[f_id] = F.interpolate(up_ft_cur[f_id], (up_ft_cur[-1].shape[-2]*up_scale, up_ft_cur[-1].shape[-1]*up_scale))
        # editing energy
        loss_edit = 0
        for f_id in range(len(up_ft_tar)):
            up_ft_cur_vec = up_ft_cur[f_id][mask_cur.repeat(1,up_ft_cur[f_id].shape[1],1,1)].view(up_ft_cur[f_id].shape[1], -1).permute(1,0)
            up_ft_tar_vec = up_ft_tar[f_id][mask_tar.repeat(1,up_ft_tar[f_id].shape[1],1,1)].view(up_ft_tar[f_id].shape[1], -1).permute(1,0)
            sim = cos(up_ft_cur_vec, up_ft_tar_vec)
            sim_global = cos(up_ft_cur_vec.mean(0, keepdim=True), up_ft_tar_vec.mean(0, keepdim=True))
            loss_edit = loss_edit + (w_edit/(1+4*sim.mean()))*loss_scale[f_id] 

        # content energy
        loss_con = 0
        if mask_x0_ref is not None:
            mask_x0_ref_cur = F.interpolate(mask_x0_ref[None,None], (mask_other.shape[-2], mask_other.shape[-1]))>0.5
        else:
            mask_x0_ref_cur = mask_other
        for f_id in range(len(up_ft_tar_org)):
            sim_other = cos(up_ft_tar_org[f_id], up_ft_cur[f_id])[0][mask_other[0,0]]
            loss_con = loss_con+w_content/(1+4*sim_other.mean())*loss_scale[f_id]

        for f_id in range(len(up_ft_tar)):
            up_ft_cur_non_overlap = up_ft_cur[f_id][mask_non_overlap.repeat(1,up_ft_cur[f_id].shape[1],1,1)].view(up_ft_cur[f_id].shape[1], -1).permute(1,0)
            up_ft_tar_non_overlap = up_ft_tar_org[f_id][mask_non_overlap.repeat(1,up_ft_tar_org[f_id].shape[1],1,1)].view(up_ft_tar_org[f_id].shape[1], -1).permute(1,0)
            sim_non_overlap = (cos(up_ft_cur_non_overlap, up_ft_tar_non_overlap)+1.)/2.
            loss_con = loss_con + w_contrast*sim_non_overlap.mean()*loss_scale[f_id]

            up_ft_cur_non_overlap = up_ft_cur[f_id][mask_non_overlap.repeat(1,up_ft_cur[f_id].shape[1],1,1)].view(up_ft_cur[f_id].shape[1],-1).permute(1,0).mean(0, keepdim=True)
            up_ft_tar_non_overlap = up_ft_tar_org[f_id][mask_x0_ref_cur.repeat(1,up_ft_tar_org[f_id].shape[1],1,1)].view(up_ft_tar_org[f_id].shape[1],-1).permute(1,0).mean(0, keepdim=True)
            sim_inpaint = ((cos(up_ft_cur_non_overlap, up_ft_tar_non_overlap)+1.)/2.)
            loss_con = loss_con + w_inpaint/(1+4*sim_inpaint.mean())

        cond_grad_edit = torch.autograd.grad(loss_edit*energy_scale, latent, retain_graph=True)[0]
        cond_grad_con = torch.autograd.grad(loss_con*energy_scale, latent)[0]
        mask_edit2 = (F.interpolate(mask_x0[None,None], (mask_cur.shape[-2], mask_cur.shape[-1]))>0.5).float()
        mask_edit1 = (mask_cur>0.5).float()
        mask = ((mask_cur+mask_edit2)>0.5).float()
        mask_edit1 = (F.interpolate(mask_edit1, (latent.shape[-2], latent.shape[-1]))>0).to(dtype=latent.dtype)
        guidance = cond_grad_edit.detach()*8e-2*mask_edit1 + cond_grad_con.detach()*8e-2*(1-mask_edit1)
        self.estimator.zero_grad()

        return guidance

    def guidance_drag(
        self, 
        mask_x0,
        mask_cur, 
        mask_tar, 
        mask_other, 
        latent, 
        latent_noise_ref, 
        t, 
        up_ft_index, 
        up_scale, 
        text_embeddings,
        energy_scale,
        w_edit,
        w_inpaint,
        w_content,
        dict_mask = None,
    ):
        cos = nn.CosineSimilarity(dim=1)
        with torch.no_grad():
            up_ft_tar = self.estimator(
                        sample=latent_noise_ref.squeeze(2),
                        timestep=t,
                        up_ft_indices=up_ft_index,
                        encoder_hidden_states=text_embeddings)['up_ft']
            for f_id in range(len(up_ft_tar)):
                up_ft_tar[f_id] = F.interpolate(up_ft_tar[f_id], (up_ft_tar[-1].shape[-2]*up_scale, up_ft_tar[-1].shape[-1]*up_scale))

        latent = latent.detach().requires_grad_(True)
        up_ft_cur = self.estimator(
                    sample=latent,
                    timestep=t,
                    up_ft_indices=up_ft_index,
                    encoder_hidden_states=text_embeddings)['up_ft']
        for f_id in range(len(up_ft_cur)):
            up_ft_cur[f_id] = F.interpolate(up_ft_cur[f_id], (up_ft_cur[-1].shape[-2]*up_scale, up_ft_cur[-1].shape[-1]*up_scale))

        # moving loss
        loss_edit = 0
        for f_id in range(len(up_ft_tar)):
            for mask_cur_i, mask_tar_i in zip(mask_cur, mask_tar):
                up_ft_cur_vec = up_ft_cur[f_id][mask_cur_i.repeat(1,up_ft_cur[f_id].shape[1],1,1)].view(up_ft_cur[f_id].shape[1], -1).permute(1,0)
                up_ft_tar_vec = up_ft_tar[f_id][mask_tar_i.repeat(1,up_ft_tar[f_id].shape[1],1,1)].view(up_ft_tar[f_id].shape[1], -1).permute(1,0)
                sim = (cos(up_ft_cur_vec, up_ft_tar_vec)+1.)/2.
                loss_edit = loss_edit + w_edit/(1+4*sim.mean())

                mask_overlap = ((mask_cur_i.float()+mask_tar_i.float())>1.5).float()
                mask_non_overlap = (mask_tar_i.float()-mask_overlap)>0.5
                up_ft_cur_non_overlap = up_ft_cur[f_id][mask_non_overlap.repeat(1,up_ft_cur[f_id].shape[1],1,1)].view(up_ft_cur[f_id].shape[1], -1).permute(1,0)
                up_ft_tar_non_overlap = up_ft_tar[f_id][mask_non_overlap.repeat(1,up_ft_tar[f_id].shape[1],1,1)].view(up_ft_tar[f_id].shape[1], -1).permute(1,0)
                sim_non_overlap = (cos(up_ft_cur_non_overlap, up_ft_tar_non_overlap)+1.)/2.
                loss_edit = loss_edit + w_inpaint*sim_non_overlap.mean()
        # consistency loss
        loss_con = 0
        for f_id in range(len(up_ft_tar)):
            sim_other = (cos(up_ft_tar[f_id], up_ft_cur[f_id])[0][mask_other[0,0]]+1.)/2.
            loss_con = loss_con+w_content/(1+4*sim_other.mean())
        loss_edit = loss_edit/len(up_ft_cur)/len(mask_cur)
        loss_con = loss_con/len(up_ft_cur)

        cond_grad_edit = torch.autograd.grad(loss_edit*energy_scale, latent, retain_graph=True)[0]
        cond_grad_con = torch.autograd.grad(loss_con*energy_scale, latent)[0]
        mask = F.interpolate(mask_x0[None,None], (cond_grad_edit[-1].shape[-2], cond_grad_edit[-1].shape[-1]))
        mask = (mask>0).to(dtype=latent.dtype)
        guidance = cond_grad_edit.detach()*4e-2*mask + cond_grad_con.detach()*4e-2*(1-mask)
        self.estimator.zero_grad()

        return guidance

    def guidance_landmark(
        self, 
        mask_cur, 
        mask_tar,
        latent, 
        latent_noise_ref, 
        t, 
        up_ft_index, 
        up_scale, 
        text_embeddings,
        energy_scale,
        w_edit,
        w_inpaint,
    ):
        cos = nn.CosineSimilarity(dim=1)
        with torch.no_grad():
            up_ft_tar = self.estimator(
                        sample=latent_noise_ref.squeeze(2),
                        timestep=t,
                        up_ft_indices=up_ft_index,
                        encoder_hidden_states=text_embeddings)['up_ft']
            for f_id in range(len(up_ft_tar)):
                up_ft_tar[f_id] = F.interpolate(up_ft_tar[f_id], (up_ft_tar[-1].shape[-2]*up_scale, up_ft_tar[-1].shape[-1]*up_scale))

        latent = latent.detach().requires_grad_(True)
        up_ft_cur = self.estimator(
                    sample=latent,
                    timestep=t,
                    up_ft_indices=up_ft_index,
                    encoder_hidden_states=text_embeddings)['up_ft']
        for f_id in range(len(up_ft_cur)):
            up_ft_cur[f_id] = F.interpolate(up_ft_cur[f_id], (up_ft_cur[-1].shape[-2]*up_scale, up_ft_cur[-1].shape[-1]*up_scale))

        # moving loss
        loss_edit = 0
        for f_id in range(len(up_ft_tar)):
            for mask_cur_i, mask_tar_i in zip(mask_cur, mask_tar):
                up_ft_cur_vec = up_ft_cur[f_id][mask_cur_i.repeat(1,up_ft_cur[f_id].shape[1],1,1)].view(up_ft_cur[f_id].shape[1], -1).permute(1,0)
                up_ft_tar_vec = up_ft_tar[f_id][mask_tar_i.repeat(1,up_ft_tar[f_id].shape[1],1,1)].view(up_ft_tar[f_id].shape[1], -1).permute(1,0)
                sim = (cos(up_ft_cur_vec, up_ft_tar_vec)+1.)/2.
                loss_edit = loss_edit + w_edit/(1+4*sim.mean())
        loss_edit = loss_edit/len(up_ft_cur)/len(mask_cur)

        cond_grad_edit = torch.autograd.grad(loss_edit*energy_scale, latent, retain_graph=True)[0]
        guidance = cond_grad_edit.detach()*4e-2
        self.estimator.zero_grad()

        return guidance

    def guidance_appearance(
        self, 
        mask_base_cur, 
        mask_replace_cur, 
        latent, 
        latent_noise_ref, 
        t, 
        up_ft_index, 
        up_scale, 
        text_embeddings,
        energy_scale,
        dict_mask,
        w_edit,
        w_content,
    ):
        cos = nn.CosineSimilarity(dim=1)
        with torch.no_grad():
            up_ft_tar_base = self.estimator(
                        sample=latent_noise_ref.squeeze(2)[::2],
                        timestep=t,
                        up_ft_indices=up_ft_index,
                        encoder_hidden_states=text_embeddings)['up_ft']
            for f_id in range(len(up_ft_tar_base)):
                up_ft_tar_base[f_id] = F.interpolate(up_ft_tar_base[f_id], (up_ft_tar_base[-1].shape[-2]*up_scale, up_ft_tar_base[-1].shape[-1]*up_scale))
        with torch.no_grad():
            up_ft_tar_replace = self.estimator(
                        sample=latent_noise_ref.squeeze(2)[1::2],
                        timestep=t,
                        up_ft_indices=up_ft_index,
                        encoder_hidden_states=text_embeddings)['up_ft']
            for f_id in range(len(up_ft_tar_replace)):
                up_ft_tar_replace[f_id] = F.interpolate(up_ft_tar_replace[f_id], (up_ft_tar_replace[-1].shape[-2]*up_scale, up_ft_tar_replace[-1].shape[-1]*up_scale))
        latent = latent.detach().requires_grad_(True)
        up_ft_cur = self.estimator(
                    sample=latent,
                    timestep=t,
                    up_ft_indices=up_ft_index,
                    encoder_hidden_states=text_embeddings)['up_ft']
        for f_id in range(len(up_ft_cur)):
            up_ft_cur[f_id] = F.interpolate(up_ft_cur[f_id], (up_ft_cur[-1].shape[-2]*up_scale, up_ft_cur[-1].shape[-1]*up_scale))
        
        # for base content
        loss_con = 0
        for f_id in range(len(up_ft_tar_base)):
            mask_cur = (1-mask_base_cur.float())>0.5
            up_ft_cur_vec = up_ft_cur[f_id][mask_cur.repeat(1,up_ft_cur[f_id].shape[1],1,1)].view(up_ft_cur[f_id].shape[1], -1).permute(1,0)
            up_ft_tar_vec = up_ft_tar_base[f_id][mask_cur.repeat(1,up_ft_tar_base[f_id].shape[1],1,1)].view(up_ft_tar_base[f_id].shape[1], -1).permute(1,0)
            sim = (cos(up_ft_cur_vec, up_ft_tar_vec)+1.)/2.
            loss_con = loss_con + w_content/(1+4*sim.mean())
        # for replace content
        loss_edit = 0
        for f_id in range(len(up_ft_tar_replace)):
            mask_cur = mask_base_cur
            mask_tar = mask_replace_cur
            up_ft_cur_vec = up_ft_cur[f_id][mask_cur.repeat(1,up_ft_cur[f_id].shape[1],1,1)].view(up_ft_cur[f_id].shape[1], -1).permute(1,0).mean(0, keepdim=True)
            up_ft_tar_vec = up_ft_tar_replace[f_id][mask_tar.repeat(1,up_ft_tar_replace[f_id].shape[1],1,1)].view(up_ft_tar_replace[f_id].shape[1], -1).permute(1,0).mean(0, keepdim=True)
            sim_all=((cos(up_ft_cur_vec, up_ft_tar_vec)+1.)/2.)
            loss_edit =  loss_edit + w_edit/(1+4*sim_all.mean())

        cond_grad_con = torch.autograd.grad(loss_con*energy_scale, latent, retain_graph=True)[0]
        cond_grad_edit = torch.autograd.grad(loss_edit*energy_scale, latent)[0]
        mask = F.interpolate(mask_base_cur.float(), (cond_grad_edit[-1].shape[-2], cond_grad_edit[-1].shape[-1]))
        mask = (mask>0).to(dtype=latent.dtype)
        guidance = cond_grad_con.detach()*(1-mask)*4e-2 + cond_grad_edit.detach()*mask*4e-2
        self.estimator.zero_grad()

        return guidance

    def guidance_paste(
        self, 
        mask_base_cur, 
        mask_replace_cur, 
        latent, 
        latent_noise_ref, 
        t, 
        up_ft_index, 
        up_scale, 
        text_embeddings,
        energy_scale,
        dict_mask,
        w_edit,
        w_content,
    ):
        cos = nn.CosineSimilarity(dim=1)
        with torch.no_grad():
            up_ft_tar_base = self.estimator(
                        sample=latent_noise_ref.squeeze(2)[::2],
                        timestep=t,
                        up_ft_indices=up_ft_index,
                        encoder_hidden_states=text_embeddings)['up_ft']
            for f_id in range(len(up_ft_tar_base)):
                up_ft_tar_base[f_id] = F.interpolate(up_ft_tar_base[f_id], (up_ft_tar_base[-1].shape[-2]*up_scale, up_ft_tar_base[-1].shape[-1]*up_scale))
        with torch.no_grad():
            up_ft_tar_replace = self.estimator(
                        sample=latent_noise_ref.squeeze(2)[1::2],
                        timestep=t,
                        up_ft_indices=up_ft_index,
                        encoder_hidden_states=text_embeddings)['up_ft']
            for f_id in range(len(up_ft_tar_replace)):
                up_ft_tar_replace[f_id] = F.interpolate(up_ft_tar_replace[f_id], (up_ft_tar_replace[-1].shape[-2]*up_scale, up_ft_tar_replace[-1].shape[-1]*up_scale))
            
        latent = latent.detach().requires_grad_(True)
        up_ft_cur = self.estimator(
                    sample=latent,
                    timestep=t,
                    up_ft_indices=up_ft_index,
                    encoder_hidden_states=text_embeddings)['up_ft']
        for f_id in range(len(up_ft_cur)):
            up_ft_cur[f_id] = F.interpolate(up_ft_cur[f_id], (up_ft_cur[-1].shape[-2]*up_scale, up_ft_cur[-1].shape[-1]*up_scale))
        # for base content
        loss_con = 0
        for f_id in range(len(up_ft_tar_base)):
            mask_cur = (1-mask_base_cur.float())>0.5
            up_ft_cur_vec = up_ft_cur[f_id][mask_cur.repeat(1,up_ft_cur[f_id].shape[1],1,1)].view(up_ft_cur[f_id].shape[1], -1).permute(1,0)
            up_ft_tar_vec = up_ft_tar_base[f_id][mask_cur.repeat(1,up_ft_tar_base[f_id].shape[1],1,1)].view(up_ft_tar_base[f_id].shape[1], -1).permute(1,0)
            sim = (cos(up_ft_cur_vec, up_ft_tar_vec)+1.)/2.
            loss_con = loss_con + w_content/(1+4*sim.mean())
        # for replace content
        loss_edit = 0
        for f_id in range(len(up_ft_tar_replace)):
            mask_cur = mask_base_cur

            up_ft_cur_vec = up_ft_cur[f_id][mask_cur.repeat(1,up_ft_cur[f_id].shape[1],1,1)].view(up_ft_cur[f_id].shape[1], -1).permute(1,0)
            up_ft_tar_vec = up_ft_tar_replace[f_id][mask_replace_cur.repeat(1,up_ft_tar_replace[f_id].shape[1],1,1)].view(up_ft_tar_replace[f_id].shape[1], -1).permute(1,0)
            sim_all=((cos(up_ft_cur_vec, up_ft_tar_vec)+1.)/2.)
            loss_edit =  loss_edit + w_edit/(1+4*sim_all.mean())

        cond_grad_con = torch.autograd.grad(loss_con*energy_scale, latent, retain_graph=True)[0]
        cond_grad_edit = torch.autograd.grad(loss_edit*energy_scale, latent)[0]
        mask = F.interpolate(mask_base_cur.float(), (cond_grad_edit[-1].shape[-2], cond_grad_edit[-1].shape[-1]))
        mask = (mask>0).to(dtype=latent.dtype)
        guidance = cond_grad_con.detach()*(1-mask)*4e-2 + cond_grad_edit.detach()*mask*4e-2
        self.estimator.zero_grad()

        return guidance