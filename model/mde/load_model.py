import os,sys
import torch
current_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(current_path)
sys.path.append(f'{current_path}/Depth-Anything')
sys.path.append(f'{current_path}/Midas')
sys.path.append(f'{current_path}/AdaBins')
sys.path.append(f'{current_path}/ZoeDepth')
# sys.path.append(current_path)

integrated_mde_models = [
    'dehin',
    'mono2',
    'mande', 
    'deany' 
    'midas', 
    'adabins',
    'glpn',
    'dpt'
]

def predict_batch(batch,MDE):
    adv_scene_image_eot, ben_scene_image_eot, scene_img_eot, patch_full_mask, object_full_mask = batch
    model_name,_ = MDE
    if model_name in ['dehin','mono2','mande', 'glpn', 'dpt']:
        depth_predicted,_ = predict_depth_fn( MDE, torch.cat([adv_scene_image_eot, ben_scene_image_eot, scene_img_eot], dim = 0))
        adv_depth, ben_depth, scene_depth = depth_predicted[0],depth_predicted[1],depth_predicted[2]
    else:
        adv_depth,_ = predict_depth_fn( MDE, adv_scene_image_eot)
        with torch.no_grad():
            ben_depth,_ = predict_depth_fn( MDE, ben_scene_image_eot)
            scene_depth,_ = predict_depth_fn( MDE, scene_img_eot)
        if model_name in ['adabins','deany']:
            adv_depth=adv_depth[0]
            ben_depth=ben_depth[0]
            scene_depth=scene_depth[0]
    tar_depth = scene_depth.clone().detach()
    batch_y=[adv_depth, ben_depth, scene_depth, tar_depth]
    return batch_y

def predict_depth_fn(MDE,scene,detach=False, outsize=(320,1024)):
    model_name,model = MDE
    if model_name in ['dehin','mono2','mande']:
        def disp_to_depth(disp,min_depth,max_depth):
            min_disp=1/max_depth
            max_disp=1/min_depth
            scaled_disp=min_disp+(max_disp-min_disp)*disp
            depth=1/scaled_disp
            return scaled_disp,depth
        depth_without_norm = model(scene)
        scaler=5.4
        depth=torch.clamp(disp_to_depth(torch.abs(depth_without_norm),0.1,80)[1]*scaler,max=80)

    elif model_name == 'midas':
        depth_without_norm = model(scene)
        depth_min = depth_without_norm.min()
        depth_without_norm = depth_without_norm - depth_min
        depth_max = depth_without_norm.max()
        depth_without_norm = -1 * depth_without_norm
        depth_without_norm = depth_without_norm + depth_max
        depth = depth_without_norm / depth_max * 80

       
    
    elif model_name == 'adabins':
        bin_edges, predicted_depth = model(scene)
        # upsample to the same size
        depth_without_norm = torch.nn.functional.interpolate(
            predicted_depth,
            size=outsize,
            mode="bicubic",
            align_corners=False,
        )
        depth=depth_without_norm
        # clip to valid values
        MIN_DEPTH = 1e-3
        MAX_DEPTH = 80
        depth[depth < MIN_DEPTH] = MIN_DEPTH
        depth[depth > MAX_DEPTH] = MAX_DEPTH
    
    elif model_name == 'glpn':
        depth_without_norm = model(pixel_values=scene).predicted_depth
        depth = torch.nn.functional.interpolate(
            depth_without_norm.unsqueeze(1),
            size=outsize,
            mode="bicubic",
            align_corners=False,
        )
        # print(scene.size(), depth_without_norm.size(), depth.size())
    
    elif model_name == 'dpt':
        depth_without_norm = model(pixel_values=scene).predicted_depth
        depth = torch.nn.functional.interpolate(
            depth_without_norm.unsqueeze(1),
            size=outsize,
            mode="bicubic",
            align_corners=False,
        )
    
    elif model_name == 'depthanything':
        scene= torch.nn.functional.interpolate(
            scene,
            size=(518, 518),
            mode="bicubic",
            align_corners=False,
        )
        depth_without_norm = model(scene)
        # print(depth_without_norm.shape)
        depth_without_norm = torch.nn.functional.interpolate(
            depth_without_norm.unsqueeze(1),
            size=outsize,
            mode="bicubic",
            align_corners=False,
        )
        depth_min = depth_without_norm.min()
        depth_without_norm = depth_without_norm - depth_min
        depth_max = depth_without_norm.max()
        depth_without_norm = -1 * depth_without_norm
        depth_without_norm = depth_without_norm + depth_max
        depth = depth_without_norm / depth_max * 80
    else:
        raise ValueError('invalid model name')
    
    if detach:
        depth_without_norm=depth_without_norm.detach()
        depth=depth.detach() 
    
    return depth, depth_without_norm

def load_target_mde_model(model_name, cuda = 0, encoder_idx = 0, weight_file=None, weight_file_enc=None, weigh_file_dec=None, weight_dir = 'weight'):
    device = 'cpu' if cuda is None else f'cuda:{cuda}'
    if model_name in ['mono2','mande','dehin']:
        from cnn_model import import_depth_model
        depth_model = import_depth_model(model_name=model_name,depth_model_dir=weight_dir)
    elif model_name == 'deany':
        from DepthAnything.depth_anything.dpt import DepthAnything
        encoder = ['vitl', 'vitb'] # can also be 'vitb' or 'vitl'
        encoder = encoder[encoder_idx]
        depth_model = DepthAnything.from_pretrained('LiheYoung/depth_anything_{:}14'.format(encoder))
    elif model_name == 'midas':
        from MiDaS.midas.model_loader import load_model as load_midas_model
        depth_model_dir = f'{weight_dir}/{weight_file}' if weight_file is not None else None
        depth_model, _, _, _ = load_midas_model(
            torch.device(device),
            depth_model_dir,
            'dpt_beit_large_512',#f'{weight_file[:-2]}',
            False, 320, False
        )
    elif model_name == 'adabins':
        from AdaBins.models import UnetAdaptiveBins
        from AdaBins.model_io import load_checkpoint as load_adabins_checkpoint
        MIN_DEPTH = 1e-3
        MAX_DEPTH_KITTI = 80
        N_BINS = 256
        model = UnetAdaptiveBins.build(n_bins=N_BINS, min_val=MIN_DEPTH, max_val=MAX_DEPTH_KITTI)
        pretrained_path = os.path.join(weight_dir,weight_file)
        depth_model, _, _ = load_adabins_checkpoint(pretrained_path, model)
    elif model_name == 'glpn':
        from transformers import GLPNImageProcessor, GLPNForDepthEstimation
        pretrained_path = os.path.join(weight_dir,weight_file)
        model = GLPNForDepthEstimation.from_pretrained(pretrained_path)
    elif model_name == 'dpt':
        from transformers import DPTForDepthEstimation
        pretrained_path = os.path.join(weight_dir,weight_file)
        model = DPTForDepthEstimation.from_pretrained(pretrained_path)
    else:
        raise ValueError('invalid model name')
        
    if device != 'cpu':
        depth_model = depth_model.cuda(device).eval()

    return depth_model