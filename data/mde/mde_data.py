
import sys,os
import os
import PIL.Image as Image
from PIL import ImageOps
from torchvision import transforms as T
import pickle, torch
import random
from torchvision.transforms import functional as Func
import matplotlib.pyplot as plt

import cv2
import numpy as np
from numpy.lib.stride_tricks import as_strided
import scipy.sparse
import scipy.sparse.linalg


def _rolling_block(A, block=(3, 3)):
    """Applies sliding window to given matrix."""
    shape = (A.shape[0] - block[0] + 1, A.shape[1] - block[1] + 1) + block
    strides = (A.strides[0], A.strides[1]) + A.strides
    return as_strided(A, shape=shape, strides=strides)


def compute_laplacian(img, mask=None, eps=10**(-7), win_rad=1):
    """Computes Matting Laplacian for a given image.

    Args:
        img: 3-dim numpy matrix with input image
        mask: mask of pixels for which Laplacian will be computed.
            If not set Laplacian will be computed for all pixels.
        eps: regularization parameter controlling alpha smoothness
            from Eq. 12 of the original paper. Defaults to 1e-7.
        win_rad: radius of window used to build Matting Laplacian (i.e.
            radius of omega_k in Eq. 12).
    Returns: sparse matrix holding Matting Laplacian.
    """

    win_size = (win_rad * 2 + 1) ** 2
    h, w, d = img.shape
    # Number of window centre indices in h, w axes
    c_h, c_w = h - 2 * win_rad, w - 2 * win_rad
    win_diam = win_rad * 2 + 1

    indsM = np.arange(h * w).reshape((h, w))
    ravelImg = img.reshape(h * w, d)
    win_inds = _rolling_block(indsM, block=(win_diam, win_diam))

    win_inds = win_inds.reshape(c_h, c_w, win_size)
    if mask is not None:
        mask = cv2.dilate(
            mask.astype(np.uint8),
            np.ones((win_diam, win_diam), np.uint8)
        ).astype(np.bool)
        win_mask = np.sum(mask.ravel()[win_inds], axis=2)
        win_inds = win_inds[win_mask > 0, :]
    else:
        win_inds = win_inds.reshape(-1, win_size)

    
    winI = ravelImg[win_inds]

    win_mu = np.mean(winI, axis=1, keepdims=True)
    win_var = np.einsum('...ji,...jk ->...ik', winI, winI) / win_size - np.einsum('...ji,...jk ->...ik', win_mu, win_mu)

    inv = np.linalg.inv(win_var + (eps/win_size)*np.eye(3))

    X = np.einsum('...ij,...jk->...ik', winI - win_mu, inv)
    vals = np.eye(win_size) - (1.0/win_size)*(1 + np.einsum('...ij,...kj->...ik', X, winI - win_mu))

    nz_indsCol = np.tile(win_inds, win_size).ravel()
    nz_indsRow = np.repeat(win_inds, win_size).ravel()
    nz_indsVal = vals.ravel()
    L = scipy.sparse.coo_matrix((nz_indsVal, (nz_indsRow, nz_indsCol)), shape=(h*w, h*w))
    return L


def closed_form_matting_with_prior(image, prior, prior_confidence, consts_map=None):
    """Applies closed form matting with prior alpha map to image.

    Args:
        image: 3-dim numpy matrix with input image.
        prior: matrix of same width and height as input image holding apriori alpha map.
        prior_confidence: matrix of the same shape as prior hodling confidence of prior alpha.
        consts_map: binary mask of pixels that aren't expected to change due to high
            prior confidence.

    Returns: 2-dim matrix holding computed alpha map.
    """

    assert image.shape[:2] == prior.shape, ('prior must be 2D matrix with height and width equal '
                                            'to image.')
    assert image.shape[:2] == prior_confidence.shape, ('prior_confidence must be 2D matrix with '
                                                       'height and width equal to image.')
    assert (consts_map is not None) or image.shape[:2] == consts_map.shape, (
        'consts_map must be 2D matrix with height and width equal to image.')

    logging.info('Computing Matting Laplacian.')
    laplacian = compute_laplacian(image, ~consts_map if consts_map is not None else None)

    confidence = scipy.sparse.diags(prior_confidence.ravel())
    logging.info('Solving for alpha.')
    solution = scipy.sparse.linalg.spsolve(
        laplacian + confidence,
        prior.ravel() * prior_confidence.ravel()
    )
    alpha = np.minimum(np.maximum(solution.reshape(prior.shape), 0), 1)
    return alpha


def closed_form_matting_with_trimap(image, trimap, trimap_confidence=100.0):
    """Apply Closed-Form matting to given image using trimap."""

    assert image.shape[:2] == trimap.shape, ('trimap must be 2D matrix with height and width equal '
                                             'to image.')
    consts_map = (trimap < 0.1) | (trimap > 0.9)
    return closed_form_matting_with_prior(image, trimap, trimap_confidence * consts_map, consts_map)


def closed_form_matting_with_scribbles(image, scribbles, scribbles_confidence=100.0):
    """Apply Closed-Form matting to given image using scribbles image."""

    assert image.shape == scribbles.shape, 'scribbles must have exactly same shape as image.'
    consts_map = np.sum(abs(image - scribbles), axis=-1) > 0.001
    return closed_form_matting_with_prior(
        image,
        scribbles[:, :, 0],
        scribbles_confidence * consts_map,
        consts_map
    )


closed_form_matting = closed_form_matting_with_trimap

def compute_lap(path_img):
    '''
    input: image path
    output: laplacian matrix of the input image, format is sparse matrix of pytorch in gpu
    '''
    image = cv2.imread(path_img, -1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = 1.0 * image / 255.0
    h, w, _ = image.shape
    const_size = np.zeros(shape=(h, w))
    M = compute_laplacian(image)    
    M = M.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((M.row, M.col))).long().cuda()
    values = torch.from_numpy(M.data).cuda()
    shape = torch.Size(M.shape)
    Ms = torch.sparse_coo_tensor(indices, values, shape, device=torch.device('cuda'))#稀疏张量Ms，用于后续计算（效率更高，仅存储非0元素的索引和值）
    return Ms

class ENV:
    def __init__(self, args, scene_file, scene_dir, idx, points, device=None): 
        self.args = args
        self.scene_file=scene_file
        self.scene_dir=scene_dir
        self.env_idx=idx
        self.keypoints = points
        self.device = device if device is not None else self.args['device']
        self.load_scene_warp(self.scene_file, self.scene_dir)

    def load_scene_from_img(self, scene_file, scene_dir):
        scene_image=Image.open(f'{scene_dir}/{scene_file}').convert('RGB')
        trans=T.Compose([T.Resize([self.args['sce_height'],self.args['sce_width']]),T.ToTensor()])
        scene_image=trans(scene_image).unsqueeze(0)
        return scene_image.cuda(self.device)
 
    def load_scene_warp(self,scene_file, scene_dir):
        self.env = self.load_scene_from_img(scene_file, scene_dir)
        self.insert_range, self.road_param = self.road_process(self.keypoints, self.args['up'], self.args['bottom'])
    
    def compute_road_width(self, heigh,road_top,road_top_width,road_bottom_width,road_height):
        ratio=(heigh-road_top)/road_height
        width=road_top_width+ratio*(road_bottom_width-road_top_width)
        width=round(width)
        return width
    
    def compute_width_coor(self,top_mid,bottom_mid,height):
        return round((height-top_mid[0])/(bottom_mid[0]-top_mid[0])*(bottom_mid[1]-top_mid[1])+top_mid[1])
    
    def road_process(self, road, u, b, t_shift=0, b_shift=0):
        def road_process_key(road_key_points):
            road_bottom=road_key_points[2][0]
            road_top=road_key_points[0][0]
            road_height=road_bottom-road_top
            road_top_width=road_key_points[1][1]-road_key_points[0][1]
            road_bottom_width=road_key_points[3][1]-road_key_points[2][1]
            return road_top,road_top_width,road_bottom_width,road_height
        
        road_key_points=[[road[1],road[2]-t_shift],[road[3],road[4]-t_shift],[road[5],road[6]-b_shift],[road[7],road[8]-b_shift]]
        road_top,road_top_width,road_bottom_width,road_height=road_process_key(road_key_points)
        top_mid=[road_key_points[0][0],round((road_key_points[0][1]+road_key_points[1][1])/2)]
        bottom_mid=[road_key_points[2][0],round((road_key_points[2][1]+road_key_points[3][1])/2)]
        road_param=[road_top,road_top_width,road_bottom_width,road_height,top_mid,bottom_mid]
        # print(road_top)
        insert_range=[u,b]#[u if road_top<u else road_top, b]
        return insert_range, road_param
    
    def accept_init(self, object_imgs, category=None, eval_flag=False,  offset_object=False):
        object_insert_param={}
        if category is None:
            categories = list(object_imgs.keys())
            # print('test!!!',categories)
            idx = random.randint(0,len(categories)-1)
            category = categories[idx]

        # for category in object_imgs.keys():
        if category == 'pas' and not eval_flag:
            random.shuffle(object_imgs['pas'])
        object_insert_param[category]=[]
        for _ in range(len(object_imgs[category])):
            if offset_object:
                if category == 'pas':
                    h , v, r = random.uniform(-1, 1), random.uniform(0., 0.13), 0.23
                else:
                    h , v, r = random.uniform(-0.05, 0.05), random.uniform(0., 0.15), 0.7
            else:
                if category == 'pas':
                    h , v, r = random.uniform(-1, 1), 0, 0.23
                else:
                    h , v, r = 0, 0, 0.65
            object_insert_param[category].append([h,v,r])
        if category == 'pas':
            object_insert_param['pas'].sort( key=lambda x:(x[1],x[0],x[2]),reverse=True)
            object_num = 3
        else:
            object_num = 1  
        object_idx = random.randint(0,len(object_imgs[category])-object_num)
        return object_insert_param, object_idx, object_num, category
    
    def accept_patch_and_objects(self, eval_flag, optmized_patch, patch_mask, object_imgs, insert_range, insert_height=None, init_patch=None, offset_patch=False, color_patch=False, offset_object=False, color_object=False , object_idx_g=None, category = None):
        
        batch = []
        if insert_height is None:
            insert_height = random.randint(*insert_range)
        
        adv_scene_image, ben_scene_image, patch_size, patch_full_mask = self.accept_patch(optmized_patch, init_patch, patch_mask, insert_height, offset_patch=offset_patch, color_patch=color_patch)
        
        object_init_scene, object_full_mask = self.accept_objects(eval_flag, object_imgs ,insert_height, color_object=color_object, offset_object= offset_object, object_idx_g=object_idx_g, category=category)

        adv_scene_image = adv_scene_image * (1-object_full_mask) + object_init_scene * object_full_mask
        ben_scene_image = ben_scene_image * (1-object_full_mask) + object_init_scene * object_full_mask

        
        batch= [adv_scene_image, ben_scene_image, self.env, patch_full_mask, object_full_mask]
        
        return batch, patch_size
        

    def accept_patch(self, optmized_patch, init_patch, patch_mask, insert_height, offset_patch=False, color_patch=False):

        if offset_patch:
            # h_shift = random.randint(-3,3) 
            # v_shift = random.randint(-3,3)
            # angle = random.randint(-1,1) 
            # optmized_patch = Func.rotate(optmized_patch, angle)
            # patch_mask = Func.rotate(patch_mask, angle)
            h_shift,v_shift=0,0
        else:
            h_shift,v_shift=0,0

        if color_patch:
            trans_seq_color = T.ColorJitter(brightness=0.3,contrast=0.2,saturation=0.2)
            optmized_patch = trans_seq_color(optmized_patch)
            
        
        insert_width=self.compute_width_coor(self.road_param[-2],self.road_param[-1],insert_height)
        insert_coor=[insert_height,insert_width]
        patch_insert_top=insert_coor[0]
        patch_insert_bottom=insert_coor[0]+self.args['patch_height']
        patch_width_top=self.compute_road_width(patch_insert_top,*self.road_param[:-2])
        patch_width_bottom=self.compute_road_width(patch_insert_bottom,*self.road_param[:-2])
        patch_insert_top_width=self.compute_width_coor(self.road_param[-2],self.road_param[-1],patch_insert_top)+h_shift
        patch_insert_bottom_width=self.compute_width_coor(self.road_param[-2],self.road_param[-1],patch_insert_bottom)+h_shift
        patch_size=[round(patch_width_bottom*self.args['h_w_ratio']),patch_width_bottom]
        
        patch= torch.nn.functional.interpolate(optmized_patch,size=(patch_size))
        if init_patch is not None:
            patch_ori= torch.nn.functional.interpolate(init_patch,size=(patch_size))
        mask= torch.nn.functional.interpolate(patch_mask,size=(patch_size))

        #透视变换
        startPoints=[
            [0,             0],
            [0,             patch.shape[2]],
            [patch.shape[3],patch.shape[2]],
            [patch.shape[3],0]]
        global_coor=[[patch_insert_top_width-round(patch_width_top/2),                         patch_insert_top],
                    [patch_insert_bottom_width-round(patch_width_bottom/2),                         patch_insert_bottom],
                    [patch_insert_bottom_width-round(patch_width_bottom/2)+patch_width_bottom,patch_insert_bottom],
                    [patch_insert_top_width-round(patch_width_top/2)+patch_width_top          ,patch_insert_top]]
        if (global_coor[1][1]-global_coor[0][1])>patch.shape[2]:
            print('patch_height is too large, please reduce patch_height or increase h_w_ratio, ensuring patch\'s height is larger than that of the transformed one')
            raise ValueError
        s_h= global_coor[1][1] - startPoints[1][1]
        s_w= global_coor[1][0] - startPoints[1][0]
        endPoints=[[global_coor[0][0]-s_w,global_coor[0][1]-s_h],
                [global_coor[1][0]-s_w,global_coor[1][1]-s_h],
                [global_coor[2][0]-s_w,global_coor[2][1]-s_h],
                [global_coor[3][0]-s_w,global_coor[3][1]-s_h]]
        
        patch_project=T.functional.perspective(patch,fill=0,startpoints=startPoints,endpoints=endPoints)
        if init_patch is not None:
            init_patch_project=T.functional.perspective(patch_ori,fill=0,startpoints=startPoints,endpoints=endPoints)
        mask_project=T.functional.perspective(mask,fill=0,startpoints=startPoints,endpoints=endPoints)

       
        adv_scene_image=self.env.clone().detach()
        adv_scene_image[0,:,global_coor[0][1]+v_shift:global_coor[1][1]+v_shift,global_coor[1][0]:global_coor[2][0]]=\
            adv_scene_image[0,:,global_coor[0][1]+v_shift:global_coor[1][1]+v_shift,global_coor[1][0]:global_coor[2][0]]*\
                (1-mask_project[0,:,endPoints[0][1]:endPoints[1][1],endPoints[1][0]:endPoints[2][0]])+\
                    patch_project[0,:,endPoints[0][1]:endPoints[1][1],endPoints[1][0]:endPoints[2][0]]*\
                        mask_project[0,:,endPoints[0][1]:endPoints[1][1],endPoints[1][0]:endPoints[2][0]]
    
        ben_scene_image = self.env.clone().detach()
        if init_patch is not None:
            ben_scene_image[0,:,global_coor[0][1]+v_shift:global_coor[1][1]+v_shift,global_coor[1][0]:global_coor[2][0]]=ben_scene_image[0,:,global_coor[0][1]+v_shift:global_coor[1][1]+v_shift,global_coor[1][0]:global_coor[2][0]]*(1-mask_project[0,:,endPoints[0][1]:endPoints[1][1],endPoints[1][0]:endPoints[2][0]])+init_patch_project[0,:,endPoints[0][1]:endPoints[1][1],endPoints[1][0]:endPoints[2][0]]*mask_project[0,:,endPoints[0][1]:endPoints[1][1],endPoints[1][0]:endPoints[2][0]]

        patch_full_mask=torch.zeros([1,1,ben_scene_image.shape[2],ben_scene_image.shape[3]]).cuda(self.device)
        patch_full_mask[0,:,global_coor[0][1]+v_shift:global_coor[1][1]+v_shift,global_coor[1][0]:global_coor[2][0]]=mask_project[0,:,endPoints[0][1]:endPoints[1][1],endPoints[1][0]:endPoints[2][0]]

        return adv_scene_image, ben_scene_image, patch_size, patch_full_mask
    
    def accept_object_core(self, scene, object_image, object_mask, insert_height, ratio, h_shift, offset_object=False, color_object=False):
        road_width=self.compute_road_width(insert_height,*self.road_param[:-2])
        _,_,H,W=object_image.shape
        object_width=round(road_width*ratio)
        object_height=round(H/W*object_width)

        if insert_height < object_height:
            
            raise ValueError
        object_size=[object_height,object_width]
        
        if offset_object:
            angle = random.uniform(-2,2)
            object_mask = Func.rotate(object_mask, angle)
            object_image = Func.rotate(object_image, angle)

        if color_object:
            trans_seq_color = T.ColorJitter(brightness=0.3,contrast=0.2,saturation=0.2)
            object_image=trans_seq_color(object_image)

        object_image=torch.nn.functional.interpolate(object_image,size=(object_size))
        object_mask=torch.nn.functional.interpolate(object_mask,size=(object_size))

        insert_width=self.compute_width_coor(self.road_param[-2],self.road_param[-1],insert_height)
        object_insert_coor=[insert_height,insert_width+h_shift]

        scene_image_=scene
        half=round(object_width/2)
        full_mask=torch.zeros((1,1,self.args['sce_height'],self.args['sce_width'])).cuda(self.device)
        full_mask[:,:,object_insert_coor[0]-object_height:object_insert_coor[0],object_insert_coor[1]-half:object_insert_coor[1]-half+object_width]=object_mask
        full_object=torch.zeros((1,3,self.args['sce_height'],self.args['sce_width'])).cuda(self.device)
        full_object[:,:,object_insert_coor[0]-object_height:object_insert_coor[0],object_insert_coor[1]-half:object_insert_coor[1]-half+object_width]=object_image
        scene_image_=scene_image_*(1-full_mask)+full_object*full_mask
        return scene_image_, full_mask, object_size, object_insert_coor

    def accept_objects(self, eval_flag, objects, insert_height, offset_object=False, color_object=False, object_idx_g = None, category = None):
        def init(object_imgs, category=None, eval_flag=False,  offset_object=False):
            object_insert_param={}
            if category is None:
                categories = list(object_imgs.keys())
                idx = random.randint(0,len(categories)-1)
                category = categories[idx]

            # for category in object_imgs.keys():
            if category == 'pas' and not eval_flag:
                random.shuffle(object_imgs['pas'])
            object_insert_param[category]=[]
            for _ in range(len(object_imgs[category])):
                if offset_object:
                    if category == 'pas':
                        h , v, r = random.uniform(-1, 1), random.uniform(0., 0.13), 0.23
                    else:
                        h , v, r = random.uniform(-0.05, 0.05), random.uniform(0., 0.15), 0.7
                else:
                    if category == 'pas':
                        h , v, r = random.uniform(-1, 1), 0, 0.23
                    else:
                        h , v, r = 0, 0, 0.65
                object_insert_param[category].append([h,v,r])
            if category == 'pas':
                object_insert_param['pas'].sort( key=lambda x:(x[1],x[0],x[2]),reverse=True)
                object_num = 3
            else:
                object_num = 1  
            object_idx = random.randint(0,len(object_imgs[category])-object_num)
            return object_insert_param, object_idx, object_num, category
        
        insert_params, object_idx_r, object_num, category = init(objects, category=category, eval_flag=eval_flag,  offset_object=offset_object)

        if object_idx_g is not None:
            object_idx=object_idx_g
        else:
            object_idx=object_idx_r

        
        object_init_scene = torch.zeros_like(self.env).cuda(self.device)
        object_full_mask = torch.zeros((1,3,self.args['sce_height'],self.args['sce_width'])).cuda(self.device)
        for idx, item in enumerate(objects[category][object_idx:object_idx+object_num]):
            object_insert_y = round(self.args['patch_height'] * insert_params[category][object_idx:object_idx+object_num][idx][1]) + insert_height
            width = self.compute_road_width(object_insert_y, *self.road_param[:-2])
            object_shift_x = round(width/2*insert_params[category][object_idx:object_idx+object_num][idx][0])
            object_img, object_mask = item

            object_init_scene, mask, _, _ = self.accept_object_core(object_init_scene, object_img, object_mask, object_insert_y, insert_params[category][object_idx:object_idx+object_num][idx][2], object_shift_x, offset_object=offset_object, color_object=color_object)
            object_full_mask += mask
        object_full_mask[object_full_mask>=0.5]=1.0
        object_full_mask[object_full_mask<0.5]=0.0
        
        return object_init_scene, object_full_mask

    # def eot(self, adv, ben, random_factors:list):
        # print('do eot:', random_factors)
        adv_scene_image=Func.adjust_brightness(adv, random_factors[0])
        adv_scene_image=Func.adjust_contrast(adv_scene_image, random_factors[1])
        adv_scene_image=Func.adjust_saturation(adv_scene_image, random_factors[2])
        ben_scene_image=Func.adjust_brightness(ben, random_factors[0])
        ben_scene_image=Func.adjust_contrast(ben_scene_image, random_factors[1])
        ben_scene_image=Func.adjust_saturation(ben_scene_image, random_factors[2])
        return adv_scene_image, ben_scene_image

class OBJ:
    def __init__(self, args, obj_dir, device = None ): 
        self.args = args
        # self.train_objs_file=None
        # self.test_objs_file=None
        # self.train_objs_img=None
        # self.test_objs_img=None
        self.obj_dir = obj_dir
        self.device = device if device is not None else self.args['device']
    
    def load_obj_warp(self,keys):
        def keys_fn(flag):
            if flag == 'all':
                keys=['pas', 'car', 'obs']
            elif flag == 'pas':
                keys=['pas']
            elif flag == 'car':
                keys=['car']
            elif flag == 'obs':
                keys=['obs']
            elif flag == 'npas':
                keys=['car','obs']
            elif flag == 'ncar':
                keys=['pas','obs']
            elif flag == 'nobs':
                keys=['pas','car']
            return keys   
        object_files = {}
        for key in keys:
            object_files[key]=[]
        object_file_list = os.listdir(self.args['obj_dir'])
        for file in object_file_list:
            if 'mask' not in file:
                for key in keys:
                    if key in file:
                        object_files[key].append(file)
                        break
        
        test_num = {'pas':-7, 'car':-5, 'obs':-5}
        keys = keys_fn(self.args['obj_type_train'])
        self.object_files_train={}
        for key in keys:
            self.object_files_train[key]=object_files[key][:test_num[key]]
        keys = keys_fn(self.args['obj_type_test'])
        self.object_files_test={}
        for key in keys:
            self.object_files_test[key]=object_files[key][test_num[key]:]


        self.object_imgs_train, self.object_imgs_test = {}, {}
        for category in self.object_files_train.keys():
            self.object_imgs_train[category]=[]
            for _, object_file in enumerate(self.object_files_train[category]):
                object_img, object_mask = self.load_object(object_file,self.obj_dir)
                self.object_imgs_train[category].append([object_img, object_mask])
        for category in self.object_files_test.keys():
            self.object_imgs_test[category]=[]
            for _, object_file in enumerate(self.object_files_test[category]):
                object_img, object_mask = self.load_object(object_file,self.obj_dir)
                self.object_imgs_test[category].append([object_img, object_mask])
        
    def load_object(self, object_file, object_dir ):
        object_mask_file=object_file[:-4]+'_mask.png'
        object_image=Image.open(f"{object_dir}/{object_file}").convert('RGB')
        object_mask=Image.open(f"{object_dir}/{object_mask_file}")
        if len(object_mask.split())>1:
            object_mask=ImageOps.grayscale(object_mask)
        trans=T.Compose([T.ToTensor(),])
        object_image=trans(object_image).unsqueeze(0)
        object_mask=trans(object_mask).unsqueeze(0)
        object_mask[object_mask>0.5]=1.
        object_mask[object_mask<=0.5]=0.
        return object_image.cuda(self.device), object_mask.cuda(self.device)
    
    def load_object_mask(self, phy_mask_file, phy_mask_dir):
        object_mask=Image.open(f"{phy_mask_dir}/{phy_mask_file}")
        if len(object_mask.split())>1:
            object_mask=ImageOps.grayscale(object_mask)
        trans=T.Compose([T.ToTensor()])
        object_mask=trans(object_mask).unsqueeze(0)
        object_mask[object_mask>0.5]=1.
        object_mask[object_mask<=0.5]=0.
        self.object_full_mask = object_mask.cuda(self.device)

class Patch:
    def __init__(self, args, patch_dir, device = None): 
        self.args = args
        # self.name = None
        self.init_patch = None
        self.optmized_patch = None
        self.mask = None
        self.default_patch_dir = patch_dir
        self.device = device if device is not None else self.args['device']
    
    def load_optimized_patch_only(self, patch_file, patch_dir):
        patch=Image.open(f'{patch_dir}/{patch_file}').convert('RGB')
        trans=T.Compose([T.ToTensor()])
        patch=trans(patch)
        patch=patch.unsqueeze(0)
        patch=patch.cuda(self.device)
        return patch


    def load_patch_from_img(self, patch_file, patch_dir, patch_size=None, rewrite=False):
        def load_content_and_style(args, content_image, style_image, mask=None, mask_sub=None, patch_size=None):
            if args['test_code']!='yes':
                print('comput lap matrix')
                laplacian_m = compute_lap(content_image)
            else:
                laplacian_m= None
            content_image=Image.open(content_image).convert('RGB')
            style_image=Image.open(style_image).convert('RGB')
            # if patch_size is not None:
            #     trans=T.Compose([T.Resize(patch_size),T.ToTensor()])
            # else:
            trans=T.Compose([T.ToTensor()])
            style_image=trans(style_image)
            content_image=trans(content_image)
           
            style_image=style_image.unsqueeze(0)
            content_image=content_image.unsqueeze(0)
            if patch_size is not None:
                style_image=torch.nn.functional.interpolate(style_image,size=(patch_size))
                content_image=torch.nn.functional.interpolate(content_image,size=(patch_size))
            if mask is None:
                style_mask=torch.torch.ones([1,style_image.shape[2],style_image.shape[3]])
                content_mask=style_mask.clone().detach()
            else:
                style_mask=Image.open(mask).convert('RGB')
                style_mask=trans(style_mask)[0].unsqueeze(0)
                style_mask[style_mask>0.5]=1.
                style_mask[style_mask<=0.5]=0.
                content_mask=style_mask.clone().detach()
            if mask_sub is not None:    
                style_mask_sub=Image.open(mask_sub).convert('RGB')
                style_mask_sub=trans(style_mask_sub).unsqueeze(0)
                style_mask_sub[style_mask_sub>0.5]=1.
                style_mask_sub[style_mask_sub<=0.5]=0.
                content_mask_sub=style_mask_sub.clone().detach()
            else:
                content_mask_sub,style_mask_sub=None,None

            return content_image,style_image,laplacian_m,content_mask,style_mask, content_mask_sub,style_mask_sub
        patch_mask_file=patch_file[:-4]+'_mask.png'
        #加载风格图像，初始化的补丁
        if rewrite or not os.path.exists(f'{patch_dir}/{patch_file[:-4]}.b'):
            self.args['test_code']='no'
            content_image,style_image,laplacian_m,content_mask,style_mask,_,_=load_content_and_style(
                self.args,
                content_image=f'{patch_dir}/{patch_file}', style_image=f'{patch_dir}/{patch_file}', 
                mask=f'{patch_dir}/{patch_mask_file}', patch_size=patch_size
                )
            with open(f'{patch_dir}/{patch_file[:-4]}.b','wb') as f:
                pickle.dump(laplacian_m,f)
        else:
            with open(f'{patch_dir}/{patch_file[:-4]}.b','rb') as f:
                laplacian_m=pickle.load(f)
            self.args['test_code']='yes'
            content_image,style_image,_,content_mask,style_mask,_,_=load_content_and_style(
                self.args,
                content_image=f'{patch_dir}/{patch_file}', style_image=f'{patch_dir}/{patch_file}', 
                mask=f'{patch_dir}/{patch_mask_file}', patch_size=patch_size
                )
        return content_image.cuda(self.device),style_image.cuda(self.device),laplacian_m,content_mask.cuda(self.device),style_mask.cuda(self.device)
    
    def load_patch_warp(self, patch_file, patch_dir, patch_size = None, rewrite=False):
        patch_img_c, patch_img_s, laplacian_m, patch_img_c_mask, _ = self.load_patch_from_img(patch_file, patch_dir, patch_size, rewrite)
        self.patch_img_c = patch_img_c
        self.patch_img_s = patch_img_s
        
        input_patch= patch_img_c.clone().detach().requires_grad_() 
        patch_mask = patch_img_c_mask.clone().detach().unsqueeze(0) 
        self.init_patch = patch_img_c
        self.optmized_patch = input_patch
        self.mask = patch_mask
        self.laplacian_m = laplacian_m
        
    def recover(self):
        self.optmized_patch = self.init_patch.clone().detach().requires_grad_() 

