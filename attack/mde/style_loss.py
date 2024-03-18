import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.autograd import Variable
import copy
class ContentLoss(nn.Module):

    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()

    def forward(self, input):
        #print('*************: ', input.size(), self.target.size())
        if input.size() != self.target.size():
            pass
        else:
            channel, height, width = input.size()[1:4]
            self.loss = F.mse_loss(input, self.target)
        return input


def gram_matrix(input):
    a, b, c, d = input.size()  # a = batch size (=1)
    # b = number of feature maps
    # (c, d) = dimensions of a f. map (N=c*d)

    features = input.view(a*b, c*d)

    G = torch.mm(features, features.t())  # compute the gram product
    # we 'normalize' the values of the gram matrix
    # by dividing by the number of element in each feature maps.
    return G.div(a * b * c *d)


class StyleLoss(nn.Module):

    def __init__(self, target_feature, style_mask, content_mask,device='cuda:0'):
        super(StyleLoss, self).__init__()
        self.device = device
        self.style_mask = style_mask.detach()
        self.content_mask = content_mask.detach()

        #print(target_feature.type(), mask.type())
        _, channel_f, height, width = target_feature.size()
        channel = self.style_mask.size()[0]
        
        # ********
        xc = torch.linspace(-1, 1, width).repeat(height, 1)
        yc = torch.linspace(-1, 1, height).view(-1, 1).repeat(1, width)
        grid = torch.cat((xc.unsqueeze(2), yc.unsqueeze(2)), 2)  # (w,h,2)
        grid = grid.unsqueeze_(0).to(self.device) # (1,w,h,2)
        mask_ = F.grid_sample(self.style_mask.unsqueeze(0), grid).squeeze(0)
        # ********       
        target_feature_3d = target_feature.squeeze(0).clone()
        size_of_mask = (channel, channel_f, height, width)
        target_feature_masked = torch.zeros(size_of_mask, dtype=torch.float).to(self.device)
        for i in range(channel):
            target_feature_masked[i, :, :, :] = mask_[i, :, :] * target_feature_3d

        self.targets = list()
        for i in range(channel):
            # print('!!!!',channel,torch.mean(mask_[i, :, :]))
            temp = target_feature_masked[i, :, :, :]
            if torch.mean(mask_[i, :, :]) > 0.0:
                # temp = target_feature_masked[i, :, :, :]
                self.targets.append( gram_matrix(temp.unsqueeze(0)).detach()/torch.mean(mask_[i, :, :]) )
            else:
                self.targets.append( gram_matrix(temp.unsqueeze(0)).detach())
    
    def forward(self, input_feature):
        self.loss = 0
        _, channel_f, height, width = input_feature.size()
        #channel = self.content_mask.size()[0]
        channel = len(self.targets)
        # ****
        xc = torch.linspace(-1, 1, width).repeat(height, 1)
        yc = torch.linspace(-1, 1, height).view(-1, 1).repeat(1, width)
        grid = torch.cat((xc.unsqueeze(2), yc.unsqueeze(2)), 2)
        grid = grid.unsqueeze_(0).to(self.device)
        mask = F.grid_sample(self.content_mask.unsqueeze(0), grid).squeeze(0)
        # ****
        #mask = self.content_mask.data.resize_(channel, height, width).clone()
        input_feature_3d = input_feature.squeeze(0).clone() #TODO why do we need to clone() here? 
        size_of_mask = (channel, channel_f, height, width)
        input_feature_masked = torch.zeros(size_of_mask, dtype=torch.float32).to(self.device)
        for i in range(channel):
            input_feature_masked[i, :, :, :] = mask[i, :, :] * input_feature_3d
        
        inputs_G = list()
        for i in range(channel):
            temp = input_feature_masked[i, :, :, :]
            mask_mean = torch.mean(mask[i, :, :])
            if mask_mean > 0.0:
                inputs_G.append( gram_matrix(temp.unsqueeze(0))/mask_mean)
            else:
                inputs_G.append( gram_matrix(temp.unsqueeze(0)))
        for i in range(channel):
            mask_mean = torch.mean(mask[i, :, :])
            self.loss += F.mse_loss(inputs_G[i], self.targets[i]) * mask_mean
        
        return input_feature

class TVLoss(nn.Module):

    def __init__(self, device='cuda:0'):

        super(TVLoss, self).__init__()
        self.device = device
        self.ky = np.array([
            [[0, 0, 0],[0, 1, 0],[0,-1, 0]],
            [[0, 0, 0],[0, 1, 0],[0,-1, 0]],
            [[0, 0, 0],[0, 1, 0],[0,-1, 0]]
        ])
        self.kx = np.array([
            [[0, 0, 0],[0, 1,-1],[0, 0, 0]],
            [[0, 0, 0],[0, 1,-1],[0, 0, 0]],
            [[0, 0, 0],[0, 1,-1],[0, 0, 0]]
        ])
        self.conv_x = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_x.weight = nn.Parameter(torch.from_numpy(self.kx).float().unsqueeze(0).to(self.device),
                                          requires_grad=False)
        self.conv_y = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_y.weight = nn.Parameter(torch.from_numpy(self.ky).float().unsqueeze(0).to(self.device),
                                          requires_grad=False)

    def forward(self, input):
        height, width = input.size()[2:4]
        gx = self.conv_x(input)
        gy = self.conv_y(input)

        # gy = gy.squeeze(0).squeeze(0)
        # cv2.imwrite('gy.png', (gy*255.0).to('cpu').numpy().astype('uint8'))
        # exit()

        self.loss = torch.sum(gx**2 + gy**2)/2.0
        return input

class RealLoss(nn.Module):
    
    def __init__(self, laplacian_m):
        super(RealLoss, self).__init__()
        self.L = Variable(laplacian_m.detach(), requires_grad=False)

    def forward(self, input):
        channel, height, width = input.size()[1:4]
        self.loss = 0
        for i in range(channel):
            temp = input[0, i, :, :]
            temp = torch.reshape(temp, (1, height*width))
            r = torch.mm(self.L, temp.t())
            self.loss += torch.mm(temp , r)
       
        return input

# create a module to normalize input image so we can easily put it in a
# nn.Sequential
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        
        # self.mean = torch.tensor(mean).view(-1, 1, 1)
        # self.std = torch.tensor(std).view(-1, 1, 1)
        self.mean = mean.view(-1, 1, 1)
        self.std = std.view(-1, 1, 1)

    def forward(self, img):
        # normalize img
        return (img - self.mean) / self.std


# desired depth layers to compute style/content losses:
content_layers_default = ['conv4_2'] 
style_layers_default = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']
def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img, style_mask, content_mask, laplacian_m,
                               content_layer= content_layers_default,
                               style_layers=style_layers_default,device = 'cuda:0'):
    cnn = copy.deepcopy(cnn)

    # normalization module
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    # just in order to have an iterable access to or list of content.style losses
    content_losses = []
    style_losses = []
    tv_losses = []
    #real_losses = []

    # assuming that cnn is a nn.Sequential, so we make a new nn. Sequential
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(normalization)

    tv_loss = TVLoss(device=device)
    model.add_module("tv_loss_{}".format(0), tv_loss)
    tv_losses.append(tv_loss)
    num_pool = 1
    num_conv = 0
    content_num = 0
    style_num = 0
    for layer in cnn.children():          # cnn feature without fully connected layers
        if isinstance(layer, nn.Conv2d):
            num_conv += 1
            name = 'conv{}_{}'.format(num_pool, num_conv)
        elif isinstance(layer, nn.ReLU):
            name = 'relu{}_{}'.format(num_pool, num_conv)
            # The in-place version doesn't play very nicely with the ContentLoss
            # and StyleLoss we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(num_pool)
            num_pool += 1
            num_conv = 0
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn{}_{}'.format(num_pool, num_conv)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layer:
            # add content loss
            target = model(content_img).detach()
            # print('content target size: ', target.size())
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(content_num), content_loss)
            content_losses.append(content_loss)
            content_num += 1
        
        if name in style_layers:
            # add style loss:
            # print('style_:', style_img.type())
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature, style_mask.detach(), content_mask.detach(),device=device)
            model.add_module("style_loss_{}".format(style_num), style_loss)
            style_losses.append(style_loss)
            style_num += 1

    # now we trim off the layers after the last content and style losses
    for i in range(len(model)-1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i+1)]

    return model, style_losses, content_losses, tv_losses#, real_losses