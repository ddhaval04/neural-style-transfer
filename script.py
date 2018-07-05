from __future__ import print_function
import time
import PIL
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from keras import backend
from keras.models import Model
from keras.applications.vgg16 import VGG16
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import models, transforms
import time
import sys
import os
import copy
from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imsave



def get_model(use_cuda):
    print("Fetching model ...")
    model = torchvision.models.vgg16(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    dtype = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

    return model, dtype


def image_loader(image_name, use_cuda):
    print("Loading Image ...")
    # imsize = 512 if use_cuda else 128  # use small size if no gpu
    imsize = 512
    prep = transforms.Compose([transforms.ToTensor()])
    # prep = transforms.Compose([transforms.ToTensor(),
    #                        transforms.Lambda(lambda x: x[torch.LongTensor([2,1,0])]), #turn to BGR
    #                        transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961], #subtract imagenet mean
    #                                             std=[1,1,1]),
    #                        transforms.Lambda(lambda x: x.mul_(255)),
    #                        ])
    image = Image.open(image_name)
    image = image.resize((imsize, imsize))
    image = Variable(prep(image))
    # fake batch dimension required to fit network's input dimensions
    image = image.unsqueeze(0)

    return image

# gram matrix and loss
class GramMatrix(nn.Module):
    def forward(self, input):
        b,c,h,w = input.size()
        F = input.view(b * c, h * w)
        G = torch.mm(F, F.t()) 
        
        return G.div_(h * w * b * c)


class StyleCNN(object):
    def __init__(self, style, content, combo_image, model):
        super(StyleCNN, self).__init__()
        self.style = style
        self.content = content
        self.combo_image = nn.Parameter(combo_image.data)
        self.content_layers = ['conv_4']
        self.style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
        self.content_weight = 1.0
        self.style_weight = 1000
        self.loss_network = model
        self.gram = GramMatrix()
        self.loss = nn.MSELoss()
        self.optimizer = optim.LBFGS([self.combo_image])
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.loss_network.cuda()
            self.gram.cuda()


    def train(self):
        def total_variation_loss(x):
            # imsize = 512 if self.use_cuda else 128
            imsize = 512
            a = torch.pow(x[:, :, :imsize-1, :imsize-1] - x[:, :, 1:, :imsize-1], 2)
            b = torch.pow(x[:, :, :imsize-1, :imsize-1] - x[:, :, :imsize-1, 1:], 2)
            return torch.pow(a + b, 1.25).sum()

        def closure_1():
            self.optimizer.zero_grad()
            combo_image = self.combo_image.clone()
            combo_image.data.clamp_(0, 1)
            content_img = self.content.clone()
            style_img = self.style.clone()
            count = 1
            not_inplace = lambda layer: nn.ReLU(inplace=False) if isinstance(layer, nn.ReLU) else layer
            style_loss = 0
            for layer in list(self.loss_network.features):
                layer = not_inplace(layer)
                if self.use_cuda:
                    layer.cuda()
                content_img = layer.forward(content_img)
                style_img = layer.forward(style_img)
                combo_image = layer.forward(combo_image)

                if isinstance(layer, nn.Conv2d):
                    layer_name = "conv_" + str(count)

                if layer_name in self.content_layers:
                    content_loss = self.content_weight * self.loss(combo_image, content_img.detach())

                if layer_name in self.style_layers:
                    style_gram = GramMatrix().forward(style_img)
                    combo_gram = GramMatrix().forward(combo_image)
                    style_loss += self.style_weight * self.loss(combo_gram, style_gram.detach())

                if isinstance(layer, nn.ReLU):
                    count += 1
            reg_loss = 1.0 * (
                    torch.sum(torch.abs(combo_image[:, :, :, :-1] - combo_image[:, :, :, 1:])) + 
                    torch.sum(torch.abs(combo_image[:, :, :-1, :] - combo_image[:, :, 1:, :]))
                )
            # variational_loss = 1.0 * total_variation_loss(combo_image)
            # reg_loss = 0
            total_loss = content_loss + style_loss + reg_loss
            total_loss.backward()
            print(total_loss.data[0])

            return total_loss

        self.optimizer.step(closure_1)

        return self.combo_image


def save_image(input):
    use_cuda = torch.cuda.is_available()
    # imsize = 512 if use_cuda else 128  # use small size if no gpu
    imsize = 512
    unloader = transforms.ToPILImage()
    image = input.data.clone().cpu()
    image = image.view(3, imsize, imsize)
    image = unloader(image)
    return image


def main():
    use_cuda = torch.cuda.is_available()
    model, dtype = get_model(use_cuda)
    content_path = "./content/dog.jpg"
    style_path = "./style/wave.jpg"
    style_img = image_loader(style_path, use_cuda).type(dtype)
    content_img = image_loader(content_path, use_cuda).type(dtype)
    combo_image = Variable(content_img.data.clone())
    # combo_image = nn.Parameter(combo_image)
    if use_cuda:
      content_img = content_img.cuda()
      style_img = style_img.cuda()
    print(content_img.size(), style_img.size())
    style_cnn = StyleCNN(style_img, content_img, combo_image, model)
    result = []
    print("Training ...")
    for i in range(30):
        combo_image = style_cnn.train()
        combo_image.data.clamp_(0, 1)
        result.append(save_image(combo_image))
        if i % 10 == 0:
            print("Iteration: ", i)
            # combo_image.data.clamp_(0, 1)
            # result.append(save_image(combo_image))

    count = 1
    for r in result:
        plt.imshow(r)
        plt.show()
        r.save("./results/" + str(count) + ".jpg")
        count = count + 1

if __name__ == '__main__':
    sys.exit(main())