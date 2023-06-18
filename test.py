import numpy as np
import os
import torch
import torchvision.transforms as transforms
from model_train import NetworkShadow
import cv2
import argparse




def load_network(ckpt_path):

    networkShadow = NetworkShadow()
    networkShadow.load_state_dict(torch.load(ckpt_path))
    networkShadow.eval()

    return networkShadow


def save_image(pre_shadow_img,model_mask, result_dir,threshold):


    final_img = torch.where( pre_shadow_img > threshold, 0, 255)
    # final_img = pre_shadow_img
    final_img = final_img.view(1, 240, 320)
    final_img = final_img.cpu().detach().numpy()
    final_img = np.transpose(final_img, (1,2,0))

    
    final_img1 = final_img * model_mask
    cv2.imwrite(result_dir , final_img1)





############################### Tester ############################

class Tester(object):
    def __init__(self,opt):

        self.dir_depth_model = opt.dir_depth_model
        self.dir_normal_model = opt.dir_normal_model
        self.dir_background = opt.dir_background
        self.dir_shadow = opt.dir_shadow
        self.dir_mask = opt.dir_mask
        self.ckpt_path = opt.ckpt_path
        self.threshold = opt.threshold
        self.result_dir = opt.save_path


    def test(self):



        filenameToPILImage = lambda x: cv2.imread(x, -1)

        self.networkShadow = load_network(self.ckpt_path)

        self.transform = transforms.Compose([filenameToPILImage,
                                            transforms.ToTensor(),   # H,W,C ->C,H,W
                                            ])


        # input image background
        image_background = self.transform(self.dir_background)  # 3

        # input model depth
        depth_model = cv2.imread(self.dir_depth_model, -1)  # 1
        depth_model = np.expand_dims(depth_model, -1)
        depth_model = np.transpose(depth_model, (2,0,1))
        depth_model = torch.from_numpy(depth_model)

        # input model normal   -1->1
        normal_model = cv2.imread(self.dir_normal_model, -1)  # 3
        normal_model = np.transpose(normal_model, (2,0,1))
        normal_model = torch.from_numpy(normal_model)


        # input background shadow
        mask_shadow_background = cv2.imread(self.dir_shadow, -1)  # 1
        mask_shadow_background = np.expand_dims(mask_shadow_background, -1)
        mask_shadow_background = np.transpose(mask_shadow_background, (2,0,1))
        mask_shadow_background = torch.from_numpy(mask_shadow_background)

        # input model mask
        model_mask = cv2.imread(self.dir_mask, -1)   # 1


        image_background = image_background.view(1, 3, 240, 320)
        depth_model = depth_model.view(1, 1, 240, 320)
        normal_model = normal_model.view(1, 3, 240, 320)
        mask_shadow_background = mask_shadow_background.view(1, 1, 240, 320)

        image_background = image_background.type(torch.FloatTensor)
        depth_model = depth_model.type(torch.FloatTensor)
        normal_model = normal_model.type(torch.FloatTensor)
        mask_shadow_background = mask_shadow_background.type(torch.FloatTensor)


        pre_shadow_img = self.networkShadow(depth_model, normal_model, image_background, mask_shadow_background)


        save_image(pre_shadow_img, model_mask, self.result_dir, self.threshold)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dir_depth_model', type=str, default="./input/COCO_train2014_000000206394_depth.exr")
    parser.add_argument('--dir_normal_model', type=str, default="./input/COCO_train2014_000000206394_normal.exr")
    parser.add_argument('--dir_background', type=str, default="./input/COCO_train2014_000000206394_background.jpg")
    parser.add_argument('--dir_shadow', type=str, default="./input/COCO_train2014_000000206394_shadow.exr")
    parser.add_argument('--dir_mask', type=str, default="./input/COCO_train2014_000000206394_mask.exr")
    parser.add_argument('--save_path', type=str, default="./result/COCO_train2014_000000206394_premask.png")
    parser.add_argument('--ckpt_path', type=str, default="./checkpoint.pkl")
    parser.add_argument('--threshold', type=int, default=0.5)
    opt, _ = parser.parse_known_args()


    tester = Tester(opt)
    print('start testing now')
    tester.test()