'''
    This part of code is completely copied from a youtuber's tutorial.
    ref link: https://github.com/niconielsen32/ComputerVision/blob/master/headPoseEstimation.py
'''
import cv2
import mediapipe as mp
import numpy as np
import time

import torch

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)


def head_pose_estimation(imgs, source=True):

    # Initialize the return variable.
    batch = imgs.size(0)
    res = np.zeros((imgs.size(0), 3, 3))
    print(res.shape)

    # Convert tensor to np.array
    imgs = imgs.numpy()

    for i in range(batch):
        img = imgs[i].astype(np.uint8)
        print("img shape in head_pose is :" + str(img.shape))
        print(img)

        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # To improve performance
        image.flags.writeable = False

        # Get the result
        results = face_mesh.process(image)

        # To improve performance
        image.flags.writeable = True

        # Convert the color space from RGB to BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        img_h, img_w, img_c = image.shape
        face_3d = []
        face_2d = []

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for idx, lm in enumerate(face_landmarks.landmark):
                    if idx == 33 or idx == 263 or idx == 1 or idx == 61 or idx == 291 or idx == 199:
                        if idx == 1:
                            nose_2d = (lm.x * img_w, lm.y * img_h)
                            nose_3d = (lm.x * img_w, lm.y * img_h, lm.z * 3000)

                        x, y = int(lm.x * img_w), int(lm.y * img_h)

                        # Get the 2D Coordinates
                        face_2d.append([x, y])

                        # Get the 3D Coordinates
                        face_3d.append([x, y, lm.z])

                        # Convert it to the NumPy array
                face_2d = np.array(face_2d, dtype=np.float64)

                # Convert it to the NumPy array
                face_3d = np.array(face_3d, dtype=np.float64)

                # The camera matrix
                focal_length = 1 * img_w

                cam_matrix = np.array([[focal_length, 0, img_h / 2],
                                       [0, focal_length, img_w / 2],
                                       [0, 0, 1]])

                # The distortion parameters
                dist_matrix = np.zeros((4, 1), dtype=np.float64)

                # Solve PnP
                success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

                # Get rotational matrix
                rmat, jac = cv2.Rodrigues(rot_vec)

                # Generate the homography matrix H = I @ [R|t]
                # where we only extract the first two columns of R
                # in order to keep the output size of trans_mat (3,3)
                # ref link: https://towardsdatascience.com/estimating-a-homography-matrix-522c70ec4b2c
                R_t = np.concatenate((rmat[:, 0:2], trans_vec), axis=1)
                trans_mat = cam_matrix.dot(R_t)
                print(trans_mat.shape)

                if source:
                    res[i] = np.linalg.inv(trans_mat).copy()
                else:
                    res[i] = trans_mat.copy()

    return torch.Tensor(res)
import cv2 as cv
import numpy as np
from mtcnn.mtcnn import MTCNN
import os


acted_anger_source_path = "./videos_zakir_80/videos_zakir_80/AA5.mp4"
genuine_anger_driver_path = "./videos_zakir_80/videos_zakir_80/GA1.mp4"
source_imgs_path = "./source_img"
driver_imgs_path = "./driver_img"

print("current work dir" + os.getcwd())

def face_detect(img):

    detector = MTCNN()
    # Detect faces
    faces = detector.detect_faces(img)
    print("The output of faces is ")
    print(faces)

    return faces




def capture_frames(path, flag, n=256):
    '''

    :param n:
    :param path:
    :param flag: if flag = 1, then it generate source img, otherwise, generate driver img.
    :return:
    '''
    counter = 1

    cap = cv.VideoCapture(path)
    #cap.set(cv.CAP_PROP_FPS, 60.0)
    print("Current sampling FPS is : " + str(cap.get(cv.CAP_PROP_FPS)))



    # ref: https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        if flag == 1:
            filename = "A"
        else:
            filename = "G"

            # face detect part
            # face = face_detect(frame)
            # x, y, w, h = face[0]['box']
            # print(frame.shape)
            # frame = frame[x:x+w, y:y+h, :]


        img = cv.resize(frame, (n, n))
        cwd = os.getcwd()
        if flag == 1:
            os.chdir(source_imgs_path)
        else:
            os.chdir(driver_imgs_path)
        filename = filename + str(counter) + ".jpg"
        cv.imwrite(filename=filename, img=img)
        os.chdir(cwd)

        counter += 1

    cap.release()


capture_frames(acted_anger_source_path, flag=1)
capture_frames(genuine_anger_driver_path, flag=0)import numpy as np
from torch.utils.data import Dataset
import cv2 as cv

source_img_path = './source_img'
driver_img_path = './driver_img'

class SourceDataset(Dataset):
    def __init__(self, source_img_path, length, transform=None):
        self.source_img_path = source_img_path
        self.length = length
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        img_path = self.source_img_path + "/A" + str(item + 1) + ".jpg"
        img = cv.imread(img_path)
        img = np.array(img)

        if self.transform:
            img = self.transform(img)

        return img.astype(np.float16)


class DriverDataset(Dataset):
    def __init__(self, driver_img_path, length, transform=None):
        self.driver_img_path = driver_img_path
        self.length = length
        self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        img_path = self.driver_img_path + "/G" + str(item + 1) + ".jpg"
        img = cv.imread(img_path)
        img = np.array(img)

        if self.transform:
            img = self.transform(img)

        return img.astype(np.float16)import gc
import torch
import torch.nn as nn
import torch.nn.functional as F



class Conv2d_WS(nn.Conv2d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2d_WS, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                  keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

class Conv3D_WS(nn.Conv3d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv3D_WS, self).__init__(in_channels, out_channels, kernel_size, stride,
                                        padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                  keepdim=True).mean(dim=3, keepdim=True).mean(
                                  dim=4, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv3d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)



class ResBlock_Custom(nn.Module):
    def __init__(self, dimension, input_channels, output_channels):
        super().__init__()
        self.dimension = dimension
        self.input_channels = input_channels
        self.output_channels = output_channels
        if dimension == 2:
            self.conv_res = nn.Conv2d(self.input_channels, self.output_channels, 3, padding= 1)
            self.conv_ws = Conv2d_WS(in_channels = self.input_channels,
                                  out_channels= self.output_channels,
                                  kernel_size = 3,
                                  padding = 1)
            self.conv = nn.Conv2d(self.output_channels, self.output_channels, 3, padding = 1)
        elif dimension == 3:
            self.conv_res = nn.Conv3d(self.input_channels, self.output_channels, 3, padding=1)
            self.conv_ws = Conv3D_WS(in_channels=self.input_channels,
                                     out_channels=self.output_channels,
                                     kernel_size=3,
                                     padding=1)
            self.conv = nn.Conv3d(self.output_channels, self.output_channels, 3, padding=1)


    def forward(self, x):
        out2 = self.conv_res(x)

        out1 = F.group_norm(x, num_groups=32)
        out1 = F.relu(out1)
        out1 = self.conv_ws(out1)
        out1 = F.group_norm(out1, num_groups=32)
        out1 = F.relu(out1)
        out1 = self.conv(out1)

        output = out1 + out2

        return output



class Eapp1(nn.Module):
    '''
        This is the first part of the Appearance Encoder. To generate
        a 4D tensor of volumetric features vs.
    '''
    def __init__(self):
        # first conv layer, output size: 512 * 512 * 64
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 7, stride=1, padding=3)                                            # output 512*512*64
        self.resblock_128 = ResBlock_Custom(dimension=2, input_channels=64, output_channels=128)        # output 512*512*128
        self.resblock_256 = ResBlock_Custom(dimension=2, input_channels=128, output_channels=256)       # output 512*512*256
        self.resblock_512 = ResBlock_Custom(dimension=2, input_channels=256, output_channels=512)       # output 512*512*512
        self.resblock3D_96 = ResBlock_Custom(dimension=3, input_channels=96, output_channels=96)        # output
        self.resblock3D_96_2 = ResBlock_Custom(dimension=3, input_channels=96, output_channels=96)
        self.conv_1 = nn.Conv2d(in_channels=512, out_channels=1536, kernel_size=1, stride=1, padding=0)

        self.avgpool = nn.AvgPool2d(kernel_size=5, stride=1, padding=2)


    def forward(self, x):
        out = self.conv(x)
        print("After first layer:" + str(out.size()))
        out = self.resblock_128(out)
        print("After resblock_128:" + str(out.size()))
        out = self.avgpool(out)
        print("After avgpool:" + str(out.size()))
        out = self.resblock_256(out)
        print("After resblock_256:" + str(out.size()))
        out = self.avgpool(out)
        print("After avgpool:" + str(out.size()))
        out = self.resblock_512(out)
        print("After resblock_512:" + str(out.size()))
        out = self.avgpool(out)
        print("After avgpool:" + str(out.size()))

        out = F.group_norm(out, num_groups=32)
        print("After group_norm:" + str(out.size()))
        out = F.relu(out)
        print("After relu:" + str(out.size()))

        out = self.conv_1(out)
        print("After conv_1:" + str(out.size()))

        # Reshape
        out = out.view(out.size(0), 96, 16, out.size(2), out.size(3))
        print("After reshape:" + str(out.size()))

        # ResBlock 3D
        out = self.resblock3D_96(out)
        print("After resblock3D:" + str(out.size()))
        out = self.resblock3D_96_2(out)
        print("After resblock3D_96_2:" + str(out.size()))
        out = self.resblock3D_96_2(out)
        print("After resblock3D_96_2:" + str(out.size()))
        out = self.resblock3D_96_2(out)
        print("After resblock3D_96_2:" + str(out.size()))
        out = self.resblock3D_96_2(out)
        print("After resblock3D_96_2:" + str(out.size()))
        out = self.resblock3D_96_2(out)
        print("After resblock3D_96_2:" + str(out.size()))

        return out




class Eapp2(nn.Module):
    '''
        This is the second part of the Appearance Encoder. To generate
        a global descriptor es that helps retain the appearance of the output
        image.
        This encoder uses ResNet-50 as backbone, and replace the residual block with the customized res-block.
        ref: https://towardsdev.com/implement-resnet-with-pytorch-a9fb40a77448
    '''
    def __init__(self, repeat, in_channels=3, outputs=256):
        super().__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        filters = [64, 256, 512, 1024, 2048]

        self.layer1 = nn.Sequential()
        self.layer1.add_module('conv2_1', ResBlock_Custom(dimension=2, input_channels=filters[0], output_channels=filters[1]))
        for i in range(1, repeat[0]):
                self.layer1.add_module('conv2_%d'%(i+1,), ResBlock_Custom(dimension=2, input_channels=filters[1], output_channels=filters[1]))

        self.layer2 = nn.Sequential()
        self.layer2.add_module('conv3_1', ResBlock_Custom(dimension=2, input_channels=filters[1], output_channels=filters[2]))
        for i in range(1, repeat[1]):
                self.layer2.add_module('conv3_%d' % (i+1,), ResBlock_Custom(dimension=2, input_channels=filters[2], output_channels=filters[2]))

        self.layer3 = nn.Sequential()
        self.layer3.add_module('conv4_1', ResBlock_Custom(dimension=2, input_channels=filters[2], output_channels=filters[3]))
        for i in range(1, repeat[2]):
            self.layer3.add_module('conv2_%d' % (i+1,), ResBlock_Custom(dimension=2, input_channels=filters[3], output_channels=filters[3]))

        self.layer4 = nn.Sequential()
        self.layer4.add_module('conv5_1', ResBlock_Custom(dimension=2, input_channels=filters[3], output_channels=filters[4]))
        for i in range(1, repeat[3]):
            self.layer4.add_module('conv3_%d'%(i+1,), ResBlock_Custom(dimension=2, input_channels=filters[4], output_channels=filters[4]))

        self.gap = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Linear(filters[4], outputs)


    def forward(self, input):
        input = self.layer0(input)
        input = self.layer1(input)
        input = self.layer2(input)
        input = self.layer3(input)
        input = self.layer4(input)
        input = self.gap(input)
        input = torch.flatten(input, start_dim=1)
        input = self.fc(input)
        print("Dimensions of final output of Eapp2: " + str(input.size()))

        return input



class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False):
        super().__init__()
        if downsample:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.shortcut = nn.Sequential()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, input):
        shortcut = self.shortcut(input)
        input = nn.ReLU()(self.bn1(self.conv1(input)))
        input = nn.ReLU()(self.bn2(self.conv2(input)))
        input = input + shortcut
        return nn.ReLU()(input)



class Emtn_facial(nn.Module):
    def __init__(self, in_channels, resblock=ResBlock, outputs=256):
        super().__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.layer1 = nn.Sequential(
            resblock(64, 64, downsample=False),
            resblock(64, 64, downsample=False)
        )

        self.layer2 = nn.Sequential(
            resblock(64, 128, downsample=True),
            resblock(128, 128, downsample=False)
        )

        self.layer3 = nn.Sequential(
            resblock(128, 256, downsample=True),
            resblock(256, 256, downsample=False)
        )

        self.layer4 = nn.Sequential(
            resblock(256, 512, downsample=True),
            resblock(512, 512, downsample=False)
        )

        self.gap = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Linear(512, outputs)

    def forward(self, input):
        input = self.layer0(input)
        input = self.layer1(input)
        input = self.layer2(input)
        input = self.layer3(input)
        input = self.layer4(input)
        input = self.gap(input)
        input = torch.flatten(input, 1)
        input = self.fc(input)
        print("Dimensions of final output of Emtn_facial: " + str(input.size()))

        return input



class Emtn_head(nn.Module):
    def __init__(self, in_channels, resblock, outputs=256):
        super().__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self.layer1 = nn.Sequential(
            resblock(64, 64, downsample=False),
            resblock(64, 64, downsample=False)
        )

        self.layer2 = nn.Sequential(
            resblock(64, 128, downsample=True),
            resblock(128, 128, downsample=False)
        )

        self.layer3 = nn.Sequential(
            resblock(128, 256, downsample=True),
            resblock(256, 256, downsample=False)
        )

        self.layer4 = nn.Sequential(
            resblock(256, 512, downsample=True),
            resblock(512, 512, downsample=False)
        )

        self.gap = torch.nn.AdaptiveAvgPool2d(1)
        self.fc = torch.nn.Linear(512, outputs)

    def forward(self, input):
        input = self.layer0(input)
        input = self.layer1(input)
        input = self.layer2(input)
        input = self.layer3(input)
        input = self.layer4(input)
        input = self.gap(input)
        input = torch.flatten(input)
        input = self.fc(input)

        return input



## Need to learn more about the adaptive group norm.
class ResBlock3D_Adaptive(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.conv_res = nn.Conv3d(self.input_channels, self.output_channels, 3, padding=1)
        self.conv_ws = Conv3D_WS(in_channels=self.input_channels,
                                 out_channels=self.output_channels,
                                 kernel_size=3,
                                 padding=1)
        self.conv = nn.Conv3d(self.output_channels, self.output_channels, 3, padding=1)


    def forward(self, x):
        out2 = self.conv_res(x)

        out1 = F.group_norm(x, num_groups=32)
        out1 = F.relu(out1)
        out1 = self.conv_ws(out1)
        out1 = F.group_norm(out1, num_groups=32)
        out1 = F.relu(out1)
        out1 = self.conv(out1)

        output = out1 + out2

        return output


class WarpGenerator(nn.Module):
    def __init__(self, input_channels):
        super(WarpGenerator, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_channels, out_channels=2048, kernel_size=1, padding=0, stride=1)
        self.hidden_layer = nn.Sequential(
            ResBlock_Custom(dimension=3, input_channels=512, output_channels=256),
            nn.Upsample(scale_factor=(2, 2, 2)),
            ResBlock_Custom(dimension=3, input_channels=256, output_channels=128),
            nn.Upsample(scale_factor=(2, 2, 2)),
            ResBlock_Custom(dimension=3, input_channels=128, output_channels=64),
            nn.Upsample(scale_factor=(1, 2, 2)),
            ResBlock_Custom(dimension=3, input_channels=64, output_channels=32),
            nn.Upsample(scale_factor=(1, 2, 2)),
        )
        self.conv3D = nn.Conv3d(in_channels=32, out_channels=3, kernel_size=3, padding=1, stride=1)

    def forward(self, x):
        out = self.conv1(x)
        print("The output shape after first conv layer is " + str(out.size()))
        # reshape
        out = out.view(out.size(0), 512, 4, 16, 16)
        print("The output shape after reshaping is " + str(out.size()))

        out = self.hidden_layer(out)
        print("The output shape after hidden_layer is " + str(out.size()))
        out = F.group_norm(out, num_groups=32)
        print("The output shape after group_norm is " + str(out.size()))
        out = F.relu(out)
        print("The output shape after relu is " + str(out.size()))
        out = torch.tanh(out)
        print("The final output shape is : " + str(out.size()))

        return out



class G3d(nn.Module):
    def __init__(self, input_channels):
        super(G3d, self).__init__()
        self.input_channels = input_channels

    def forward(self, x):
        out = ResBlock_Custom(dimension=3, input_channels=self.input_channels, output_channels=192)(x)
        out = nn.Upsample(scale_factor=(1/2, 2, 2))(out)
        short_cut1 = ResBlock_Custom(dimension=3, input_channels=192, output_channels=196)(out)
        out = ResBlock_Custom(dimension=3, input_channels=192, output_channels=384)(out)
        out = nn.Upsample(scale_factor=(2, 2, 2))(out)
        short_cut2 = ResBlock_Custom(dimension=3, input_channels=384, output_channels=384)(out)
        out = ResBlock_Custom(dimension=3, input_channels=384, output_channels=512)(out)
        short_cut3 = out
        out = ResBlock_Custom(dimension=3, input_channels=512, output_channels=512)(out)
        out = short_cut3 + out
        out = ResBlock_Custom(dimension=3, input_channels=512, output_channels=384)(out)
        out = nn.Upsample(scale_factor=(2, 2, 2))(out)
        out = out + short_cut2
        out = ResBlock_Custom(dimension=3, input_channels=384, output_channels=196)(out)
        out = nn.Upsample(scale_factor=(1/2, 2, 2))(out)
        out = out + short_cut1
        out = ResBlock_Custom(dimension=3, input_channels=196, output_channels=96)(out)

        # Last Layer.
        out = F.group_norm(out, num_groups=32)
        out = F.relu(out)
        out = nn.Conv3d(in_channels=96, out_channels=96, kernel_size=3, padding=1, stride=1)(out)

        return out




class G2d(nn.Module):
    def __init__(self, input_channels):
        super(G3d, self).__init__()
        self.input_channels = input_channels
        self.conv1 = nn.Conv2d(in_channels=self.input_channels, out_channels=512, kernel_size=1, padding=0, stride=1)
        self.repeat_resblock = nn.Sequential(
            ResBlock_Custom(dimension=2, input_channels=512, output_channels=512)
        )

        for i in range(1, 8):
            self.repeat_resblock.add_module(module=ResBlock_Custom(dimension=2, input_channels=512, output_channels=512))

        self.upsamples = nn.Sequential(
            nn.Upsample(scale_factor=(2, 2)),
            ResBlock_Custom(dimension=2, input_channels=512, output_channels=256),
            nn.Upsample(scale_factor=(2, 2)),
            ResBlock_Custom(dimension=2, input_channels=256, output_channels=128),
            nn.Upsample(scale_factor=(2, 2)),
            ResBlock_Custom(dimension=2, input_channels=128, output_channels=64),
        )

        self.last_layer = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, padding=1, stride=1)
        )

    def forward(self, x):
        ##TODO: Placeholder for reshaping.
        out = self.conv1(x)
        out = self.repeat_resblock(out)
        out = self.upsamples(out)
        out = F.group_norm(out, num_groups=32)
        out = F.relu(out)
        out = self.last_layer(out)
        out = F.sigmoid(out)

        return out

'''
    An pytorch implementation of patchGAN.
'''
import torch.nn as nn
import torch.nn.functional as F
import torch

class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels * 2, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
        )

    def forward(self, img_A, img_B):
        # Concatenate image and condition image by channels to produce input
        img_input = torch.cat((img_A, img_B), 1)
        return self.model(img_input)import numpy as np
import torch
import cv2 as cv
import matplotlib.pyplot as plt
import gc
import torch.nn.functional as F
import model
import HeadPoseEstimation
from torch.utils.data import DataLoader
import dataset

source_img_path = './source_img'
driver_img_path = './driver_img'
data_set_length = 42

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

input1 = torch.randn(2, 3, 256, 256)


def tensor2image(image, norm=255.0):
    return (image.squeeze().permute(0, 1, 2).numpy() * norm)


# mean = input1.mean(dim=0, keepdim=True)
#
# test_model = model.Emtn_facial(in_channels=3)
# test_model2 = model.Eapp2(repeat=[3, 4, 6, 3])
# output = test_model(input1)
# print(output)
# output2 = test_model2(input1)
# print(output2)
# sum = output2 + output
# print(sum)
#
# print("The dimension of the sum is: " + str(sum.size()))
# test_model = model.Eapp1()
# output = test_model(input1)

# Test for dataloader
source_set_origin = dataset.SourceDataset(source_img_path, data_set_length, transform=None)
driver_set_origin = dataset.DriverDataset(driver_img_path, data_set_length, transform=None)

source_loader_origin = DataLoader(source_set_origin, batch_size=2, shuffle=False)
driver_loader_origin = DataLoader(driver_set_origin, batch_size=2, shuffle=False)

imgs = next(iter(source_loader_origin))
print("imgs.shape is :" + str(imgs.size()))
w_rt = HeadPoseEstimation.head_pose_estimation(imgs)

#print(output)

print(w_rt.numpy().shape)

#Test for warp_generator
# input_warp = torch.randn(2, 1,  256)
# warp_generator = model.WarpGenerator(input_channels=1)
# output_warp = warp_generator(input_warp)
#
# print(output_warp)

# Test for head-pose removal.


for i in range(2):
    print(imgs[i].shape)
    img = tensor2image(imgs[i]).astype(np.uint8)
    print("********** image *********")
    print(img.shape)
    homography = w_rt[i].numpy()
    print("********* homography ********")
    print(homography.shape)
    output = cv.warpPerspective(img, homography, (img.shape[1], img.shape[0]))

    plt.imshow(output)
    plt.show()
import argparse
import torch
import dataset
import model
import cv2 as cv
import HeadPoseEstimation
import vgg_face
import numpy as np
import torch.nn as nn
import patchGAN
import random
import torch.optim as optim

from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.autograd import Variable
from torch.optim.lr_scheduler import CosineAnnealingLR

source_img_path = './source_img'
driver_img_path = './driver_img' 
data_set_length = 42
img_size = 512

hinge_loss = nn.HingeEmbeddingLoss(reduction='mean')
L1_loss = nn.L1Loss(reduction='mean')
feature_matching_loss = nn.MSELoss()
cosine_dist = nn.CosineSimilarity()

patch = (1, img_size // 2 ** 4, img_size // 2 ** 4)

def cosine_distance(args, z1, z2):
  
  res = args.s_cos * (torch.sum(cosine_dist(z1[0], z2[0])) - args.m_cos)
  res += args.s_cos * (torch.sum(cosine_dist(z1[1], z2[1])) - args.m_cos)   
  res += args.s_cos * (torch.sum(cosine_dist(z1[2], z2[2])) - args.m_cos)
  
  return res

def cosine_loss(args, descriptor_driver, 
                descriptor_source_rand, descriptor_driver_rand):
  
  z_dri = descriptor_driver
  z_dri_rand = descriptor_driver_rand
  
  # Create descriptors to form the pairs
  # z_s*->d
  z_src_rand_dri = [descriptor_source_rand[0], 
                    z_dri[1], 
                    z_dri[2]]

  # Form the pairs
  pos_pairs = [(z_dri, z_dri), (z_src_rand_dri, z_dri)]
  neg_pairs = [(z_dri, z_dri_rand), (z_src_rand_dri, z_dri_rand)]

  # Calculate cos loss
  sum_neg_paris = torch.exp(cosine_distance(args, neg_pairs[0][0], neg_pairs[0][1])) + torch.exp(cosine_distance(args, neg_pairs[1][0], neg_pairs[1][1]))
  
  L_cos = torch.zeros(dtype=torch.float)
  for i in range(len(pos_pairs)):
    L_cos += torch.log(torch.exp(cosine_distance(args, pos_pairs[0][0], pos_pairs[0][1])) / (torch.exp(cosine_distance(args, pos_pairs[0][0], pos_pairs[0][1])) + sum_neg_paris))
  
  return L_cos
      

def train(args, models, device, driver_loader, source_loader, optimizers, schedulers, source_img_random, driver_img_random,
          source_loader_origin, driver_loader_origin):

  # Instantiate each model.
  Eapp1 = models['Eapp1'] 
  Eapp2 = models['Eapp2']
  Emtn_facial = models['Emtn_facial']
  Warp_G = models['Warp_G']
  G3d = models['G3d']
  G2d = models['G2d']
  vgg_IN = models['Vgg_IN']
  vgg_face = models['Vgg_face']
  discriminator = models['patchGAN']
         
  train_loss = 0.0

  # Training procedure starts here.
  for idx in range(args.iteration):
    # Ending condition for training.
    if idx > args.iteration:
      print(" Training complete!")

      break
    else:
      idx += 1

    # loading a single data
    source_img = next(iter(source_loader)).to(device)
    driver_img = next(iter(driver_loader)).to(device)
    source_imgs_origin = next(iter(source_loader_origin))
    driver_imgs_origin = next(iter(driver_loader_origin))

    # pass the data through Eapp1 & 2.
    v_s = Eapp1(source_img) 
    e_s = Eapp2(source_img)

    # Emtn.

    # Second part of Emtn : Generate facial expression latent vector z
    # based on a ResNet-18 network
    z_s = Emtn_facial(source_img)
    z_d = Emtn_facial(driver_img)
    
    # Warp_Generator
    
    # First part of Warp Generator: Generate warping matrix  
    # based on its transformation matrix.
    # Note: the head pose prediction is also completed in this function.
    W_rt_s = HeadPoseEstimation.head_pose_estimation(source_imgs_origin)
    W_rt_d = HeadPoseEstimation.head_pose_estimation(driver_imgs_origin)
    
    # Second part of Warp Generator: Generate emotion warper.
    W_em_s = Warp_G(z_s + e_s)
    W_em_d = Warp_G(z_d + e_s)
    
    # 3D warping of w_s and v_s
    # First, 3D warping using w_rt_s
    warp_3d_vs = cv.warpPerspective(v_s, W_rt_s, (v_s.shape[1], v_s.shape[0]))
    # Next, 3D warping using w_em_s
    warp_3d_vs = cv.warpPerspective(warp_3d_vs, W_em_s, (warp_3d_vs.shape[1], warp_3d_vs.shape[0]))
    
    # Pass data into G3d
    output = G3d(warp_3d_vs)
    
    # 3D warping with w_d
    vs_d = cv.warpPerspective(warp_3d_vs, W_rt_d, (warp_3d_vs.shape[1], warp_3d_vs.shape[0]))
    vs_d = cv.warpPerspective(vs_d, W_em_d, (vs_d.shape[1], vs_d.shape[0]))

    # Pass into G2d.
    output = G2d(vs_d)
  
    # IN loss
    L_IN = L1_loss(vgg_IN(output), vgg_IN(driver_img))
    
    # face loss
    L_face = L1_loss(vgg_face(output), vgg_face(driver_img))
    
    # adv loss
    # Adversarial ground truths
    valid = Variable(torch.Tensor(np.ones((driver_img.size(0), *patch))), requires_grad=False)
    fake = Variable(torch.Tensor(-1*np.ones((driver_img.size(0), *patch))), requires_grad=False)
        
    # real loss
    pred_real = discriminator(driver_img, source_img)
    loss_real = hinge_loss(pred_real, valid)

    # fake loss        
    pred_fake = discriminator(output.detach(), source_img)
    loss_fake = hinge_loss(pred_fake, fake)
        
    L_adv = 0.5 * (loss_real + loss_fake)
    
    # feature mapping loss
    L_feature_matching = feature_matching_loss(output, driver_img)
    
    # Cycle consistency loss 
    # Feed base model with randomly sampled image.
    e_s_rand = Eapp2(source_img_random)
    trans_mat_source_rand = HeadPoseEstimation.head_pose_estimation(source_img_random)
    z_s_rand = Emtn_facial(source_img_random)

    trans_mat_driver_rand = HeadPoseEstimation.head_pose_estimation(driver_img_random)
    z_d_rand = Emtn_facial(driver_img_random)
    
    descriptor_driver = [e_s, W_rt_d, z_d]
    descriptor_source_rand = [e_s_rand, trans_mat_source_rand, z_s_rand]
    descriptor_driver_rand = [e_s_rand, trans_mat_driver_rand, z_d_rand]
    
    L_cos = cosine_loss(args,descriptor_driver, 
                            descriptor_source_rand, descriptor_driver_rand)
    
    L_per = args.weight_IN * L_IN + args.weight_face * L_face
        
    L_gan = args.weight_adv * L_adv + args.weight_FM * L_feature_matching
        
    L_final = L_per + L_gan + args.weight_cos * L_cos
        
    # Optimizer and Learning rate scheduler.
    # optimizer
    for i in range(len(optimizers)):
      optimizers[i].zero_grad()
        
    L_final.backward()
        
    for i in range(len(optimizers)):  
      optimizers[i].step()
      schedulers[i].step()
        
    train_loss += L_final
    train_loss /= idx
    
    # Print log 
    print('Iteration: {} / {} : train loss is: {}'.format(idx, args.iteration, train_loss))

def distill(args, teacher, student, device, driver_loader):

    # Set teacher model to eval mode
    teacher.eval()

    # Training loop for student
    for idx in range(args.iteration):

        # Sample driver frame and index
        driver_img = next(iter(driver_loader)).to(device)
        idx = random.randint(0, args.num_avatars-1)

        # Generate pseudo-ground truth with teacher
        with torch.no_grad():
            output_HR = teacher(driver_img, idx) 

        # Get student prediction
        output_DT = student(driver_img, idx)

        # Calculate losses
        L_per = perceptual_loss(output_DT, output_HR) 
        L_adv = adversarial_loss(output_DT, output_HR)

        L_final = L_per + L_adv

        # Optimize student
        student_optimizer.zero_grad()
        L_final.backward()
        student_optimizer.step()

        # Print log
        if idx % args.print_freq == 0:
            print('Iteration: {} / {} : distillation loss is: {}'.format(idx, args.iteration, L_final.item()))
            
            
def main():
    parser = argparse.ArgumentParser(description="Megaportrait Pytorch implementation.")
    parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                        help='input batch size for training. default=16')
    parser.add_argument('--iteration', type=int, default=20000, metavar='N',
                        help='input batch size for training. default=20000')  
    parser.add_argument('--epochs', type=int, default=50, metavar='N',
                        help='number of epochs to train. default = 50')
    parser.add_argument('--weight-IN', type=int, default=20, metavar='N',
                        help='weight parameter for IN loss. default = 20')
    parser.add_argument('--weight-face', type=int, default=5, metavar='N',
                        help='weight parameter for face loss. default = 5')
    parser.add_argument('--weight-adv', type=int, default=1, metavar='N',
                        help='weight parameter for adv loss. default = 1')
    parser.add_argument('--weight-FM', type=int, default=40, metavar='N',
                        help='weight parameter for feature matching loss. default = 40')
    parser.add_argument('--weight-cos', type=int, default=2, metavar='N',
                        help='weight parameter for cos loss. default = 2')
    parser.add_argument('--s-cos', type=int, default=5, metavar='N',
                        help='s parameter in cos loss. default = 5')
    parser.add_argument('--m-cos', type=float, default=0.2, metavar='N',
                        help='m parameter in cos loss. default = 0.2')
    parser.add_argument('--lr', type = float, default=2e-4, metavar='LR',
                        help='learning rate')
    parser.add_argument('--num-avatars', type=int, default=100, metavar='N',
                        help='number of avatars for distillation')
    parser.add_argument('--print-freq', type=int, default=100, metavar='N',
                        help='print frequency for distillation')               
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")
  
    # Transformation on source images.
    transform_source = transforms.Compose([
        transforms.ToTensor(),
        # Normalize data into range(-1, 1)
        transforms.Normalize([0.5], [0.5]),
        # Randomly flip train data(left and right).
        transforms.RandomHorizontalFlip(),
        # Color jitter on data.
        transforms.ColorJitter()
    ])

    # Transformation on driver images.
    transform_driver = transforms.Compose([
        transforms.ToTensor(),
        # Normalize data into range(-1, 1)
        transforms.Normalize([0.5], [0.5]),
        # Randomly flip train data(left and right).
        transforms.RandomHorizontalFlip(),
        # Color jitter on data.
        transforms.ColorJitter()
    ])

    # Define dataset loaders
    source_set = dataset.SourceDataset(source_img_path, data_set_length, transform=transform_source)
    driver_set = dataset.DriverDataset(driver_img_path, data_set_length, transform=transform_driver)

    source_loader = DataLoader(source_set, batch_size=args.batch_size, shuffle=False)
    driver_loader = DataLoader(driver_set, batch_size=args.batch_size, shuffle=False)

    # Original data to compute transformation matrix.
    source_set_origin = dataset.SourceDataset(source_img_path, data_set_length, transform=None)
    driver_set_origin = dataset.DriverDataset(driver_img_path, data_set_length, transform=None)

    source_loader_origin = DataLoader(source_set_origin, batch_size=args.batch_size, shuffle=False)
    driver_loader_origin = DataLoader(driver_set_origin, batch_size=args.batch_size, shuffle=False)


    # Generate random pairs for calculating cos loss.
    random.seed(0)
    [index_source, index_driver] = random.sample(range(0, 29999), 2)
    source_img_random = Image.open("./CelebA-HQ-img/" + str(index_source) + ".jpg")
    driver_img_random = Image.open("./CelebA-HQ-img/" + str(index_driver) + ".jpg")
    # Apply the same transformation on these two images.
    source_img_random = transform_source(source_img_random)
    driver_img_random = transform_driver(driver_img_random)

    Eapp1 = model.Eapp1().to(device)
    Eapp2 = model.Eapp2().to(device)
    Emtn_facial = model.Emtn_facial().to(device)
    Emtn_head = model.Emtn_head().to(device)
    Warp_G = model.WarpGenerator().to(device)
    G3d = model.G3d().to(device)
    G2d = model.G2d().to(device)
    Vgg_IN = torch.hub.load('pytorch/vision:v0.10.0', 'vgg19', pretrained=True)
    Vgg_face = vgg_face.VggFace().to(device)
    discriminator = patchGAN.Discriminator().to(device)

    models = {
        'Eapp1': Eapp1,
        'Eapp2': Eapp2,
        'Emtn_facial': Emtn_facial,
        'Emtn_head': Emtn_head,
        'Warp_G': Warp_G,
        'G3d': G3d,
        'G2d': G2d,
        'Vgg_IN': Vgg_IN,
        'Vgg_face': Vgg_face,
        'patchGAN': discriminator
    }

    optimizers = [
        optim.Adam(models['Eapp1'], lr=args.lr, betas=(0.5, 0.999), eps=1e-8, weight_decay=1e-2),
        optim.Adam(models['Eapp2'], lr=args.lr, betas=(0.5, 0.999), eps=1e-8, weight_decay=1e-2),
        optim.Adam(models['Emtn_facial'], lr=args.lr, betas=(0.5, 0.999), eps=1e-8, weight_decay=1e-2),
        optim.Adam(models['Emtn_head'], lr=args.lr, betas=(0.5, 0.999), eps=1e-8, weight_decay=1e-2),
        optim.Adam(models['Warp_G'], lr=args.lr, betas=(0.5, 0.999), eps=1e-8, weight_decay=1e-2),
        optim.Adam(models['G3d'], lr=args.lr, betas=(0.5, 0.999), eps=1e-8, weight_decay=1e-2),
        optim.Adam(models['G2d'], lr=args.lr, betas=(0.5, 0.999), eps=1e-8, weight_decay=1e-2),
        optim.Adam(models['patchGAN'], lr=args.lr, betas=(0.5, 0.999), eps=1e-8, weight_decay=1e-2)
    ]

    schedulers = []

    for i in range(len(optimizers)):
        scheduler = CosineAnnealingLR(optimizers[0], T_max=args.iteration, eta_min=1e-6)
        schedulers.append(scheduler)

    train(args, args, models, device, driver_loader, source_loader, optimizers, schedulers, source_img_random, driver_img_random)


# Start Training.