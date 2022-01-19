import os
import argparse

import numpy
import numpy as np
import torch
import torch.nn as nn
from data import S3DIS, ArCH, Sinthcity, ModelNet40, ShapeNetPart
from model import DGCNN_semseg, DGCNN_cls
from torch.utils.data import DataLoader
from gradcam_exp import gradcam

import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from copy import deepcopy

def extract_cls(args):

    ###
    if True:
        if True:
            objs = np.load("data/objs.npy")
            labs = np.genfromtxt("data/GT.txt", delimiter=' ').astype("int64")

            #selection
            #idx= np.array([430, 2048, 1925, 391, 600, 2320, 2017, 772, 274, 1949, 1132, 1229, 1457, 738, 220, 2179, 604, 2276, 2371, 896, 2013, 505, 896, 1432, 972, 852, 1858, 1672, 1675, 852, 1858, 1672])
            #idx = np.array([2452,563,1297,171,937,2288,128,1614,532,1523,1118,465,1948,1102,372,10,2378,765,394,1859,630,1924,928,460,677, 1197, 2319])
            idx = np.array([677])#677, 1197, 2319])
            #unique and sort
            idx= np.sort(np.unique(idx))
            objs= objs[idx]
            labs = labs[idx]

            test_loader = zip(objs, labs)  ##

            directory="plotNEW0\\"
            preds = np.genfromtxt("results/classification/predicts.txt", delimiter=' ').astype("int64")
            preds= preds[idx]

            #####
            if not args.no_cuda:
                device = torch.device("cuda")
            else:
                device = torch.device("cpu")
            
            # Try to load models
            if args.model == 'dgcnn_cls':
                model = DGCNN_cls(args).to(device)
            else:
                raise Exception("Not implemented")
            
            model = nn.DataParallel(model)
            
            print(os.path.join(args.model_path))
            if args.model_path == "":
                print(os.path.join(args.model_root, 'model_%s.t7' % test_area))
                model.load_state_dict(torch.load(os.path.join(args.model_root, 'model_%s.t7' % test_area)))
            else:
                if not args.no_cuda:
                    model.load_state_dict(torch.load(os.path.join(args.model_path)))
                else:
                    model.load_state_dict(torch.load(os.path.join(args.model_path),map_location = torch.device('cpu')))
            
            model = model.train()
            
            if not args.no_cuda:
                cam = gradcam.GradCAM(model=model,
                                      target_layer=model.module.conv5,
                                      use_cuda=True)
            else:
                cam = gradcam.GradCAM(model=model,
                                      target_layer=model.module.conv5,
                                      use_cuda=False)
            
            print("Model defined...")
            #####
            
            # i = 0
            # results = []
            # maxes = []
            # mines = []
            
            for cls in range(0, 40):
                i=0
                test_loader = zip(objs, labs)
                for data, max in test_loader:
            
                    data = torch.tensor([data])
            
                    data = data.permute(0, 2, 1).to(device)
                    # res, min_v, max_v, output = cam(input_tensor=data,
                    #           target_category=cls,
                    #           aug_smooth=False,
                    #           eigen_smooth=False)
                    # res_one = res[:, :].squeeze()
                    output, a, g = cam(input_tensor=data,
                                        target_category=cls,
                                        aug_smooth=False,
                                        eigen_smooth=False)
            
                    torch.set_printoptions(edgeitems=20, sci_mode=False)
                    numpy.set_printoptions(edgeitems=20, suppress=True)
            
                    out = torch.argmax(output.squeeze())
            
                    # results.append((data, res_one, max, out ))
                    # maxes.append(max_v)
                    # mines.append(min_v)
            
                    if cls==0:
                        np.save("results/actGradExtraction/act_conv5_{}.npy".format(idx[i]),a)
                    np.save("results/actGradExtraction/grad_conv5_{}_tg{}.npy".format(idx[i],cls), g)
            
                    print(i)
                    i += 1
                    #break
            
                print("class " + str(cls) + " DONE")

            # #plot grad
            # for cls in [0]: #range(0, 40):
            #     i=0
            #     test_loader = deepcopy(zip(objs, labs))
            #     for data, gt in test_loader:

            #         #g = np.load("C:\\Users\\andrea\\Desktop\\PCmax\\classification\\actGrad_\\act_conv5_{}.npy".format(idx[i]))
            #         g=np.load("C:\\Users\\andrea\\Desktop\\PCmax\\classification\\actGrad_\\grad_conv5_{}_tg{}.npy".format(idx[i], cls))

            #         # histogram of features across point
            #         # histogram of point across features
            #         H =[]
            #         B= []
            #         GM=[]
            #         Gm=[]
            #         M=[]
            #         m=[]
            #         for i in range(g.shape[2]):
            #             h, b = np.histogram(g[0,:,i], bins=100)
            #             H.append(h)
            #             B.append(b)
            #             GM.append(np.median(g[0,:,i]))
            #             Gm.append(np.mean(g[0,:,i]))
            #             M.append(np.max(g[0,:,i]))
            #             m.append(np.min(g[0,:,i]))

            #         # m= np.min(g, axis=1)
            #         # M= np.max(g, axis=1)
            #         # Gm= np.mean(g, axis=1)
            #         # GM= np.median(g, axis=1)

            #         plt.figure()
            #         plt.plot(m)
            #         plt.plot(M)
            #         plt.plot(Gm)
            #         plt.plot(GM)

            #         plt.figure()
            #         h, b = np.histogram(m, bins=100)
            #         plt.stem(b[1:], h)

            #         plt.figure()
            #         h, b = np.histogram(M, bins=100)
            #         plt.stem(b[1:], h)

            #         plt.figure()
            #         h, b = np.histogram(Gm, bins=100)
            #         plt.stem(b[1:], h)

            #         plt.figure()
            #         h, b = np.histogram(GM, bins=100)
            #         plt.stem(b[1:], h)

            #         gM= np.median(g,axis=1)
            #         #agM = np.mean(ag, axis=1)
            #         var = gM[0]

            #         min_v = np.min(var)
            #         max_v = np.max(var)
            #         gt = labs[i]
            #         pred = preds[i]

            #         data[:, [1, 2]] = data[:, [2, 1]]

            #         #varst = (var - min_v) / (max_v - min_v)  # +0.000001)

            #         # simmetrizzazione
            #         abs_max_v = max(abs(min_v), abs(max_v))
            #         min_v = -abs_max_v
            #         max_v = abs_max_v
            #         varst = (var - min_v) / (max_v - min_v)  # +0.000001)

            #         ply = data  # numpy.stack((data, axis=-1)
            #         pcd = o3d.geometry.PointCloud()

            #         cmap = plt.cm.get_cmap("jet")
            #         varst = cmap(varst)[:, :3]

            #         pcd.points = o3d.utility.Vector3dVector(ply)
            #         pcd.colors = o3d.utility.Vector3dVector(varst)

            #         o3d.io.write_point_cloud(directory + "g_MED6_{}_tg{}_gt{}_p{}.ply".format(idx[i], cls, gt, pred), pcd)

            #         print(i)
            #         i += 1

            # #plot gradcam
            # for cls in range(0, 40):
            #     i=0
            #     test_loader = deepcopy(zip(objs, labs))
            #     for data, gt in test_loader:
            #
            #         a=np.load("C:\\Users\\andrea\\Desktop\\PCmax\\classification\\actGrad_\\act_conv5_{}.npy".format(idx[i]))
            #         g=np.load("C:\\Users\\andrea\\Desktop\\PCmax\\classification\\actGrad_\\grad_conv5_{}_tg{}.npy".format(idx[i], cls))
            #
            #         ag= a*g
            #         agM= np.median(ag,axis=1)
            #
            #         # aM= np.median(a,axis=1)
            #         # gM = np.median(g, axis=1)
            #         # agM = aM * gM
            #
            #         # gM = np.median(g, axis=1)
            #         # ag = a * gM[:,np.newaxis,:]
            #         # agM = np.median(ag, axis=1)
            #
            #         # aM = np.median(a, axis=1)
            #         # ag = g * aM[:,np.newaxis,:]
            #         # agM = np.median(ag, axis=1)
            #
            #         var = agM[0]
            #
            #         min_v = np.min(var)
            #         max_v = np.max(var)
            #         gt = labs[i]
            #         pred = preds[i]
            #
            #         data[:, [1, 2]] = data[:, [2, 1]]
            #
            #         # varst = (var - min_v) / (max_v - min_v)  # +0.000001)
            #
            #         # simmetrizzazione
            #         abs_max_v = max(abs(min_v), abs(max_v))
            #         min_v = -abs_max_v
            #         max_v = abs_max_v
            #         varst = (var - min_v) / (max_v - min_v)  # +0.000001)
            #
            #         ply = data  # numpy.stack((data, axis=-1)
            #         pcd = o3d.geometry.PointCloud()
            #
            #         cmap = plt.cm.get_cmap("jet")
            #         varst = cmap(varst)[:, :3]
            #
            #         pcd.points = o3d.utility.Vector3dVector(ply)
            #         pcd.colors = o3d.utility.Vector3dVector(varst)
            #
            #         o3d.io.write_point_cloud(directory + "ag_mean6_{}_tg{}_gt{}_p{}.ply".format(idx[i], cls, gt, pred), pcd)
            #
            #         print(i)
            #         i += 1

            # #plot only activation
            # i=0
            # for data, gt in test_loader:
            #     a = np.load("C:\\Users\\andrea\\Desktop\\PCmax\\classification\\actGrad_\\act_conv5_{}.npy".format(idx[i]))
            #
            #     aM = np.min(a, axis=1)
            #     var = aM[0]
            #
            #     min_v = np.min(var)
            #     max_v = np.max(var)
            #     gt = labs[i]
            #     pred = preds[i]
            #
            #     data[:, [1, 2]] = data[:, [2, 1]]
            #
            #     # varst = (var - min_v) / (max_v - min_v)  # +0.000001)
            #
            #     # simmetrizzazione
            #     abs_max_v= max(abs(min_v),abs(max_v))
            #     min_v= -abs_max_v
            #     max_v = abs_max_v
            #     varst = (var - min_v) / (max_v - min_v)  # +0.000001)
            #
            #
            #     ply = data  # numpy.stack((data, axis=-1)
            #     pcd = o3d.geometry.PointCloud()
            #
            #     cmap = plt.cm.get_cmap("jet")
            #     varst = cmap(varst)[:, :3]
            #
            #     pcd.points = o3d.utility.Vector3dVector(ply)
            #     pcd.colors = o3d.utility.Vector3dVector(varst)
            #
            #     o3d.io.write_point_cloud(directory + "a_min_{}_gt{}_p{}.ply".format(idx[i], gt, pred), pcd)
            #
            #     print(i)
            #     i += 1


class args(object):
    model_path= "models/model.cls.1024.t7"
    model= 'dgcnn_cls'
    k= 20
    emb_dims= 1024
    dropout= 0 #0.5
    no_cuda= True

extract_cls(args)