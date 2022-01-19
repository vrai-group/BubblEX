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

from gradcam_exp.attgrad import ActivationsAndGradients
from sklearn.metrics import classification_report, confusion_matrix

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True,
                          save_path=None):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(20, 20))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)

def extract_cls(args):
    if True:
        if True:
            objs = np.load("data/objs.npy")
            labs = np.genfromtxt("data/GT.txt", delimiter=' ').astype("int64")

            test_loader= zip(objs,labs) ##
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

            #model = model.train()
            model = model.eval()
            target_layer = model.module.linear2 #linear2 #linear1 conv5
            if not args.no_cuda:
                model = model.cuda()

            activations_and_grads = ActivationsAndGradients(model, target_layer, None)

            print(model)
            print("Model defined...")
            i = 0
            ACTIVATIONS = []
            gts = []
            predicts = []
            for data, gt in test_loader:

                data= torch.tensor([data])

                if not args.no_cuda:
                    data = data.cuda()

                data = data.permute(0, 2, 1).to(device)
                output = activations_and_grads(data)

                am, idx = torch.max(output, 1)
                output = idx
                #output = torch.argmax(output.squeeze())

                #model.zero_grad()
                #loss = torch.mean(get_loss(output, target_category))
                #loss.backward(retain_graph=True)

                activations = activations_and_grads.activations[-1].cpu().data.numpy()
                #grads = self.activations_and_grads.gradients[-1].cpu().data.numpy()

                #np.save("classification\\act_conv5_{}.npy".format(i), activations)
                ACTIVATIONS.append(activations)
                #gts.append(gt[0].cpu().data.numpy().astype('int64'))
                gts.append(gt)
                predicts.append(output.cpu().data.numpy().astype('int64'))

                print("sample " + str(i) + " DONE")
                i+=1

                # if i==3:
                #   break

            ACTIVATIONS = np.concatenate(ACTIVATIONS)
            gts= np.array(gts)
            predicts= np.concatenate(predicts)

            numpy.savetxt("results/classification/gts.txt", gts, delimiter=" ",fmt='%d')
            numpy.savetxt("results/classification/predicts.txt", predicts, delimiter=" ",fmt='%d')
            numpy.savetxt("results/classification/ACT_linear2.txt", ACTIVATIONS, delimiter=" ",fmt='%.6f')


            print(classification_report(gts, predicts, target_names=CLASS_MAP))
            cm=confusion_matrix(gts, predicts)
            print(cm)
            plot_confusion_matrix(cm,CLASS_MAP,title='Confusion matrix',normalize=False,save_path="results/classification/cm.png")

CLASS_MAP = ['airplane','bathtub','bed','bench','bookshelf','bottle','bowl','car','chair','cone','cup','curtain','desk','door','dresser','flower_pot','glass_box','guitar','keyboard','lamp','laptop','mantel','monitor','night_stand','person','piano','plant','radio','range_hood','sink','sofa','stairs','stool','table','tent','toilet','tv_stand','vase','wardrobe','xbox']

class args(object):
    model_path= "models/model.cls.1024.t7"
    model= 'dgcnn_cls'
    k= 20
    emb_dims= 1024
    dropout= 0 #0.5
    no_cuda= True

extract_cls(args)
