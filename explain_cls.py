import numpy as np
import pickle
import os

import torch
import random
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from datetime import datetime

import argparse

import paramSettings


########################## INPUT #######################
'''
layers = [1, 2, 3, 4, 5]   # lista in cui inserire i layer da utilizzare: da 1 a 5

dataset = "s3dis"   # arch | synthcity | s3dis
base_path = "checkpoints/semseg_eval_6/"

tipo = "tsne"   #   tsne | umap
balanced = True    # True | False --> True significa che bilancio le classi e quindi userÃƒÂ² molti meno punti
'''

########################################################


def explain(layer, dataset, base_path, tipo, balanced, tipo_label, use_pca):
    feat_path = base_path + "classification/ACT_linear{}.txt".format(layer)
    if tipo_label=="gt":
        gt_path = base_path + "classification/gts.txt"
    elif tipo_label=="pred":
        gt_path = base_path + "classification/predicts.txt"
    save_path = base_path + "explain/"
    os.makedirs(save_path, exist_ok=True)
    save_path = save_path + tipo_label+"/"
    os.makedirs(save_path, exist_ok=True)

    map_feats_layer=[512,256,15] #[64,64,64,1024,256]
    N_FEATS = map_feats_layer[layer-1]

    if dataset == "synthcity":
        colors_per_class = {
            0: [254, 202, 87],
            1: [255, 107, 107],
            2: [10, 189, 227],
            3: [255, 159, 243],
            4: [16, 172, 132],
            5: [128, 80, 128],
            6: [87, 101, 116],
            7: [52, 31, 151],
            8: [0, 0, 0]
        }

        CLASS_MAP = ["building", "car", "natural-ground", "ground", "pole-like", "road", "street-furniture", "tree", "pavement"]

    elif dataset == "arch":
        colors_per_class = {
            0: [254, 202, 87],
            1: [255, 107, 107],
            2: [10, 189, 227],
            3: [255, 159, 243],
            4: [16, 172, 132],
            5: [128, 80, 128],
            6: [87, 101, 116],
            7: [52, 31, 151],
            8: [0, 0, 0],
            9: [100, 100, 255],
        }

        CLASS_MAP = ["arc", "column", "moulding", "floor", "door-window", "wall", "stairs", "vault", "roof", "other"]

    elif dataset == "arch9l":
        colors_per_class = {
            0: [254, 202, 87],
            1: [255, 107, 107],
            2: [10, 189, 227],
            3: [255, 159, 243],
            4: [16, 172, 132],
            5: [128, 80, 128],
            6: [87, 101, 116],
            7: [52, 31, 151],
            8: [0, 0, 0]
        }

        CLASS_MAP = ["arc", "column", "moulding", "floor", "door-window", "wall", "stairs", "vault", "roof"]

    elif dataset == "s3dis":
        colors_per_class = {
            0: [254, 202, 87],
            1: [255, 107, 107],
            2: [10, 189, 227],
            3: [255, 159, 243],
            4: [16, 172, 132],
            5: [128, 80, 128],
            6: [87, 101, 116],
            7: [52, 31, 151],
            8: [0, 0, 0],
            9: [100, 100, 255],
            10: [100, 0, 255],
            11: [0, 100, 255],
            12: [100, 100, 0],
        }

        CLASS_MAP = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door', 'table', 'chair', 'sofa', 'bookcase', 'board', 'clutter']

    elif dataset == "modelnet40":
        colors_per_class = {
            0: [0, 0, 0],
            1: [0, 0, 170],
            2: [0, 0, 255],
            3: [0, 85, 85],
            4: [0, 85, 170],
            5: [0, 170, 0],
            6: [0, 170, 85],
            7: [0, 170, 255],
            8: [0, 255, 85],
            9: [0, 255, 170],
            10: [85, 0, 0],
            11: [85, 0, 85],
            12: [85, 0, 255],
            13: [85, 85, 0],
            14: [85, 85, 170],
            15: [85, 170, 0],
            16: [85, 170, 85],
            17: [85, 170, 255],
            18: [85, 255, 0],
            19: [85, 255, 170],
            20: [170, 0, 0],
            21: [170, 0, 85],
            22: [170, 0, 255],
            23: [170, 85, 0],
            24: [170, 85, 170],
            25: [170, 85, 255],
            26: [170, 170, 85],
            27: [170, 170, 255],
            28: [170, 255, 0],
            29: [170, 255, 170],
            30: [170, 255, 255],
            31: [255, 0, 85],
            32: [255, 0, 170],
            33: [255, 85, 0],
            34: [255, 85, 170],
            35: [255, 85, 255],
            36: [255, 170, 85],
            37: [255, 170, 170],
            38: [255, 255, 0],
            39: [255, 255, 85]
        }

        CLASS_MAP = ['airplane','bathtub','bed','bench','bookshelf','bottle','bowl','car','chair','cone','cup','curtain','desk','door','dresser','flower_pot','glass_box','guitar','keyboard','lamp','laptop','mantel','monitor','night_stand','person','piano','plant','radio','range_hood','sink','sofa','stairs','stool','table','tent','toilet','tv_stand','vase','wardrobe','xbox']
        
    elif dataset == "scanObjectNN":
        colors_per_class = {
            0: [0, 0, 0],
            1: [0, 0, 255],
            2: [0, 170, 85],
            3: [0, 255, 85],
            4: [85, 0, 255],
            5: [85, 85, 170],
            6: [85, 170, 85],
            7: [85, 255, 0],
            8: [170, 0, 255],
            9: [170, 85, 170],
            10: [170, 170, 85],
            11: [170, 255, 255],
            12: [255, 0, 170],
            13: [255, 170, 85],
            14: [255, 255, 0],
        }

        CLASS_MAP = ['bag','bin','box','cabinet','chair','desk','display','dor','shelf','table','bed','pillow','sink','sofa','toilet']

    else:
        input("Errore dataset: {}".format(dataset))

    print("\n\nLayer {} with {} features".format(layer, N_FEATS))

    if balanced:
        save_path = save_path + "balanced/"
        os.makedirs(save_path, exist_ok=True)
    else:
        save_path = save_path + "all/"
        os.makedirs(save_path, exist_ok=True)

    if os.path.exists(save_path+"feats_{}.pickle".format(layer)):
        print("Carico i file esistenti...")
        with open(save_path+"feats_{}.pickle".format(layer), "rb") as f:
            feats, preds = pickle.load(f)
    else:
        if balanced:
            preds = []
            cont=0
            dizio={}
            print("Loading GT...")
            with open(gt_path, "r") as fr:
                for l in fr:
                    x, y, z, r, g, b, gt, pred = l.strip().split()
                    if tipo_label=="gt":
                        ipred = int(gt)
                    else:
                        ipred = int(pred)
                    preds.append(ipred)

                    if ipred not in dizio:
                        dizio[ipred] = []
                    dizio[ipred].append(cont)

                    cont+=1
                    if cont%100000==0:    print(cont)
            print(cont)

            print("Prendo un sottoinsieme dei punti, bilanciando le classi:")
            lens = []
            for k in sorted(dizio.keys()):
                lens.append(len(dizio[k]))
                print(" -",k,CLASS_MAP[k],len(dizio[k]))
            minimo = min(lens)
            print(" - minimo:{}".format(minimo))

            feats = np.zeros((cont,N_FEATS),dtype="float32")
            i=0
            print("Carico tutte le FEATS...")
            with open(feat_path, "r") as fr:
                for l in fr:
                    ff = l.strip().split()
                    for j, fff in enumerate(ff):
                        feats[i,j] = float(fff)
                    i += 1
                    if i % 100000 == 0:    print(i)
            print(i)

            #PRENDO GLI INDICI
            if os.path.exists(save_path + "indici.pickle"):
                print("Carico gli indici esistenti...")
                with open(save_path + "indici.pickle", "rb") as f:
                    dizio2 = pickle.load(f)
            else:
                dizio2={}
                #for i in range(len(dizio.keys())):
                for i in dizio:
                    dizio2[i] = random.sample(dizio[i], minimo)

                print("Salvo gli indici...")
                with open(save_path + "indici.pickle", "wb") as f:
                    pickle.dump(dizio2, f)


            preds2 = []
            #PRENDO LE FEATURES USANDO GLI INDICI
            feats2 = np.zeros((minimo * len(dizio.keys()), N_FEATS),dtype="float32")
            k=0
            print("Estraggo le FEATS BILANCIATE...")
            #for i in range(len(dizio.keys())):
            for i in dizio:
                ind = dizio2[i]
                for j in ind:
                    feats2[k] = feats[j]
                    k+=1

                preds2 += ([i] * minimo)    # ricreo le labels

            feats = feats2
            preds = preds2

            #print("Salvo i file...")
            #with open(save_path+"feats_{}.pickle".format(layer), "wb") as f:
            #    pickle.dump([feats, preds], f)

        else:
            preds = []
            cont = 0
            dizio = {}
            print("Loading GT...")
            with open(gt_path, "r") as fr:
                for l in fr:
                    # x, y, z, r, g, b, gt, pred = l.strip().split()
                    # if tipo_label == "gt":
                    #     ipred = int(gt)
                    # else:
                    #     ipred = int(pred)
                    # preds.append(ipred)

                    [gt] = l.strip().split()
                    ipred = int(gt)
                    preds.append(ipred)

                    if ipred not in dizio:
                        dizio[ipred] = []
                    dizio[ipred].append(cont)

                    cont += 1
                    if cont % 100000 == 0:    print(cont)
            print(cont)


            feats = np.zeros((cont,N_FEATS),dtype="float32")

            i=0
            print("Loading FEATS...")
            with open(feat_path, "r") as fr:
                for l in fr:
                    ff = l.strip().split()
                    for j, fff in enumerate(ff):
                        feats[i,j] = float(fff)
                    i += 1
                    if i % 100000 == 0:    print(i)
            print(i)

        print("Salvo i file...")
        with open(save_path + "feats_{}.pickle".format(layer), "wb") as f:
            pickle.dump([feats, preds], f, protocol=4)

    print("FEATS: {}".format(feats.shape))
    print("LABELS: {}".format(len(preds)))

    if use_pca:
        save_path = save_path + "pca/"
        os.makedirs(save_path, exist_ok=True)

        if os.path.exists(save_path + "feats_{}.pickle".format(layer)):
            print("Carico i file PCA esistenti...")
            with open(save_path + "feats_{}.pickle".format(layer), "rb") as f:
                feats, preds = pickle.load(f)
        else:

            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print("Utilizzo la PCA...", current_time)

            from sklearn.decomposition import PCA
            pca = PCA(n_components=5)
            feats = pca.fit_transform(feats)

            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print("FEATS_PCA: {} - {}".format(feats.shape,current_time))
            print("Salvo i file...")
            with open(save_path + "feats_{}.pickle".format(layer), "wb") as f:
                pickle.dump([feats, preds], f)

    ########################### t-SNE ed UMAP #############################################################

    def fix_random_seeds():
        seed = 10
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)

    def visualize_tsne2(tsne, labels, dataset, save_path):
        # extract x and y coordinates representing the positions of the images on T-SNE plot
        tx = tsne[:, 0]
        ty = tsne[:, 1]

        # scale and move the coordinates so they fit [0; 1] range
        tx = scale_to_01_range(tx)
        ty = scale_to_01_range(ty)

        # visualize the plot: samples as colored points
        visualize_tsne_points_tot(tx, ty, labels, save_path+"_tot.png")
        visualize_tsne_points_sing(tx, ty, labels, save_path)
        visualize_tsne_points_sing_tot(tx, ty, labels, dataset, save_path+"_sing-tot.png")

    # scale and move the coordinates so they fit [0; 1] range
    def scale_to_01_range(x):
        # compute the distribution range
        value_range = (np.max(x) - np.min(x))

        # move the distribution so that it starts from zero
        # by extracting the minimal value from all its values
        starts_from_zero = x - np.min(x)

        # make the distribution fit [0; 1] by dividing by its range
        return starts_from_zero / value_range

    def visualize_tsne_points_tot(tx, ty, labels, save_path = ""):
        # initialize matplotlib plot
        #fig = plt.figure()
        fig = plt.figure(figsize=(8,5))

        ax = fig.add_subplot(111)
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.0])

        # for every class, we'll add a scatter plot separately
        for label in colors_per_class:
            # find the samples of the current class in the data
            indices = [i for i, l in enumerate(labels) if l == label]

            # extract the coordinates of the points of this class only
            current_tx = np.take(tx, indices)
            current_ty = np.take(ty, indices)

            # convert the class color to matplotlib format:
            # BGR -> RGB, divide by 255, convert to np.array
            color = np.array([colors_per_class[label][::-1]], dtype=np.float) / 255

            # add a scatter plot with the correponding color and label
            ax.scatter(current_tx, current_ty, c=color, label=CLASS_MAP[label], s=3)

        # build a legend using the labels we set previously
        #ax.legend(loc='lower right')

        # Shrink current axis by 20%
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        # Put a legend to the right of the current axis
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        # finally, show the plot
        if save_path=="":
            plt.show()
        else:
            plt.savefig(save_path)

    def visualize_tsne_points_sing(tx, ty, labels, save_path = ""):

        # for every class, we'll add a scatter plot separately
        for label in colors_per_class:
            # initialize matplotlib plot
            fig = plt.figure(figsize=(8, 5))
            ax = fig.add_subplot(111)
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.0])

            # find the samples of the current class in the data
            indices = [i for i, l in enumerate(labels) if l == label]

            # extract the coordinates of the points of this class only
            current_tx = np.take(tx, indices)
            current_ty = np.take(ty, indices)

            # convert the class color to matplotlib format:
            # BGR -> RGB, divide by 255, convert to np.array
            color = np.array([colors_per_class[label][::-1]], dtype=np.float) / 255

            # add a scatter plot with the correponding color and label
            ax.scatter(current_tx, current_ty, c=color, label=CLASS_MAP[label], s=3)

            # Shrink current axis by 20%
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

            # Put a legend to the right of the current axis
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

            # finally, show the plot
            if save_path=="":
                plt.show()
            else:
                plt.savefig(save_path+"_l{}.png".format(label))


    def visualize_tsne_points_sing_tot(tx, ty, labels, dataset, save_path=""):
        if dataset=="s3dis": # 13 classi
            rig=2
            col=8
        elif dataset=="arch":   # 10 classi
            rig = 2
            col = 6
        elif dataset=="arch9l": # 9 classi
            rig = 2
            col = 6
        elif dataset=="synthcity": # 9 classi
            rig = 2
            col = 6
        elif dataset=="modelnet40": # 9 classi
            rig = 5
            col = 8
        elif dataset=="scanObjectNN": # 9 classi
            rig = 5
            col = 3

        fig, axs = plt.subplots(rig, col)

        axis = []
        for j in range(col):
            for i in range(rig):
                axs[i, j].set_xlim([0.0, 1.0])
                axs[i, j].set_ylim([0.0, 1.0])
                axis.append(axs[i, j])
        fig.set_size_inches(18, 5)

        # for every class, we'll add a scatter plot separately
        for label in colors_per_class:
            # find the samples of the current class in the data
            indices = [i for i, l in enumerate(labels) if l == label]

            # extract the coordinates of the points of this class only
            current_tx = np.take(tx, indices)
            current_ty = np.take(ty, indices)

            # convert the class color to matplotlib format:
            # BGR -> RGB, divide by 255, convert to np.array
            color = np.array([colors_per_class[label][::-1]], dtype=np.float) / 255

            # add a scatter plot with the correponding color and label
            # ax.scatter(current_tx, current_ty, c=color, label=CLASS_MAP[label])
            axis[label].scatter(current_tx, current_ty, c=color, label=CLASS_MAP[label], s=3)

        for k in range(len(CLASS_MAP),len(axis)):
            axis[k].set_visible(False)

        # Put a legend to the right of the current axis
        fig.legend(loc='center right')

        fig.tight_layout()

        # finally, show the plot
        if save_path == "":
            plt.show()
        else:
            plt.savefig(save_path)



    features = feats
    labels = preds

    if tipo == "tsne":
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("t-SNE:",current_time)

        if os.path.exists(save_path+"tsne_{}.pickle".format(layer)):
            print("Carico i file t-sne esistenti...")
            with open(save_path+"tsne_{}.pickle".format(layer), "rb") as f:
                tsne = pickle.load(f)
        else:

            fix_random_seeds()

            print(" - compute...")
            tsne = TSNE(n_components=2).fit_transform(features)

            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print(" - save...", current_time)
            with open(save_path + "tsne_{}.pickle".format(layer), "wb") as f:
                pickle.dump(tsne, f)

        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print(" - visualize...",current_time)
        visualize_tsne2(tsne, labels, dataset, save_path+"tsne_{}".format(layer))


    elif tipo == "umap":
        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print("UMAP:",current_time)

        if os.path.exists(save_path+"umap_{}.pickle".format(layer)):
            print("Carico i file umap esistenti...")
            with open(save_path+"umap_{}.pickle".format(layer), "rb") as f:
                embedding = pickle.load(f)
        else:

            import umap
            #import umap.plot

            features = feats
            print(" - compute...")
            mapper = umap.UMAP().fit(features)
            embedding = mapper.transform(features)

            # Verify that the result of calling transform is
            # idenitical to accessing the embedding_ attribute
            assert (np.all(embedding == mapper.embedding_))

            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print(" - save...", current_time)
            with open(save_path + "umap_{}.pickle".format(layer), "wb") as f:
                pickle.dump(embedding, f)

        now = datetime.now()
        current_time = now.strftime("%H:%M:%S")
        print(" - visualize...",current_time)
        visualize_tsne_points_tot(scale_to_01_range(embedding[:, 0]), scale_to_01_range(embedding[:, 1]), labels, save_path+"umap_{}_tot.png".format(layer))
        visualize_tsne_points_sing(scale_to_01_range(embedding[:, 0]), scale_to_01_range(embedding[:, 1]), labels, save_path+"umap_{}".format(layer))
        visualize_tsne_points_sing_tot(scale_to_01_range(embedding[:, 0]), scale_to_01_range(embedding[:, 1]), labels, dataset, save_path + "umap_{}_sing-tot.png".format(layer))


    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Done!!",current_time)



if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud XAI')
    parser.add_argument('--dataset', type=str, default='arch', help='Dataset name',
                        choices=["s3dis", "arch", "arch9l", "synthcity"])
    parser.add_argument('--layers', type=str, default='1,2,3,4,5', help='layers 1,2,3,4,5')
    parser.add_argument('--base_path', type=str, default='', help='Path for the inputs and outputs')
    parser.add_argument('--tipo', type=str, default='tsne', help='Type of the XAI method (tnse,umap)',
                        choices=["tsne", "umap", "both"])
    parser.add_argument('--label', type=str, default='gt', help='Type of label (gt,pred)',
                        choices=["gt", "pred", "both"])
    parser.add_argument('--balanced', type=bool, default=False, help='Choice to balance data')
    parser.add_argument('--use_pca', type=bool, default=False, help='Choice to balance data')

    args = parser.parse_args()

    args.base_path = "results/" ###
    args.dataset = "scanObjectNN" # "modelnet40" ###

    if not os.path.exists(args.base_path):
        print("Errore: {} NON ESISTE".format(args.base_path))
    else:
        layers = args.layers.split(",")
        layers = [int(l) for l in layers]
        layers= [2] ### 2,3

        if args.tipo=="both":         xai = ["tsne","umap"]
        else:                         xai = [args.tipo]
        xai= ["tsne"] ###

        if args.label=="both":        labels = ["gt","pred"]
        else:                         labels = [args.label]
        labels= ["gt","pred"] ###

        for tipo_tecnica in xai:
            for tipo_label in labels:
                for layer in layers:
                    explain(layer, args.dataset, args.base_path, tipo_tecnica, args.balanced, tipo_label, args.use_pca)