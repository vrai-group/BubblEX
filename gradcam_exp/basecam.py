import cv2
import numpy as np
import torch
import ttach as tta
from .attgrad import ActivationsAndGradients


def get_2d_projection(activation_batch):
    # TBD: use pytorch batch svd implementation
    projections = []
    for activations in activation_batch:
        reshaped_activations = (activations).reshape(activations.shape[0], -1).transpose()
        # Centering before the SVD seems to be important here,
        # Otherwise the image returned is negative
        reshaped_activations = reshaped_activations - reshaped_activations.mean(axis=0)
        U, S, VT = np.linalg.svd(reshaped_activations, full_matrices=True)
        projection = reshaped_activations @ VT[0, :]
        projection = projection.reshape(activations.shape[1:])
        projections.append(projection)
    return np.float32(projections)


class BaseCAM:
    def __init__(self,
                 model,
                 target_layer,
                 use_cuda=False,
                 reshape_transform=None):
        self.model = model.eval()
        self.target_layer = target_layer
        self.cuda = use_cuda

        if self.cuda:
            self.model = model.cuda()

        self.reshape_transform = reshape_transform
        self.activations_and_grads = ActivationsAndGradients(self.model,
                                                             target_layer, reshape_transform)

    def forward(self, input_img):
        return self.model(input_img)

    def get_cam_weights(self,
                        input_tensor,
                        target_category,
                        activations,
                        grads):
        raise Exception("Not Implemented")

    def get_loss(self, output, target_category):
        loss = 0
        for i in range(len(target_category)):
            loss = loss + output[i, target_category[i]]
        return loss

    def get_cam_image(self,
                      input_tensor,
                      target_category,
                      activations,
                      grads,
                      eigen_smooth=False):
        weights = self.get_cam_weights(input_tensor, target_category, activations, grads)
        if len(weights.shape) == 3:
            weighted_activations = weights[:, None, None] * activations
        else:
            weighted_activations = weights[:, None] * activations
        if eigen_smooth:
            cam = get_2d_projection(weighted_activations)
        else:
            cam = weighted_activations.sum(axis=1)
        return cam

    def forward(self, input_tensor, target_category=None, eigen_smooth=False):

        if self.cuda:
            input_tensor = input_tensor.cuda()
        output = self.activations_and_grads(input_tensor)
        # am, idx = torch.max(output, 1)
        # output = idx
        if len(output.size()) == 3:
            output = output[:, :, :]
        else:
            output = output[:, :]

        if type(target_category) is int:
            target_category = [target_category] * input_tensor.size(0)

        if target_category is None:
            target_category = np.argmax(output.cpu().data.numpy(), axis=-1)
        else:
            assert (len(target_category) == input_tensor.size(0))

        # print("Target category:", target_category)

        self.model.zero_grad()
        loss = torch.mean(self.get_loss(output, target_category))
        loss.backward(retain_graph=True)

        activations = self.activations_and_grads.activations[-1].cpu().data.numpy()
        grads = self.activations_and_grads.gradients[-1].cpu().data.numpy()

        # cam = self.get_cam_image(input_tensor, target_category,
        #                          activations, grads, eigen_smooth)
        #
        # cam = np.maximum(cam, 0)
        #
        # result = []
        # min_v = np.min(cam)
        # max_v = np.max(cam)
        j=0
        # acts_p= []
        # acts_n = []
        # gs_p=[]
        # gs_n = []
        # gs_median= []
        # gs_meansq = []

        # activations_p= (activations * (activations > 0)).sum(axis=1)
        # activations_n= (activations * (activations < 0)).sum(axis=1)
        # grads_p= (grads*(grads > 0)).sum(axis=1)
        # grads_n= (grads*(grads < 0)).sum(axis=1)

        ag= activations * grads
        ag_p = (ag * (ag > 0)).sum(axis=1)
        ag_n= (ag * (ag < 0)).sum(axis=1)
        ag_max = np.max(ag, axis=1)
        ag_min = np.min(ag, axis=1)
        ag_med = np.median(ag, axis=1)
        ag_ms = np.sqrt((np.square(ag)).sum(axis=1)/1024)

        # for agi in ag:
        # #for img in cam:
        #     # act_p= activations_p[j]
        #     # act_n = activations_n[j]
        #     # g_p = grads_p[j]
        #     # g_n = grads_n[j]
        #     # g_median = np.median(grads,axis=1)[j]
        #     # g_meansq = np.sqrt((np.square(grads)).sum(axis=1)/1024)[j]
        #
        #
        #
        #     # img = cv2.resize(img, input_tensor.shape[-2:][::-1])
        #     # img = img - min_v
        #     # if max_v != 0:
        #     #     img = img / max_v
        #     # result.append(img)
        #
        #     # acts_p.append(act_p)
        #     # acts_n.append(act_n)
        #     # gs_p.append(g_p)
        #     # gs_n.append(g_n)
        #     # gs_median.append(g_median)
        #     # gs_meansq.append(g_meansq)
        #
        # # acts_p.append(act_p)
        # # acts_n.append(act_n)
        # # gs_p.append(g_p)
        # # gs_n.append(g_n)
        # # gs_median.append(g_median)
        # # gs_meansq.append(g_meansq)
        #
        #     j+=1
        #result = np.float32(result), min_v, max_v, output
        #result = np.float32(result), min_v, max_v, output, np.float32(gs_median), np.float32(gs_meansq) #, np.float32(acts_p), np.float32(acts_n),np.float32(gs_p),np.float32(gs_n)
        #result = output, np.float32(ag_p), np.float32(ag_n), np.float32(ag_max), np.float32(ag_min), np.float32(ag_med), np.float32(ag_ms)
        result = output, np.float32(activations), np.float32(grads)

        return result

    def forward_augmentation_smoothing(self,
                                       input_tensor,
                                       target_category=None,
                                       eigen_smooth=False):
        transforms = tta.Compose(
            [
                tta.HorizontalFlip(),
                tta.Multiply(factors=[0.9, 1, 1.1]),
            ]
        )
        cams = []
        for transform in transforms:
            augmented_tensor = transform.augment_image(input_tensor)
            cam = self.forward(augmented_tensor,
                               target_category, eigen_smooth)

            # The ttach library expects a tensor of size BxCxHxW
            cam = cam[:, None, :, :]
            cam = torch.from_numpy(cam)
            cam = transform.deaugment_mask(cam)

            # Back to numpy float32, HxW
            cam = cam.numpy()
            cam = cam[:, 0, :, :]
            cams.append(cam)

        cam = np.mean(np.float32(cams), axis=0)
        return cam

    def __call__(self,
                 input_tensor,
                 target_category=None,
                 aug_smooth=False,
                 eigen_smooth=False):
        if aug_smooth is True:
            return self.forward_augmentation_smoothing(input_tensor,
                                                       target_category, eigen_smooth)

        return self.forward(input_tensor,
                            target_category, eigen_smooth)
