import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
path_images = 'C:\\Users\\adels\PycharmProjects\data\OpticalFlowImages\\people\\people'
tau = 0.0039


def read_image(path, color=False):
    if color:
        image = np.array(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB), dtype=np.float64)
    else:
        image = np.array(cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY), dtype=np.float64)
    return image


def show_image(image, title=None, cmap=None):
    if cmap is not None:
        plt.imshow(np.uint8(image),cmap=cmap)
    else:
        plt.imshow(np.uint8(image))
    if title is not None:
        plt.title(title)

    # plt.axis('off')
    plt.show()


def calc_diff(img1, img2):
    kernel = np.array([[-1,8,0,-8,1]])/12
    Ix = cv2.filter2D(img1,-1,kernel)
    Iy = cv2.filter2D(img1,-1,kernel.T)
    It = img1 - img2
    return Ix, Iy, It


def prepare_gradients(Ix, Iy, It):
    d = np.array([[1, 4, 6, 4, 1]]) / 16
    w = np.power(np.dot(d.T, d), 2)
    wIx2 = cv2.filter2D(np.power(Ix, 2), -1, w).reshape(-1)
    wIy2 = cv2.filter2D(np.power(Iy, 2), -1, w).reshape(-1)
    wIxIy = cv2.filter2D(Ix * Iy, -1, w).reshape(-1)
    wIxIt = cv2.filter2D(Ix * It, -1, w).reshape(-1)
    wIyIt = cv2.filter2D(Iy * It, -1, w).reshape(-1)
    Ix = Ix.reshape(-1)
    Iy = Iy.reshape(-1)
    It = It.reshape(-1)
    return wIx2, wIy2, wIxIy, wIxIt, wIyIt, Ix, Iy, It


def calc_displacements_vec(wIx2, wIy2, wIxIy, wIxIt, wIyIt, Ix, Iy, It):
    v = np.zeros((2, len(wIx2)))
    for i in range(len(wIx2)):

        first_oper = (wIx2[i] + wIy2[i]) / 2
        second_oper = np.sqrt(4 * wIxIy[i] * wIxIy[i] + np.power((wIx2[i] - wIy2[i]), 2)) / 2
        lambda1 = first_oper + second_oper
        lambda2 = first_oper - second_oper
        if lambda1 >= tau and lambda2 >= tau:
            a = np.array([[wIx2[i], wIxIy[i]],
                          [wIxIy[i], wIy2[i]]])
            v[:, i:i + 1] = np.linalg.inv(a) @ np.array([[-wIxIt[i]], [-wIyIt[i]]])
        elif lambda1 >= tau and lambda2 < tau or lambda1 < tau and lambda2 >= tau:
            vec = np.array([Ix[i], Iy[i]])
            vec = -It[i] / np.linalg.norm(vec) * (vec / np.linalg.norm(vec))
            v[:, i] = vec
        else:
            v[:, i] = np.array([0, 0])
    return v


def vec2norm(vectors, img):
    orignal_img = img.shape
    img = img.reshape(-1)
    for i in range(vectors.shape[1]):
        img[i] = np.linalg.norm(vectors[:, i])
    return img.reshape(orignal_img)


if __name__ == "__main__":

    first = read_image(os.path.join(path_images, '1.tif'))
    # first = np.mean(first, axis=2)
    second = read_image(os.path.join(path_images, '2.tif'))
    # second = np.mean(second, axis=2)
    show_image(first,cmap='gray')
    show_image(second,cmap='gray')
    img_shape = first.shape


    glue_first = cv2.GaussianBlur(first,(3,3), 1.5)
    glue_second = cv2.GaussianBlur(second, (3, 3), 1.5)
    clear_img = np.zeros(img_shape)
    vectors_displacements = calc_displacements_vec(*prepare_gradients(*calc_diff(glue_first, glue_second)))
    img = vec2norm(vectors_displacements, clear_img)
    show_image(img,title=f'Thr = {tau}',cmap='gray')