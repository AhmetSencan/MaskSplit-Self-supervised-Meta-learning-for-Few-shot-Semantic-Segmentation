from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
import matplotlib
from typing import List
cmaps = ['winter', 'hsv', 'Wistia', 'BuGn']
import os


def make_episode_visualization(img_s: np.ndarray,
                               img_q: np.ndarray,
                               gt_s: np.ndarray,
                               gt_q: np.ndarray,
                               preds: np.ndarray,
                               save_path: str,
                               mean: List[float] = [0.485, 0.456, 0.406],
                               std: List[float] = [0.229, 0.224, 0.225]):

    # 0) Preliminary checks
    assert len(img_s.shape) == 4, f"Support shape expected : K x 3 x H x W or K x H x W x 3. Currently: {img_s.shape}"
    assert len(img_q.shape) == 3, f"Query shape expected : 3 x H x W or H x W x 3. Currently: {img_q.shape}"
    assert len(preds.shape) == 4, f"Predictions shape expected : T x num_classes x H x W. Currently: {preds.shape}"
    assert len(gt_s.shape) == 3, f"Support GT shape expected : K x H x W. Currently: {gt_s.shape}"
    assert len(gt_q.shape) == 2, f"Query GT shape expected : H x W. Currently: {gt_q.shape}"
    # assert img_s.shape[-1] == img_q.shape[-1] == 3, "Images need to be in the format H x W x 3"
    if img_s.shape[1] == 3:
        img_s = np.transpose(img_s, (0, 2, 3, 1))
    if img_q.shape[0] == 3:
        img_q = np.transpose(img_q, (1, 2, 0))

    assert img_s.shape[-3:-1] == img_q.shape[-3:-1] == gt_s.shape[-2:] == gt_q.shape

    if not os.path.exists("qualitative_results"):
        os.makedirs("qualitative_results")
    if not os.path.exists(os.path.join("qualitative_results", save_path)):
        os.makedirs(os.path.join("qualitative_results", save_path))

    if img_s.min() <= 0:
        img_s *= std
        img_s += mean

    if img_q.min() <= 0:
        img_q *= std
        img_q += mean

    img_s = img_s[0]

    mask_pred = preds.argmax(1)[0]

    gt_s[np.where(gt_s == 255)] = 0
    gt_q[np.where(gt_q == 255)] = 0

    make_plot(img_q, gt_q, "qualitative_results/" + save_path + "/" + save_path + "_qry.png", ['hsv'])
    make_plot(img_s, gt_s[0], "qualitative_results/" + save_path + "/" + save_path + "_sup.png", ['hsv'])
    make_plot(img_q, mask_pred, "qualitative_results/" + save_path + "/" + save_path + "_pred.png", ['hsv'])


def make_plot(img: np.ndarray,
              mask: np.ndarray,
              save_path: str,
              cmap_names):
    sizes = np.shape(img)
    fig = plt.figure()
    fig.set_size_inches(4. * sizes[0] / sizes[1], 4, forward=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    ax.imshow(img, interpolation='none')
    cmap = eval(f'plt.cm.{cmap_names[0]}')
    alphas = Normalize(0, .3, clip=True)(mask)
    alphas = np.clip(alphas, 0., 0.5)  # alpha value clipped at the bottom at .4
    colors = Normalize()(mask)
    colors = cmap(colors)
    colors[..., -1] = alphas
    ax.imshow(colors, cmap=cmap)  # interpolation='none'
    plt.savefig(save_path)
    plt.close()

