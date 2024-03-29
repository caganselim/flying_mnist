from PIL import Image
import os
import torch
import numpy as np
import matplotlib.pyplot as plt


import numpy
#adapted from http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy
# WARNING: this will work on little-endian architectures (eg Intel x86) only!
def read_flo_file(fp):
  with open(fp, 'rb') as f:
    magic = numpy.fromfile(f, numpy.float32, count=1)
    if 202021.25 != magic:
      print('Magic number incorrect. Invalid .flo file', fp)
    else:
      w = numpy.fromfile(f, numpy.int32, count=1)[0]
      h = numpy.fromfile(f, numpy.int32, count=1)[0]
      data = numpy.fromfile(f, numpy.float32, count=2 * w * h)
      data2D = data.reshape((h, w, 2))
      return data2D

def mesh_grid(B, H, W):
    # mesh grid
    x_base = torch.arange(0, W, dtype=torch.float32).repeat(B, H, 1)  # BHW
    y_base = torch.arange(0, H, dtype=torch.float32).repeat(B, W, 1).transpose(1, 2)  # BHW

    base_grid = torch.stack([x_base, y_base], 1)  # B2HW
    return base_grid

def norm_grid(v_grid):

    _, _, H, W = v_grid.size()

    # scale grid to [-1,1]
    v_grid_norm = torch.zeros_like(v_grid)
    v_grid_norm[:, 0, :, :] = 2.0 * v_grid[:, 0, :, :] / float(W - 1) - 1.0
    v_grid_norm[:, 1, :, :] = 2.0 * v_grid[:, 1, :, :] / float(H - 1) - 1.0
    return v_grid_norm.permute(0, 2, 3, 1)  # BHW2


def get_colorwheel_img(size = 473):

    base_grid = mesh_grid(1, size, size)  # B2HW
    v_grid = norm_grid(base_grid) # BHW2
    img = flow_to_img(v_grid[0].numpy())

    return img

def make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf
    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.
    Returns:
        np.ndarray: Color wheel
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0,RY)/RY)
    col = col+RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0,YG)/YG)
    colorwheel[col:col+YG, 1] = 255
    col = col+YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0,GC)/GC)
    col = col+GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(CB)/CB)
    colorwheel[col:col+CB, 2] = 255
    col = col+CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0,BM)/BM)
    col = col+BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(MR)/MR)
    colorwheel[col:col+MR, 0] = 255
    return colorwheel


def flow_uv_to_colors(u, v, convert_to_bgr=False):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.
    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun
    Args:
        u (np.ndarray): Input horizontal flow of shape [H,W]
        v (np.ndarray): Input vertical flow of shape [H,W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.
    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u)/np.pi
    fk = (a+1) / 2*(ncols-1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:,i]
        try:
            col0 = tmp[k0] / 255.0
            col1 = tmp[k1] / 255.0
            col = (1-f)*col0 + f*col1
            idx = (rad <= 1)
            col[idx]  = 1 - rad[idx] * (1-col[idx])
            col[~idx] = col[~idx] * 0.75   # out of range
            # Note the 2-i => BGR instead of RGB
            ch_idx = 2-i if convert_to_bgr else i
            flow_image[:,:,ch_idx] = np.floor(255 * col)
        except:
            print("Caution.")

    return flow_image


def flow_to_img(flow_uv, clip_flow=None, convert_to_bgr=False):
    """
    Expects a two dimensional flow image of shape.
    Args:
        flow_uv (np.ndarray): Flow UV image of shape [H,W,2]
        clip_flow (float, optional): Clip maximum of flow values. Defaults to None.
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.
    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """

    assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'

    if clip_flow is not None:
        # numpy.clip(a, a_min, a_max, out=None, **kwargs)
        flow_uv = np.clip(flow_uv, 0, clip_flow)

    u = flow_uv[:,:,0]
    v = flow_uv[:,:,1]

    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)
    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)

    return flow_uv_to_colors(u, v, convert_to_bgr)


vid_dir= "./trn"
video_id = "00000"
seq_len = len(os.listdir(os.path.join(vid_dir, "JPEGImages", video_id)))
fig, axs = plt.subplots(1,6, figsize=(20,10))

for i in range(seq_len -1):

    im_1 = np.array(Image.open(os.path.join(vid_dir, "JPEGImages", video_id, f"{i:05d}.jpg")))
    im_2 = np.array(Image.open(os.path.join(vid_dir, "JPEGImages", video_id, f"{i:05d}.jpg")))

    flow = read_flo_file(os.path.join(vid_dir, "OpticalFlow", video_id, f"{i:05d}.flo"))

    print("flow: ", flow.max())
    # flw = flow[0].numpy().transpose(1, 2, 0)
    # print(flw.min())
    flow_out = flow_to_img(flow)

    axs[0].set_title("Image 1")
    axs[0].imshow(im_1)

    axs[1].set_title("Seg 1")
    axs[1].imshow(plt.imread(os.path.join(vid_dir, "Annotations", video_id, f"{i:05d}.png")))

    axs[2].set_title("Image 2")
    axs[2].imshow(im_2)

    axs[3].set_title("Seg 2")
    axs[3].imshow(plt.imread(os.path.join(vid_dir, "Annotations", video_id, f"{(i+1):05d}.png")))

    axs[4].set_title("Flow")
    axs[4].imshow(flow_out)

    axs[5].set_title("Colorwheel")
    axs[5].imshow(get_colorwheel_img())

    axs[0].axes.get_xaxis().set_visible(False)
    axs[0].axes.get_yaxis().set_visible(False)
    axs[1].axes.get_xaxis().set_visible(False)
    axs[1].axes.get_yaxis().set_visible(False)
    axs[2].axes.get_xaxis().set_visible(False)
    axs[2].axes.get_yaxis().set_visible(False)
    axs[3].axes.get_xaxis().set_visible(False)
    axs[3].axes.get_yaxis().set_visible(False)
    axs[4].axes.get_xaxis().set_visible(False)
    axs[4].axes.get_yaxis().set_visible(False)
    axs[5].axes.get_xaxis().set_visible(False)
    axs[5].axes.get_yaxis().set_visible(False)

    fig.savefig(f"./out/{i:03d}.png")
    plt.close(fig)