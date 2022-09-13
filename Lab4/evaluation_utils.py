import math
# from operator import pos
import imageio
import numpy as np
import torch
# from PIL import Image, ImageDraw
from scipy import signal
from torch.autograd import Variable
# from torchvision import transforms
from PIL.Image import fromarray

def mse_metric(x1, x2):
    err = np.sum((x1 - x2) ** 2)
    err /= float(x1.shape[0] * x1.shape[1] * x1.shape[2])
    return err

# ssim function used in Babaeizadeh et al. (2017), Fin et al. (2016), etc.
def finn_eval_seq(gt, pred):
    T = len(gt)
    bs = gt[0].shape[0]
    ssim = np.zeros((bs, T))
    psnr = np.zeros((bs, T))
    mse = np.zeros((bs, T))
    for i in range(bs):
        for t in range(T):
            origin = gt[t][i].detach().cpu().numpy()
            predict = pred[t][i].detach().cpu().numpy()
            for c in range(origin.shape[0]):
                res = finn_ssim(origin[c], predict[c]).mean()
                if math.isnan(res):
                    ssim[i, t] += -1
                else:
                    ssim[i, t] += res
                psnr[i, t] += finn_psnr(origin[c], predict[c])
            ssim[i, t] /= origin.shape[0]
            psnr[i, t] /= origin.shape[0]
            mse[i, t] = mse_metric(origin, predict)

    return mse, ssim, psnr

def finn_psnr(x, y, data_range=1.):
    mse = ((x - y)**2).mean()
    return 20 * math.log10(data_range) - 10 * math.log10(mse)

def fspecial_gauss(size, sigma):
    x, y = np.mgrid[-size // 2 + 1:size // 2 + 1, -size // 2 + 1:size // 2 + 1]
    g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2)))
    return g / g.sum()

def finn_ssim(img1, img2, data_range=1., cs_map=False):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    size = 11
    sigma = 1.5
    window = fspecial_gauss(size, sigma)

    K1 = 0.01
    K2 = 0.03

    C1 = (K1 * data_range) ** 2
    C2 = (K2 * data_range) ** 2
    mu1 = signal.fftconvolve(img1, window, mode='valid')
    mu2 = signal.fftconvolve(img2, window, mode='valid')
    mu1_sq = mu1*mu1
    mu2_sq = mu2*mu2
    mu1_mu2 = mu1*mu2
    sigma1_sq = signal.fftconvolve(img1*img1, window, mode='valid') - mu1_sq
    sigma2_sq = signal.fftconvolve(img2*img2, window, mode='valid') - mu2_sq
    sigma12 = signal.fftconvolve(img1*img2, window, mode='valid') - mu1_mu2

    if cs_map:
        return (((2 * mu1_mu2 + C1) * (2 * sigma12 + C2))/((mu1_sq + mu2_sq + C1) *
                    (sigma1_sq + sigma2_sq + C2)), 
                (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2))
    else:
        return ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                    (sigma1_sq + sigma2_sq + C2))

def is_seq(obj):
    return (not (type(obj) is np.ndarray) and
        not hasattr(obj, "strip") and 
        not hasattr(obj, "dot") and
        (hasattr(obj, "__getitem__") or hasattr(obj, "__iter__")))

def image_tensor(X, padding = 1):
    assert len(X) > 0

    # list of lists case: unpack them and grid them up
    if is_seq(X[0]) or (hasattr(X, "dim") and X.dim() > 4):
        images = [image_tensor(i) for i in X]
        c_dim = images[0].size(0)
        h_dim = images[0].size(1)
        w_dim = images[0].size(2)
    
        result = torch.ones(c_dim, h_dim * len(images) + padding * (len(images) - 1), w_dim)
        for i, image in enumerate(images):
            result[:, (i * h_dim + i * padding):((i + 1) * h_dim + i * padding), :].copy_(image)
        
        return result
    # pure list case: make stacked image
    else:
        images = [i.data if isinstance(i, Variable) else i for i in X]
        c_dim = images[0].size(0)
        h_dim = images[0].size(1)
        w_dim = images[0].size(2)

        result = torch.ones(c_dim, h_dim, w_dim * len(images) + padding * (len(images) - 1))
        for i, image in enumerate(images):
            result[:, :, (i * w_dim + i * padding):((i + 1) * w_dim + i * padding)].copy_(image)
        
        return result

def make_image(tensor):
    tensor = tensor.cpu().clamp(0, 1)
    return fromarray((np.transpose(tensor.numpy(), (1, 2, 0)) * 255).astype(np.uint8))

def save_image(fname, tensor):
    img = make_image(tensor)
    img.save(fname)

def save_tensors_image(fname, X, padding = 1):
    images = image_tensor(X, padding)
    return save_image(fname, images)

def save_gif(fname, X, duration = 0.25):
    images = []
    for tensor in X:
        img = image_tensor(tensor, padding = 0)
        img = img.cpu()
        img = img.transpose(0, 1).transpose(1, 2).clamp(0, 1)
        # img = img.clamp(0, 1)
        images.append((img.numpy() * 255).astype(np.uint8))
    imageio.mimsave(fname, images, duration = duration)

def pred(x, cond, modules, args):
    gen_seq = [x[0]]

    modules['frame_predictor'].hidden = modules['frame_predictor'].init_hidden()
    # modules['posterior'].hidden = modules['posterior'].init_hidden() # why?
    h_seq = [modules['encoder'](x[i]) for i in range(args.n_past)]

    x_in = x[0]
    for i in range(1, args.n_eval):
        if args.last_frame_skip or i < args.n_past:
            h, skip = h_seq[i-1]
            h = h.detach()
        else:
            h = h_seq[i-1][0]
            h = h.detach()
        cond_curr = cond[i]
        
        if i < args.n_past:
            # z_t, _, _ = modules['posterior'](torch.cat([cond[i], h_seq[i][0]], 1))
            z_t, _, _ = modules['posterior'](h_seq[i][0])
            modules['frame_predictor'](torch.cat([cond_curr, h, z_t], 1))
            x_in = x[i]
            gen_seq.append(x_in)
        else:
            z_t = torch.cuda.FloatTensor(args.batch_size, args.z_dim).normal_() # prediction, no hidden states
            h = modules['frame_predictor'](torch.cat([cond_curr, h, z_t], 1)).detach()
            x_in = modules['decoder']([h, skip]).detach()
            gen_seq.append(x_in)
            h_seq.append(modules['encoder'](x_in)) # new
    
    return gen_seq

def plot_pred(x, cond, modules, epoch, args):
    gen_seq = [x[0]]
    gt_seq = [x[i] for i in range(args.n_past + args.n_future)]

    modules['frame_predictor'].hidden = modules['frame_predictor'].init_hidden()
    # modules['posterior'].hidden = modules['posterior'].init_hidden() # why?
    h_seq = [modules['encoder'](x[i]) for i in range(args.n_past)]

    x_in = x[0]
    for i in range(1, args.n_past + args.n_future):
        if args.last_frame_skip or i < args.n_past:
            h, skip = h_seq[i-1]
            h = h.detach()
        else:
            h = h_seq[i-1][0]
            h = h.detach()
        cond_curr = cond[i]
        
        if i < args.n_past:
            z_t, _, _ = modules['posterior'](h_seq[i][0])
            modules['frame_predictor'](torch.cat([cond_curr, h, z_t], 1))
            x_in = x[i]
            gen_seq.append(x_in)
        else:
            z_t = torch.cuda.FloatTensor(args.batch_size, args.z_dim).normal_() # prediction, no hidden states
            h = modules['frame_predictor'](torch.cat([cond_curr, h, z_t], 1)).detach()
            x_in = modules['decoder']([h, skip]).detach()
            gen_seq.append(x_in)
            h_seq.append(modules['encoder'](x_in))
    
    to_plot = []
    gifs = [[] for _ in range(args.n_past + args.n_future)]
    # nrow = min(args.batch_size, 10)
    nrow = 1
    for i in range(nrow):
        # ground truth seq
        row = []
        for t in range(args.n_past + args.n_future):
            row.append(gt_seq[t][i])
        to_plot.append(row)

        row = []
        for t in range(args.n_past + args.n_future):
            row.append(gen_seq[t][i])
        to_plot.append(row)

        for t in range(args.n_past + args.n_future):
            row = []
            row.append(gt_seq[t][i])
            row.append(gen_seq[t][i])
            gifs[t].append(row)
    
    fname = f"test.png"
    save_tensors_image(fname, to_plot)

    fname = f"test.gif"
    save_gif(fname, gifs)