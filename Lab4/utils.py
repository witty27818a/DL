import math
# from operator import pos
import imageio
import numpy as np
import matplotlib.pyplot as plt
import torch
# from PIL import Image, ImageDraw
from scipy import signal
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric
from torch.autograd import Variable
# from torchvision import transforms
from PIL.Image import fromarray


def kl_criterion(mu, logvar, args):
  # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD /= args.batch_size  
    return KLD
    
def eval_seq(gt, pred):
    T = len(gt)
    bs = gt[0].shape[0]
    ssim = np.zeros((bs, T))
    psnr = np.zeros((bs, T))
    mse = np.zeros((bs, T))
    for i in range(bs):
        for t in range(T):
            origin = gt[t][i]
            predict = pred[t][i]
            for c in range(origin.shape[0]):
                ssim[i, t] += ssim_metric(origin[c], predict[c]) 
                psnr[i, t] += psnr_metric(origin[c], predict[c])
            ssim[i, t] /= origin.shape[0]
            psnr[i, t] /= origin.shape[0]
            mse[i, t] = mse_metric(origin, predict)

    return mse, ssim, psnr

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

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# def normalize(dtype, seq):
#     seq.transpose_(0, 1)
#     seq.transpose_(3, 4)
#     seq.transpose_(2, 3)
#     # result shape = (1, batch_size, 3, 64, 64)

#     return [Variable(X.type(dtype)) for X in seq]

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
        
        if i < args.n_past:
            z_t, _, _ = modules['posterior'](h_seq[i][0])
            modules['frame_predictor'](torch.cat([cond[i], h, z_t], 1))
            x_in = x[i]
            gen_seq.append(x_in)
        else:
            z_t = torch.cuda.FloatTensor(args.batch_size, args.z_dim).normal_() # prediction, no hidden states
            h = modules['frame_predictor'](torch.cat([cond[i], h, z_t], 1)).detach()
            x_in = modules['decoder']([h, skip]).detach()
            gen_seq.append(x_in)
            h_seq.append(modules['encoder'](x_in)) # new
    
    return gen_seq

def plot_pred(x, cond, modules, epoch, args):
    gen_seq = [x[0]]
    gt_seq = [x[i] for i in range(len(x))]

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
        
        if i < args.n_past:
            z_t, _, _ = modules['posterior'](h_seq[i][0])
            modules['frame_predictor'](torch.cat([cond[i], h, z_t], 1))
            x_in = x[i]
            gen_seq.append(x_in)
        else:
            z_t = torch.cuda.FloatTensor(args.batch_size, args.z_dim).normal_() # prediction, no hidden states
            h = modules['frame_predictor'](torch.cat([cond[i], h, z_t], 1)).detach()
            x_in = modules['decoder']([h, skip]).detach()
            gen_seq.append(x_in)
            h_seq.append(modules['encoder'](x_in))
    
    to_plot = []
    gifs = [[] for _ in range(args.n_eval)]
    # nrow = min(args.batch_size, 10)
    nrow = 1
    for i in range(nrow):
        # ground truth seq
        row = []
        for t in range(args.n_eval):
            row.append(gt_seq[t][i])
        to_plot.append(row)

        row = []
        for t in range(args.n_eval):
            row.append(gen_seq[t][i])
        to_plot.append(row)

        for t in range(args.n_eval):
            row = []
            row.append(gt_seq[t][i])
            row.append(gen_seq[t][i])
            gifs[t].append(row)
    
    fname = f"{args.log_dir}/gif_and_imgseq/sample_{epoch}.png"
    save_tensors_image(fname, to_plot)

    fname = f"{args.log_dir}/gif_and_imgseq/sample_{epoch}.gif"
    save_gif(fname, gifs)

def plot_curves(rec_dict, epoch, savepath, step = 5):
    epochs = np.arange(epoch)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.plot(epochs, rec_dict["tfr"], '--b', label = "tfr")
    ax1.plot(epochs, rec_dict["KL weight"], '--g', label = "KL weight")
    ax1a = ax1.twinx()
    ax1a.plot(epochs, rec_dict["loss"], 'y', label = "loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("score/weight")
    ax1a.set_ylabel("loss")
    ax1.legend()
    ax1a.legend()

    epochs2 = np.arange(0, epoch, step)
    ax2.plot(epochs, rec_dict["tfr"], '--b', label = "tfr")
    ax2.plot(epochs, rec_dict["KL weight"], '--g', label = "KL weight")
    ax2a = ax2.twinx()
    ax2a.plot(epochs2, rec_dict["psnr"], ':r', label = "psnr")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("score/weight")
    ax2a.set_ylabel("psnr score")
    ax2.legend()
    ax2a.legend()
    plt.tight_layout()
    plt.savefig(savepath)
    plt.show()