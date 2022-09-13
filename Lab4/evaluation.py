import torch
import random
import numpy as np
from dataset import bair_robot_pushing_dataset
from torch.utils.data import DataLoader
from evaluation_utils import pred, finn_eval_seq, plot_pred
from kl_annealing import kl_annealing

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "log_linearKL/model.pth"

    saved_model = torch.load(model_path)

    random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)

    args = saved_model["args"]
    args.data_root = "D://DL_lab4_data"
    frame_predictor = saved_model['frame_predictor']
    posterior = saved_model['posterior']
    decoder = saved_model['decoder']
    encoder = saved_model['encoder']
    frame_predictor.to(device)
    posterior.to(device)
    encoder.to(device)
    decoder.to(device)
    frame_predictor.eval()
    posterior.eval()
    encoder.eval()
    decoder.eval()

    test_data = bair_robot_pushing_dataset(args, "test")
    test_loader = DataLoader(test_data,
                            num_workers=args.num_workers,
                            batch_size=args.batch_size,
                            shuffle=True,
                            drop_last=True,
                            pin_memory=True)
    test_iterator = iter(test_loader)

    modules = {
        'frame_predictor': frame_predictor,
        'posterior': posterior,
        'encoder': encoder,
        'decoder': decoder
    }

    psnr_list = []
    for _ in range(len(test_data) // args.batch_size):
        try:
            test_seq, test_cond = next(test_iterator)
        except StopIteration:
            test_iterator = iter(test_loader)
            test_seq, test_cond = next(test_iterator)
        test_seq = test_seq.transpose_(0, 1).to(device)
        test_cond = test_cond.transpose_(0, 1).to(device)

        with torch.no_grad():
            pred_seq = pred(test_seq, test_cond, modules, args)
        # _, _, psnr = finn_eval_seq(test_seq[args.n_past:], pred_seq[args.n_past:])
        _, _, psnr = finn_eval_seq(test_seq[args.n_past: args.n_past+args.n_future], pred_seq[args.n_past: args.n_past+args.n_future])
        psnr_list.append(psnr)
        
    ave_psnr = np.mean(np.concatenate(psnr_list))
    print(ave_psnr)

    try:
        test_seq, test_cond = next(test_iterator)
    except StopIteration:
        test_iterator = iter(test_loader)
        test_seq, test_cond = next(test_iterator)
    test_seq = test_seq.transpose_(0, 1).to(device)
    test_cond = test_cond.transpose_(0, 1).to(device)

    with torch.no_grad():
        plot_pred(test_seq, test_cond, modules, "test", args)

if __name__ == '__main__':
    main()