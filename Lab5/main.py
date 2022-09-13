import os
from copy import deepcopy
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from dataset import CLEVRDataset
from model import Generator, Discriminator
from evaluator import evaluation_model
from utils import test_obj_getter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

z_dim = 100
c_dim = 300
epochs = 500
# origin = 2e-4
lr_generator = 1e-4
lr_discriminator = 4e-4
batch_size = 64 # 64
loss_fn = nn.BCELoss()
log_dir = "D://DL_lab5_params/GANlog"
model_dir = os.path.join(log_dir, "model_weights")
model_dir_new = os.path.join(log_dir, "model_weights_new")
result_img_dir = os.path.join(log_dir, "result_images")
os.makedirs(log_dir, exist_ok = True)
os.makedirs(model_dir, exist_ok = True)
os.makedirs(model_dir_new, exist_ok = True)
os.makedirs(result_img_dir, exist_ok = True)
file = open(os.path.join(log_dir, "train_log.txt"), "w")

if __name__ == "__main__":
    # load training data
    train_dataset = CLEVRDataset("D://DL_lab5_data/iclevr")
    train_dataloader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers = 4)

    # create models
    generator = Generator(z_dim, c_dim).to(device)
    discriminator = Discriminator((64, 64, 3), c_dim).to(device)
    evaluator = evaluation_model()

    generator.init_weights() # (0, 0.02) for deconv and (1, 0.02) for batchnorm
    discriminator.init_weights() # (0, 0.02) for conv and (1, 0.02) for batchnorm

    # optimizer
    optimizer_generator = optim.Adam(generator.parameters(), lr = lr_generator) # can try betas = (0.5, 0.99)
    optimizer_discriminator = optim.Adam(discriminator.parameters(), lr = lr_discriminator)

    # training
    best_score = 0
    best_score_new = 0
    best_models_weights = None
    best_models_weights_new = None
    reals = torch.ones(batch_size).to(device)
    fakes = torch.zeros(batch_size).to(device)
    # the last batch might be smaller
    reals_last_batch = torch.ones(len(train_dataloader.dataset) % batch_size).to(device)
    fakes_last_batch = torch.zeros(len(train_dataloader.dataset) % batch_size).to(device)
    test_objs = test_obj_getter().to(device)
    test_objs_new = test_obj_getter(test_json_path = "new_test.json").to(device)
    z_for_evaluation = torch.randn(len(test_objs), z_dim).to(device)
    z_for_evaluation_new = torch.randn(len(test_objs_new), z_dim).to(device)

    progress = tqdm(total = epochs)
    df = {"generator loss": [], "discriminator loss": [], "score": []}
    for e in range(epochs):
        generator_loss, discriminator_loss = 0, 0
        generator.train()
        discriminator.train()

        for i, (imgs, objs) in enumerate(train_dataloader):
            imgs = imgs.to(device)
            objs = objs.to(device)

            '''train discriminator first'''
            optimizer_discriminator.zero_grad()

            # for real images
            predictions = discriminator(imgs, objs)
            if predictions.size(0) == batch_size:
                loss_reals = loss_fn(predictions, reals)
            else: #last batch
                loss_reals = loss_fn(predictions, reals_last_batch)
            # for fake images
            if predictions.size(0) == batch_size:
                z = torch.randn(batch_size, z_dim).to(device)
            else:
                z = torch.randn(len(train_dataloader.dataset) % batch_size, z_dim).to(device)
            fake_imgs = generator(z, objs)
            predictions = discriminator(fake_imgs.detach(), objs)
            if predictions.size(0) == batch_size:
                loss_fakes = loss_fn(predictions, fakes)
            else:
                loss_fakes = loss_fn(predictions, fakes_last_batch)

            discriminator_loss_batched = loss_reals + loss_fakes
            discriminator_loss_batched.backward()
            optimizer_discriminator.step()

            '''train generator later, 5 times the training iterations''' # can try 4
            for _ in range(4):
                optimizer_generator.zero_grad()

                # generate fake images and compute loss w.r.t. real images
                if predictions.size(0) == batch_size:
                    z = torch.randn(batch_size, z_dim).to(device)
                else:
                    z = torch.randn(len(train_dataloader.dataset) % batch_size, z_dim).to(device)
                fake_imgs = generator(z, objs)
                predictions = discriminator(fake_imgs, objs)
                
                if predictions.size(0) == batch_size:
                    generator_loss_batched = loss_fn(predictions, reals)
                else:
                    generator_loss_batched = loss_fn(predictions, reals_last_batch)
                generator_loss_batched.backward()
                optimizer_generator.step()
            
            # print(f"Epoch {e+1:>03d}, {i+1:>03d} / {len(train_dataloader)}: generator loss = {generator_loss_batched.item():.3f}, discriminator loss = {discriminator_loss_batched.item():.3f}")
            # file.write(f"Epoch {e+1:>03d}, {i+1:>03d} / {len(train_dataloader)}: generator loss = {generator_loss_batched.item():.3f}, discriminator loss = {discriminator_loss_batched.item():.3f}\n")
            generator_loss += generator_loss_batched.item()
            discriminator_loss += discriminator_loss_batched.item()
        
        '''evaluation'''
        generator.eval()
        discriminator.eval()

        with torch.no_grad():
            fake_imgs = generator(z_for_evaluation, test_objs)
        score = evaluator.eval(fake_imgs, test_objs)

        if score > best_score:
            best_score = score
            best_models_weights = deepcopy(generator.state_dict())
            torch.save(best_models_weights, os.path.join(model_dir, "model_epoch_{:03d}_score_{:.2f}.pt".format(e+1, score)))
            save_image(fake_imgs, os.path.join(result_img_dir, "epoch_{:03d}.png".format(e+1)), nrow = 8, normalize = True)
        print(f"average generator loss = {generator_loss / len(train_dataloader):.3f}, average discriminator loss = {discriminator_loss / len(train_dataloader):.3f}")
        file.write(f"average generator loss = {generator_loss / len(train_dataloader):.3f}, average discriminator loss = {discriminator_loss / len(train_dataloader):.3f}\n")
        print(f"testing score = {score:.2f}")
        file.write(f"testing score = {score:.2f}\n")
        print("--------------------------------------------------")
        file.write("--------------------------------------------------\n")
        df["discriminator loss"].append(generator_loss / len(train_dataloader))
        df["generator loss"].append(discriminator_loss / len(train_dataloader))
        df["score"].append(score)

        '''test'''
        generator.eval()
        discriminator.eval()

        with torch.no_grad():
            fake_imgs_new = generator(z_for_evaluation_new, test_objs_new)
        score_new = evaluator.eval(fake_imgs_new, test_objs_new)

        if score_new > best_score_new:
            best_score_new = score_new
            best_models_weights_new = deepcopy(generator.state_dict())
            torch.save(best_models_weights_new, os.path.join(model_dir_new, "model_epoch_{:03d}_score_{:.2f}.pt".format(e+1, score_new)))

        progress.update(1)
    torch.save(df, os.path.join(log_dir, "curves.pt"))

file.close()