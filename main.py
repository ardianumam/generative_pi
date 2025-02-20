import os, argparse
import numpy as np
from data import PiDataset
import torch
from torch.utils.data import DataLoader
from model import VAE, loss_function
import torch.optim as optim
import torch.nn.functional as F
from utils import compare_dist

def train(args):
    """
    Helper function to train
    """

    # query the device
    device = torch.device('cuda' if (torch.cuda.is_available() and args.use_cuda) else 'cpu')
    print(f"**** Start the training with device: {device} ****")

    # create data loader
    dataset = PiDataset(root="data")
    data_loader = DataLoader(dataset, 
                             batch_size=args.batch_size, 
                             shuffle=True, 
                             num_workers=4)
    
    # write the original data for checking purpose
    dataset.dump_data(data=dataset.data,
                      dir_out=args.output_dir,
                      filename="data_ori")
    
    # create model
    model = VAE(args.input_dim, args.latent_dim)
    model =model.to(device)

    # define optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Training loop
    for epoch in range(args.num_epochs):
        recons_loss_avg = 0; kldiv_loss_avg = 0; loss_all_avg = 0
        for input in data_loader:
            input = input.to(device) #[BS=32, n_data=5=xyrgb]
            optimizer.zero_grad()
            
            input_recons, z_mean, z_log_var = model(input) # input_recons=[BS, 5], z_mean=[BS, e=64], z_log_var=[BS, e=64] 
            loss_all, recons_loss, kldiv_loss = loss_function(input, input_recons, z_mean, z_log_var)
            
            loss_all_avg += loss_all.item()
            recons_loss_avg += recons_loss
            kldiv_loss_avg += kldiv_loss

            loss_all.backward()
            optimizer.step()

        recons_loss_avg /= float(len(data_loader))
        kldiv_loss_avg /= float(len(data_loader))
        loss_all_avg /= float(len(data_loader))
        recons_loss_avg = np.around(recons_loss_avg, decimals=4)
        kldiv_loss_avg = np.around(kldiv_loss_avg, decimals=4)
        loss_all_avg = np.around(loss_all_avg, decimals=4)

        print(f"Epoch {epoch+1}/{args.num_epochs} --> loss all: {loss_all_avg}, loss recons: {recons_loss_avg}, loss_kldiv: {kldiv_loss_avg}")
    
    # store the model
    os.makedirs(args.model_store_path, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict(),
                "epoch": args.num_epochs,
                "loss_all": recons_loss_avg+kldiv_loss_avg,
                "loss_recons": recons_loss_avg,
                "loss_kldiv": kldiv_loss_avg,
                "opt_state_dict": optimizer.state_dict()}, 
                os.path.join(args.model_store_path, "ckpt.pt"))

    # sample new sample from the latent space: random Gaussian noise
    model.eval()
    gen_data = model.sample_new_img_from_rand(num_samples=5000)
    dataset.dump_data(data=gen_data,
                      dir_out=args.output_dir,
                      filename="gen_train_from-random-noise")

    # sample new sample from the latent space: learned mean & var latent data
    gen_data = model.sample_new_img_from_learned_latent(input=dataset.data)
    dataset.dump_data(data=gen_data,
                      dir_out=args.output_dir,
                      filename="gen_train_from-learned-latent")



def test(args):
    # query the device
    device = torch.device('cuda' if (torch.cuda.is_available() and args.use_cuda) else 'cpu')
    print(f"**** Start the testing with device: {device} ****")

    # create data loader
    dataset = PiDataset(root="data")

    # create model
    model = VAE(args.input_dim, args.latent_dim)
    model = model.to(device)

    # load model weights
    checkpoint = torch.load(os.path.join(args.model_store_path, "ckpt.pt"))
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Model is loaded! Train epoch: {checkpoint['epoch']}, loss recons.: {checkpoint['loss_recons']}, loss kldiv.: {checkpoint['loss_kldiv']}")

    # sample new sample from the latent space: random Gaussian noise
    model.eval()
    gen_data_rand_noise = model.sample_new_img_from_rand(num_samples=5000)
    dataset.dump_data(data=gen_data_rand_noise,
                      dir_out=args.output_dir,
                      filename="gen_test_from-random-noise")

    # sample new sample from the latent space: learned mean & var latent data
    gen_data_learned_latent = model.sample_new_img_from_learned_latent(input=dataset.data)
    dataset.dump_data(data=gen_data_learned_latent,
                      dir_out=args.output_dir,
                      filename="gen_test_from-learned-latent")
    

    # compare distributions
    target_data = np.copy(dataset.data)
    ## 1. compare original data vs. generated data from random noise
    kl_gen_rand_noise = compare_dist(source=gen_data_rand_noise, 
                                     target=target_data, 
                                     warp_funct=dataset.warp_fg_to_image)

    ## 2. compare original data vs. generated data from random noise
    kl_gen_learned_latent = compare_dist(source=gen_data_learned_latent,
                                         target=target_data,
                                         warp_funct=dataset.warp_fg_to_image)
       
    ## 3. compare original data vs. random data
    rand_data = np.random.rand(dataset.n_pts, dataset.n_feat)
    rand_data[:,0] = rand_data[:,0]*dataset.img_h
    rand_data[:,1] = rand_data[:,1]*dataset.img_w
    rand_data[:,2:] = rand_data[:,2:]*255

    kl_rand = compare_dist(source=rand_data,
                           target=target_data,
                           warp_funct=dataset.warp_fg_to_image)
    
    ## 4. compare original data vs. itself as reference
    kl_ori = compare_dist(source=target_data,
                          target=target_data,
                          warp_funct=dataset.warp_fg_to_image)
    
    print("************** Distribution info via KL divergence **************")
    print(f"1. Original data vs. gen. data (rand-noise)    : {kl_gen_rand_noise}")
    print(f"2. Original data vs. gen. data (learned-latent): {kl_gen_learned_latent}")
    print(f"3. Original data vs. random data               : {kl_rand}")
    print(f"4. Original data vs. itself (as ref.)          : {kl_ori}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pi Generator arguments using VAE')
    parser.add_argument("--is_train", type=int, default=0, help="Set 1 to train or 0 to test")
    parser.add_argument("--use_cuda", type=int, default=0, help="Set 1 to use CUDA or 0 to use CPU")
    parser.add_argument("--model_store_path", type=str, default="model/store", help="Where to store the trained model")
    parser.add_argument("--output_dir", type=str, default="output", help="Where to store the generated data output")
    parser.add_argument("--num_epochs", type=int, default="500", help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default="32", help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--input_dim", type=int, default="5", help="Input dimension. By default is 5, for x,y,r,g,b")
    parser.add_argument("--latent_dim", type=int, default="32", help="Latent space dimension for the VAE model")
    args = parser.parse_args()
    
    if args.is_train:
        train(args)
    else:
        test(args)