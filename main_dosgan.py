import os
import argparse
from solver_dosgan import Solver
from torch.backends import cudnn
from torchvision import transforms, datasets
import torch.utils.data as data
def str2bool(v):
    return v.lower() in ('true')
def train_trans():
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.Resize((128,128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
def test_trans():
    return transforms.Compose([
        transforms.Resize((128,128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
def main(config):
    # For fast training.
    cudnn.benchmark = True

    # Create directories if not exist.
    config.log_dir = os.path.join(config.model_dir, 'logs')
    config.model_save_dir = os.path.join(config.model_dir, 'models')
    config.sample_dir = os.path.join(config.model_dir, 'samples')
    config.result_dir = os.path.join(config.model_dir, 'results')
    
    if not os.path.exists(config.log_dir):
        os.makedirs(config.log_dir)
    if not os.path.exists(config.model_save_dir):
        os.makedirs(config.model_save_dir)
    if not os.path.exists(config.sample_dir):
        os.makedirs(config.sample_dir)
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)

    # Data loader.
    
    train_dataset = datasets.ImageFolder(config.train_data_path, train_trans())
    
    test_dataset = datasets.ImageFolder(config.test_data_path, test_trans())
    if config.mode == 'train' or config.mode == 'cls':
        data_loader_my = data.DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, drop_last=True)
        print('train dataset loaded')
    else:
        data_loader_my = data.DataLoader(dataset=test_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers, drop_last=True)
        print('test dataset loaded')
   
    
    
    

    # Solver for training and testing StarGAN.
    solver = Solver(data_loader_my, config)

    if config.mode == 'train':
        if config.non_conditional:
            solver.train()
        else:
            solver.train_conditional()
    elif config.mode == 'test':
        solver.test()
    elif config.mode == 'cls':
        solver.cls()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model configuration.
    parser.add_argument('--c_dim', type=int, default=531, help='number of domains')
    parser.add_argument('--d_conv_dim', type=int, default=64, help='number of conv filters in the first layer of D')
    parser.add_argument('--d_repeat_num', type=int, default=6, help='number of strided conv layers in D')
    parser.add_argument('--n_blocks', type=int, default=0, help='number of res conv layers in C')
    parser.add_argument('--image_size', type=int, default=128, help='image resolution')
    parser.add_argument('--lambda_rec', type=float, default=10, help='weight for self-reconstruction loss')
    parser.add_argument('--lambda_rec2', type=float, default=10, help='weight for cross-reconstruction2 loss')
    parser.add_argument('--lambda_gp', type=float, default=10, help='weight for gradient penalty')
    parser.add_argument('--lambda_fs', type=float, default=5, help='weight for fs recontrcution')
    parser.add_argument('--ft_num', type=int, default=1024, help='number of ds feature')
    
    # Training configuration.
    parser.add_argument('--batch_size', type=int, default=6, help='mini-batch size')
    parser.add_argument('--num_iters', type=int, default=200000, help='number of total iterations for training D')
    parser.add_argument('--num_iters_decay', type=int, default=100000, help='number of iterations for decaying lr')
    parser.add_argument('--g_lr', type=float, default=0.0001, help='learning rate for encoder and decoder')
    parser.add_argument('--d_lr', type=float, default=0.0001, help='learning rate for D')
    parser.add_argument('--n_critic', type=int, default=5, help='number of D updates per each generator update')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam optimizer')
    parser.add_argument('--resume_iters', type=int, default=None, help='resume training from this step')
    # Test configuration.
    parser.add_argument('--test_iters', type=int, default=200000, help='test model from this step')
    parser.add_argument('--non_conditional', type=str2bool, default=True)

    # Miscellaneous.
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'cls'])

    # Directories.
    parser.add_argument('--train_data_path', type=str, default='/data1/linjx/intnet/data/facescrub_train/')
    parser.add_argument('--test_data_path', type=str, default='/data1/linjx/intnet/data/facescrub_test/')
    parser.add_argument('--model_dir', type=str, default='stargan')
    parser.add_argument('--cls_save_dir', type=str, default='stargan_cls/models')


    # Step size.
    parser.add_argument('--log_step', type=int, default=10)
    parser.add_argument('--sample_step', type=int, default=1000)
    parser.add_argument('--model_save_step', type=int, default=10000)
    parser.add_argument('--lr_update_step', type=int, default=1000)

    config = parser.parse_args()
    print(config)
    main(config)

