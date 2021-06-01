"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import argparse
import os
import torch

from termcolor import colored
from utils.config import create_config
from utils.common_config import get_train_transformations, get_val_transformations,\
                                get_train_dataset, get_train_dataloader,\
                                get_val_dataset, get_val_dataloader,\
                                get_optimizer, get_model, get_criterion,\
                                adjust_learning_rate
from utils.evaluate_utils import linearprobe_evaluate
from utils.train_utils import linearprobe_train

FLAGS = argparse.ArgumentParser(description='Linear Probing')
FLAGS.add_argument('--config_env', help='Location of path config file')
FLAGS.add_argument('--config_exp', help='Location of experiments config file')
FLAGS.add_argument('--mode', help='evaluate or train')

def main():
    args = FLAGS.parse_args()
    p = create_config(args.config_env, args.config_exp)
    print(colored(p, 'red'))

    # CUDNN
    torch.backends.cudnn.benchmark = True

    # Data
    print(colored('Get dataset and dataloaders', 'blue'))
    train_transformations = get_train_transformations(p)
    val_transformations = get_val_transformations(p)
    print('Train transforms:', train_transformations)
    print('Validation transforms:', val_transformations)
    train_dataset = get_train_dataset(p, train_transformations, 
                                        split='train')
    val_dataset = get_val_dataset(p, val_transformations)
    train_dataloader = get_train_dataloader(p, train_dataset)
    val_dataloader = get_val_dataloader(p, val_dataset)
    print('Train samples %d - Val samples %d' %(len(train_dataset), len(val_dataset)))
    
    # Model
    print(colored('Get model', 'blue'))
    model = get_model(p, p['pretext_model'])
    print(model)
    model = torch.nn.DataParallel(model)
    model = model.cuda()
    state = torch.load(p['pretext_model'], map_location='cpu')
    missing = model.load_state_dict(state, strict=False)
    print('missing components', missing)

    # Optimizer
    print(colored('Get optimizer', 'blue'))
    optimizer = get_optimizer(p, model, p['update_cluster_head_only'])
    print(optimizer)
    
    # Warning
    if p['update_cluster_head_only']:
        print(colored('WARNING: Linear probing will only update the cluster head', 'red'))

    # Loss function
    print(colored('Get loss', 'blue'))
    criterion = get_criterion(p) 
    criterion.cuda()
    print(criterion)

    if args.mode == 'train':
        # Checkpoint
        if os.path.exists(p['linearprobe_checkpoint']):
            print(colored('Restart from checkpoint {}'.format(p['linearprobe_checkpoint']), 'blue'))
            checkpoint = torch.load(p['linearprobe_checkpoint'], map_location='cpu')
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])        
            start_epoch = checkpoint['epoch']
            best_loss = checkpoint['best_loss']

        else:
            print(colored('No checkpoint file at {}'.format(p['linearprobe_checkpoint']), 'blue'))
            start_epoch = 0
            best_loss = 1e4
    
        # Main loop
        print(colored('Starting main loop', 'blue'))

        for epoch in range(start_epoch, p['epochs']):
            print(colored('Epoch %d/%d' %(epoch+1, p['epochs']), 'yellow'))
            print(colored('-'*15, 'yellow'))

            # Adjust lr
            lr = adjust_learning_rate(p, optimizer, epoch)
            print('Adjusted learning rate to {:.5f}'.format(lr))

            # Train
            print('Train ...')
            linearprobe_train(train_dataloader, model, criterion, optimizer, epoch)

            if (epoch +  1) % 5 == 0:
                print('Evaluate based on CE loss ...')
                linearprobe_stats = linearprobe_evaluate(val_dataloader, model, criterion)
                loss = linearprobe_stats['loss']
                if loss < best_loss:
                    best_loss = loss
                    torch.save({'model': model.module.state_dict()}, p['linearprobe_model'])
                    
                # Checkpoint
                print('Checkpoint ...')
                print(linearprobe_stats)
                torch.save({'optimizer': optimizer.state_dict(), 'model': model.state_dict(), 
                            'epoch': epoch + 1, 'best_loss': loss},
                            p['linearprobe_checkpoint'])
    
    # Evaluate and save the final model
    print(colored('Evaluate best model', 'blue'))
    model_checkpoint = torch.load(p['linearprobe_model'], map_location='cpu')
    model.module.load_state_dict(model_checkpoint['model'])
    linearprobe_stats = linearprobe_evaluate(val_dataloader, model, criterion)
    print(linearprobe_stats)
    print('Final Accuracy:', linearprobe_stats['accuracy'])
    
if __name__ == "__main__":
    main()
