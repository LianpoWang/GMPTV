from dataloader.flyingthings3d import FlyingThings3D_subset, FlyingThings3D_flownet3d
from dataloader.fluidflow3d import Fluidflow3D

def build_train_dataset(args):

    if args.stage == 'fluidflow3d':
        train_dataset = Fluidflow3D(partition='train')
    if args.stage == 'kitti15':
        train_dataset = KITTI_flownet3d(split='training40')
    if args.stage == 'things_subset':
        train_dataset = FlyingThings3D_subset(split='train', occ=True)
    if args.stage == 'things_subset_non-occluded':
        train_dataset = FlyingThings3D_subset(split='train', occ=False)
    if args.stage == 'things_flownet3d':
        train_dataset = FlyingThings3D_flownet3d(train=True)
    if args.stage == 'waymo':
        train_dataset = Waymo(split='train')
#    else:
#        raise ValueError(f'stage {args.stage} is not supported')

    return train_dataset
