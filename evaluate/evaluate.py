from PIL import Image
import os
import time
import numpy as np
import torch
import torch.nn.functional as F

from dataloader.flyingthings3d import FlyingThings3D_subset, FlyingThings3D_flownet3d
from dataloader.fluidflow3d import Fluidflow3D
from glob import glob

@torch.no_grad()
def validate_things(stage,
                    model,
                    ):
    """ Perform validation using the Things (test) split with added NEPE metric """
    model.eval()

    if stage == 'fluidflow3d':
        val_dataset = Fluidflow3D(partition='test')
    if stage == 'things_flownet3d':
        val_dataset = FlyingThings3D_flownet3d(train=False)
    if stage == 'things_subset':
        val_dataset = FlyingThings3D_subset(split='val', occ=True)
    if stage == 'things_subset_non-occluded':
        val_dataset = FlyingThings3D_subset(split='val', occ=False)

    print('Number of validation image pairs: %d' % len(val_dataset))
    epe_list = []
    results = {}
    metrics_3d = {'counts': 0, 'EPE3d': 0.0, 'NEPE3d': 0.0, '5cm': 0.0, '10cm': 0.0, 'outlier': 0.0}
    metrics_3d_noc = {'counts': 0, 'EPE3d': 0.0, 'NEPE3d': 0.0, '5cm': 0.0, '10cm': 0.0, 'outlier': 0.0}

    import timeit
    start = timeit.default_timer()
    for val_id in range(len(val_dataset)):
        data_dict = val_dataset[val_id]
        pcs = data_dict['pcs'].unsqueeze(0) # 8192*6
        flow_3d = data_dict['flow_3d'].unsqueeze(0).cuda()
        pc1 = pcs[:,:,0:3].cuda()
        pc2 = pcs[:,:,3:6].cuda()

        results_dict_point = model(pc0=pc1, pc1=pc2)
        flow_3d_pred = results_dict_point['flow_preds'][-1]

        if flow_3d[0].shape[0] > 3:
            flow_3d_mask = flow_3d[0][3] > 0
            flow_3d_target = flow_3d[0][:3]
        else:
            flow_3d_mask = torch.ones(flow_3d[0].shape[1], dtype=torch.int64).cuda()
            flow_3d_target = flow_3d[0][:3]

        # Calculate EPE3D and NEPE3D
        epe3d_map = torch.sqrt(torch.sum((flow_3d_pred[0] - flow_3d_target) ** 2, dim=0))
        nepe3d_map = epe3d_map / torch.sqrt(torch.sum(flow_3d_target ** 2, dim=0))  # Normalized error

        # Filter out NaNs in the mask
        flow_3d_mask = torch.logical_and(flow_3d_mask, torch.logical_not(torch.isnan(epe3d_map)))
        metrics_3d['counts'] += epe3d_map[flow_3d_mask].shape[0]
        metrics_3d['EPE3d'] += epe3d_map[flow_3d_mask].sum().item()
        metrics_3d['NEPE3d'] += nepe3d_map[flow_3d_mask].sum().item()
        metrics_3d['5cm'] += torch.count_nonzero(torch.logical_or((epe3d_map[flow_3d_mask] < 0.05), (epe3d_map[flow_3d_mask]/torch.sqrt(torch.sum(flow_3d_target ** 2, dim=0))[flow_3d_mask] < 0.05))).item()
        metrics_3d['10cm'] += torch.count_nonzero(torch.logical_or((epe3d_map[flow_3d_mask] < 0.1), (epe3d_map[flow_3d_mask]/torch.sqrt(torch.sum(flow_3d_target ** 2, dim=0))[flow_3d_mask] < 0.1))).item()
        metrics_3d['outlier'] += torch.count_nonzero(torch.logical_or((epe3d_map[flow_3d_mask] > 0.3), (epe3d_map[flow_3d_mask]/torch.sqrt(torch.sum(flow_3d_target ** 2, dim=0))[flow_3d_mask] > 0.1))).item()

        # Evaluate on non-occluded points
        epe3d_map_noc = epe3d_map[flow_3d_mask]
        nepe3d_map_noc = nepe3d_map[flow_3d_mask]
        metrics_3d_noc['counts'] += epe3d_map_noc.shape[0]
        metrics_3d_noc['EPE3d'] += epe3d_map_noc.sum().item()
        metrics_3d_noc['NEPE3d'] += nepe3d_map_noc.sum().item()
        metrics_3d_noc['5cm'] += torch.count_nonzero(torch.logical_or((epe3d_map_noc < 0.05), (epe3d_map_noc/torch.sqrt(torch.sum(flow_3d_target ** 2, dim=0))[flow_3d_mask] < 0.05))).item()
        metrics_3d_noc['10cm'] += torch.count_nonzero(torch.logical_or((epe3d_map_noc < 0.1), (epe3d_map_noc/torch.sqrt(torch.sum(flow_3d_target ** 2, dim=0))[flow_3d_mask] < 0.1))).item()
        metrics_3d_noc['outlier'] += torch.count_nonzero(torch.logical_or((epe3d_map_noc > 0.3), (epe3d_map_noc/torch.sqrt(torch.sum(flow_3d_target ** 2, dim=0))[flow_3d_mask] > 0.1))).item()

    stop = timeit.default_timer()
    print('in-loop Time: ', (stop - start) / len(val_dataset))

    print('#### 3D Metrics ####')
    results['EPE'] = metrics_3d['EPE3d'] / metrics_3d['counts']
    results['NEPE'] = metrics_3d['NEPE3d'] / metrics_3d['counts']
    results['5cm'] = metrics_3d['5cm'] / metrics_3d['counts'] * 100.0
    results['10cm'] = metrics_3d['10cm'] / metrics_3d['counts'] * 100.0
    results['outlier'] = metrics_3d['outlier'] / metrics_3d['counts'] * 100.0
    print("Validation Things EPE: %.4f, NEPE: %.4f, 5cm: %.4f, 10cm: %.4f, outlier: %.4f" %
          (results['EPE'], results['NEPE'], results['5cm'], results['10cm'], results['outlier']))

    print('#### 3D Metrics non-occluded ####')
    results['EPE_non-occluded'] = metrics_3d_noc['EPE3d'] / metrics_3d_noc['counts']
    results['NEPE_non-occluded'] = metrics_3d_noc['NEPE3d'] / metrics_3d_noc['counts']
    results['5cm_non-occluded'] = metrics_3d_noc['5cm'] / metrics_3d_noc['counts'] * 100.0
    results['10cm_non-occluded'] = metrics_3d_noc['10cm'] / metrics_3d_noc['counts'] * 100.0
    results['outlier_non-occluded'] = metrics_3d_noc['outlier'] / metrics_3d_noc['counts'] * 100.0
    print("Validation Things EPE: %.4f, NEPE: %.4f, 5cm: %.4f, 10cm: %.4f, outlier: %.4f" %
          (results['EPE_non-occluded'], results['NEPE_non-occluded'], results['5cm_non-occluded'], results['10cm_non-occluded'], results['outlier_non-occluded']))

    return results


@torch.no_grad()
def validate_fluidflow3d(stage,
                   model,
                   ):
    model.eval()

    if stage == 'fluidflow3d':
        val_dataset = Fluidflow3D(partition='test')
    if stage == 'things_flownet3d':
        val_dataset = KITTI_flownet3d(split='training150')
    if stage == 'things_subset':
        val_dataset = KITTI_hplflownet()
    if stage == 'things_subset_non-occluded':
        val_dataset = KITTI_hplflownet()

    print('Number of validation image pairs: %d' % len(val_dataset))

    epe_list = []
    results = {}

    metrics_3d = {'counts': 0, 'EPE3d': 0.0, '5cm': 0.0, '10cm': 0.0, 'outlier': 0.0}

    import timeit
    start = timeit.default_timer()
    for val_id in range(len(val_dataset)):

        data_dict = val_dataset[val_id]
        pcs = data_dict['pcs'].unsqueeze(0) # 8192*6
        flow_3d = data_dict['flow_3d'].unsqueeze(0).cuda()
        pc1 = pcs[:,:,0:3].cuda()
        pc2 = pcs[:,:,3:6].cuda()
        flow_3d_target = flow_3d[:,:3,:].cuda()


        results_dict_point = model(pc0 = pc1, pc1 = pc2)
        flow_3d_pred = results_dict_point['flow_preds'][-1]

        epe3d_map = torch.sqrt(torch.sum((flow_3d_pred[0] - flow_3d_target[0]) ** 2, dim=0))

        # save testing images
        # np.save('results/'+format(val_id, '04d')+'_pc1', pc1[0].cpu())
        # np.save('results/'+format(val_id, '04d')+'_pc2', pc2[0].cpu())
        np.save('../results/'+format(val_id, '04d')+'_flow_3d_pred', torch.permute(flow_3d_pred[0], (1,0)).cpu())
        # np.save('results/'+format(val_id, '04d')+'_flow_3d_target', torch.permute(flow_3d_target[0], (1,0)).cpu())

        if flow_3d_target[0].shape[0] > 3:
            flow_3d_mask = flow_3d_target[0][3] > 0
            flow_3d_target = flow_3d_target[0][:3]
        else:
            flow_3d_mask = torch.ones(flow_3d_target[0].shape[1], dtype=torch.int64).cuda()
            flow_3d_target = flow_3d_target[0]

        flow_3d_mask = torch.logical_and(flow_3d_mask, torch.logical_not(torch.isnan(epe3d_map)))
        metrics_3d['counts'] += epe3d_map[flow_3d_mask].shape[0]
        metrics_3d['EPE3d'] += epe3d_map[flow_3d_mask].sum().item()

        # Fluidflow3D
        metrics_3d['5cm'] += torch.count_nonzero(torch.logical_or((epe3d_map[flow_3d_mask] < 0.05), (epe3d_map[flow_3d_mask]/torch.sqrt(torch.sum((flow_3d_target) ** 2, dim=0))[flow_3d_mask] < 0.05))).item()
        metrics_3d['10cm'] += torch.count_nonzero(torch.logical_or((epe3d_map[flow_3d_mask] < 0.1), (epe3d_map[flow_3d_mask]/torch.sqrt(torch.sum((flow_3d_target) ** 2, dim=0))[flow_3d_mask] < 0.1))).item()
        metrics_3d['outlier'] += torch.count_nonzero(torch.logical_or((epe3d_map[flow_3d_mask] > 0.3), (epe3d_map[flow_3d_mask]/torch.sqrt(torch.sum((flow_3d_target) ** 2, dim=0))[flow_3d_mask] > 0.1))).item()

    stop = timeit.default_timer()
    print('in-loop Time: ', (stop - start)/len(val_dataset))

    print('#### 3D Metrics ####')
    results['EPE'] = metrics_3d['EPE3d'] / metrics_3d['counts']
    results['5cm'] = metrics_3d['5cm'] / metrics_3d['counts'] * 100.0
    results['10cm'] = metrics_3d['10cm'] / metrics_3d['counts'] * 100.0
    results['outlier'] = metrics_3d['outlier'] / metrics_3d['counts'] * 100.0
    print("Validation Fluidflow3D EPE: %.4f, 5cm: %.4f, 10cm: %.4f, outlier: %.4f" % (results['EPE'], results['5cm'], results['10cm'], results['outlier']))

    return results
