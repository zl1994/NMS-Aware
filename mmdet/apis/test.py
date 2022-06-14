import os.path as osp
import pickle
import shutil
import tempfile
import time
import numpy as np
import mmcv
import torch
import torch.distributed as dist
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info

from mmdet.core import encode_mask_results


def single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    show_score_thr=0.3):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    gt_list = []
    pred_list = []
    imp_iou_list = []

    delta_tl_x_list = []
    delta_br_x_list = []
    delta_tl_y_list = []
    delta_br_y_list = []
    tl_scores_list = []
    br_scores_list = []
    tl_offs_list = []
    br_offs_list = []
    dense_bboxes = np.load("/mnt/truenas/scratch/li.zhu/mmdetection_sparse/mmdetection/dense_bboxes.npy")
    dense_bboxes = torch.from_numpy(dense_bboxes)
    
    for i, data in enumerate(data_loader):
        if i<81:
            """
            with torch.no_grad():
                dense_bbox = dense_bboxes[1000*i:1000*(i+1)]
                #result, delta_tl_x, delta_tl_y, delta_br_x, delta_br_y, tl_scores, br_scores, tl_offs, br_offs = model(return_loss=False, rescale=True, dense_bboxes=dense_bbox, **data)
                result = model(return_loss=False, rescale=True, **data)
                #result, side_distance, side_confids = model(return_loss=False, rescale=True, **data)
                #result, gt, pred, imp_iou = model(return_loss=False, rescale=True, **data)
            '''
            tl_scores_list.append(tl_scores.cpu())
            br_scores_list.append(br_scores.cpu())
            tl_offs_list.append(tl_offs.cpu())
            br_offs_list.append(br_offs.cpu())
            
            if delta_tl_x is not None:
                delta_tl_x_list.append(delta_tl_x)
                delta_br_x_list.append(delta_br_x)
                delta_tl_y_list.append(delta_tl_y)
                delta_br_y_list.append(delta_br_y)
            '''
            '''
            if side_distance is not None:
                delta_tl_x_list.append(side_distance[:, 0])
                delta_br_x_list.append(side_distance[:, 2])
                delta_tl_y_list.append(side_distance[:, 1])
                delta_br_y_list.append(side_distance[:, 3])
                tl_scores_list.append(side_confids[:, 0])
                br_scores_list.append(side_confids[:, 2])
            '''
            '''
            if gt is not None:
                gt_list.append(gt)
                pred_list.append(pred)
                imp_iou_list.append(imp_iou)
            '''
            """
            with torch.no_grad():
                result, dis, iou, bbox_pred, max_scores, node_feats = model(return_loss=False, rescale=True, **data)
                np.save("max_scores_p3.npy", max_scores[0].cpu().numpy())
                np.save("max_scores_p4.npy", max_scores[1].cpu().numpy())
                np.save("max_scores_p5.npy", max_scores[2].cpu().numpy())
                np.save("max_scores_p6.npy", max_scores[3].cpu().numpy())
                np.save("max_scores_p7.npy", max_scores[4].cpu().numpy())

                np.save("bbox_pred_p3.npy", bbox_pred[0].cpu().numpy())
                np.save("bbox_pred_p4.npy", bbox_pred[1].cpu().numpy())
                np.save("bbox_pred_p5.npy", bbox_pred[2].cpu().numpy())
                np.save("bbox_pred_p6.npy", bbox_pred[3].cpu().numpy())
                np.save("bbox_pred_p7.npy", bbox_pred[4].cpu().numpy())
                np.save("loc_p3.npy", dis[0].cpu().numpy())
                np.save("loc_p4.npy", dis[1].cpu().numpy())
                np.save("loc_p5.npy", dis[2].cpu().numpy())
                np.save("loc_p6.npy", dis[3].cpu().numpy())
                np.save("loc_p7.npy", dis[4].cpu().numpy())
                np.save("iou_p3.npy", iou[0].cpu().numpy())
                np.save("iou_p4.npy", iou[1].cpu().numpy())
                np.save("iou_p5.npy", iou[2].cpu().numpy())
                np.save("iou_p6.npy", iou[3].cpu().numpy())
                np.save("iou_p7.npy", iou[4].cpu().numpy())

                np.save("reg_feat_p3.npy", node_feats[0].cpu().numpy())
                np.save("reg_feat_p4.npy", node_feats[1].cpu().numpy())
                np.save("reg_feat_p5.npy", node_feats[2].cpu().numpy())
                np.save("reg_feat_p6.npy", node_feats[3].cpu().numpy())
                np.save("reg_feat_p7.npy", node_feats[4].cpu().numpy())
            
            batch_size = len(result)
            if show or out_dir:
                if batch_size == 1 and isinstance(data['img'][0], torch.Tensor):
                    img_tensor = data['img'][0]
                else:
                    img_tensor = data['img'][0].data[0]
                img_metas = data['img_metas'][0].data[0]
                imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
                assert len(imgs) == len(img_metas)

                for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                    h, w, _ = img_meta['img_shape']
                    img_show = img[:h, :w, :]

                    ori_h, ori_w = img_meta['ori_shape'][:-1]
                    img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                    if out_dir:
                        out_file = osp.join(out_dir, img_meta['ori_filename'])
                    else:
                        out_file = None

                    model.module.show_result(
                        img_show,
                        result[i],
                        show=show,
                        out_file=out_file,
                        score_thr=0.3)

            # encode mask results
            if isinstance(result[0], tuple):
                result = [(bbox_results, encode_mask_results(mask_results))
                        for bbox_results, mask_results in result]
            results.extend(result)

            for _ in range(batch_size):
                prog_bar.update()
        else:
            break
    '''
    gts   = torch.cat(gt_list, dim=0)
    preds = torch.cat(pred_list, dim=0)
    imp_ious = torch.cat(imp_iou_list, dim=0)
    print(imp_ious.mean())
    np.save("gts.npy", gts.cpu().numpy())
    np.save("preds.npy", preds.cpu().numpy())
    np.save("imp_ious.npy", imp_ious.cpu().numpy())
    '''
    '''
    delta_tl_x = torch.cat(delta_tl_x_list, dim=0)
    delta_br_x = torch.cat(delta_br_x_list, dim=0)
    delta_tl_y = torch.cat(delta_tl_y_list, dim=0)
    delta_br_y = torch.cat(delta_br_y_list, dim=0)
    
    tl_scores  = torch.stack(tl_scores_list, dim=0)
    br_scores  = torch.stack(br_scores_list, dim=0)
    tl_offs  = torch.stack(tl_offs_list, dim=0)
    br_offs  = torch.stack(br_offs_list, dim=0)
    
    np.save("delta_tl_x.npy", delta_tl_x.cpu().numpy())
    np.save("delta_br_x.npy", delta_br_x.cpu().numpy())
    np.save("delta_tl_y.npy", delta_tl_y.cpu().numpy())
    np.save("delta_br_y.npy", delta_br_y.cpu().numpy())

    np.save("tl_scores.npy", tl_scores.numpy())
    np.save("br_scores.npy", br_scores.numpy())
    np.save("tl_offs.npy", tl_offs.numpy())
    np.save("br_offs.npy", br_offs.numpy())
    '''
    return results


def multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.

    Returns:
        list: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
            # encode mask results
            if isinstance(result[0], tuple):
                result = [(bbox_results, encode_mask_results(mask_results))
                          for bbox_results, mask_results in result]
        results.extend(result)

        if rank == 0:
            batch_size = len(result)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results


def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            mmcv.mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def collect_results_gpu(result_part, size):
    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_list.append(
                pickle.loads(recv[:shape[0]].cpu().numpy().tobytes()))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results
