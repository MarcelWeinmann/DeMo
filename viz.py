import os
import torch
import hydra
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate, to_absolute_path
import pytorch_lightning as pl
import numpy as np

def move_to_device(obj, device):
    """Recursively moves all tensors in a nested structure to the target device."""
    if torch.is_tensor(obj):
        return obj.to(device)
    elif isinstance(obj, dict):
        return {k: move_to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [move_to_device(v, device) for v in obj]
    elif isinstance(obj, tuple):
        return tuple(move_to_device(v, device) for v in obj)
    else:
        return obj 

def local_to_global(points, origin, theta):
    """
    Transforms local coordinates back to global coordinates.
    points: (..., 2) numpy array of coordinates
    origin: (2,) numpy array of the [x, y] translation origin
    theta: scalar float, heading/rotation angle in radians
    """
    c, s = np.cos(theta), np.sin(theta)
    rot_mat = np.array([[c, -s], 
                        [s,  c]])
    return np.dot(points, rot_mat.T) + origin

@hydra.main(version_base=None, config_path="./conf/", config_name="config")
def main(conf):
    pl.seed_everything(conf.seed)
    output_dir = HydraConfig.get().runtime.output_dir
    
    ckpt_path = to_absolute_path("/dev_ws/src/tam_deep_prediction/models/DeMo/DeMo/outputs/change_av2_default_baseline_default/20260625-170328/checkpoints/last.ckpt")
    assert os.path.exists(ckpt_path), f"Checkpoint {ckpt_path} does not exist"

    os.system('cp -a %s %s' % ('conf', output_dir))
    
    datamodule: pl.LightningDataModule = instantiate(conf.datamodule.target, test=conf.test)
    datamodule.setup(stage="test" if conf.test else "validate")
    dataloader = datamodule.test_dataloader() if conf.test else datamodule.val_dataloader()

    model = instantiate(conf.model.target)
    
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)
        
    device = torch.device("cuda" if torch.cuda.is_available() and getattr(conf, 'gpus', 0) > 0 else "cpu")
    model.to(device)
    model.eval()

    if getattr(conf, 'local_rank', 0) == 0:
        with open(os.path.join(output_dir, "model.log"), "w") as f:
            print(model.net, file=f)

    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    print(f"[*] Saving visualizations to: {viz_dir}")

    is_test = conf.get('test', False)
    limit_batches = conf.get('limit_test_batches') if is_test else conf.get('limit_val_batches')
    
    if limit_batches is None:
        print("[!] No batch limit found in config. Defaulting to max 1000 batches.")
        limit_batches = 1000

    total_ade = 0.0
    total_fde = 0.0
    total_samples = 0

    print(f"[*] Starting inference and plotting... (Limit: {limit_batches} batches)")
    with torch.no_grad():
        for batch_idx, data in enumerate(dataloader):
            if limit_batches is not None and batch_idx >= limit_batches:
                break
                
            data = move_to_device(data, device)
            input_data = data[-1] if isinstance(data, list) else data

            out = model(input_data)
            preds = out.get('y_hat', out.get('new_y_hat')).cpu().numpy()
            
            if input_data['target'].ndim == 4: # [batch, agents, horizon, coords]
                targets = input_data['target'][:, 0].cpu().numpy()
            else:
                targets = input_data['target'].cpu().numpy()
                
            batch_size = targets.shape[0]
            origins = input_data.get('origin', np.zeros((batch_size, 2)))
            thetas = input_data.get('theta', np.zeros((batch_size,)))
            
            if torch.is_tensor(origins): origins = origins.cpu().numpy()
            if torch.is_tensor(thetas): thetas = thetas.cpu().numpy()

            for i in range(batch_size):
                fig, (ax_spatial, ax_vel) = plt.subplots(1, 2, figsize=(16, 7))
                origin = origins[i]
                theta = thetas[i]

                # --- 1. Reconstruct History from DeMo Keys ---
                # Added 'x_angles' to the check
                has_demo_keys = all(k in input_data for k in ['x_centers', 'x_positions_diff', 'x_valid_mask', 'x_angles'])
                
                if has_demo_keys:
                    current_positions = input_data["x_centers"][i].cpu().numpy()       # [Num_Agents, 2]
                    current_angles = input_data["x_angles"][i, :, -1].cpu().numpy()    # [Num_Agents] (heading at t=0)
                    hist_diff = input_data["x_positions_diff"][i].cpu().numpy()        # [Num_Agents, Timesteps, 2]
                    hist_mask = input_data["x_valid_mask"][i].cpu().numpy().astype(bool) # [Num_Agents, Timesteps]
                    
                    hist_path = np.cumsum(hist_diff, axis=1)
                    offset = current_positions - hist_path[:, -1, :] 
                    hist_local_abs = hist_path + offset[:, None, :]

                    # Focal Agent History (Index 0)
                    focal_mask = hist_mask[0]
                    if np.any(focal_mask):
                        focal_hist_local = hist_local_abs[0][focal_mask]
                        focal_hist_global = local_to_global(focal_hist_local, origin, theta)
                        hist_x, hist_y = focal_hist_global[:, 0], focal_hist_global[:, 1]
                        
                        ax_spatial.plot(hist_x, hist_y, color='blue', linestyle='-', linewidth=2.5, label='Focal History')
                        ax_spatial.plot(hist_x[-1], hist_y[-1], 'bo', markersize=8, label='Focal Pos (t=0)')
                        
                        if len(hist_x) > 1:
                            hist_vx, hist_vy = np.diff(hist_x), np.diff(hist_y)
                            hist_speed = np.sqrt(hist_vx**2 + hist_vy**2) * 10.0 
                            t_hist = np.arange(-len(hist_speed), 0)
                            ax_vel.plot(t_hist, hist_speed, color='blue', marker='o', markersize=4, label='Focal History Speed')

                    # Other Agents Rotated Bounding Boxes (Indices 1 to N)
                    if len(current_positions) > 1:
                        valid_other_mask = hist_mask[1:, -1]
                        valid_other_local = current_positions[1:][valid_other_mask]
                        valid_other_angles = current_angles[1:][valid_other_mask]
                        
                        if len(valid_other_local) > 0:
                            valid_other_global = local_to_global(valid_other_local, origin, theta)
                            
                            # The global heading is the agent's local heading + the focal agent's global theta
                            valid_other_global_angles = valid_other_angles + theta
                            
                            box_w, box_l = 2.0, 4.5 
                            
                            for (ox, oy), o_theta in zip(valid_other_global, valid_other_global_angles):
                                # Calculate direction vectors
                                cos_t, sin_t = np.cos(o_theta), np.sin(o_theta)
                                
                                # Forward and Left vectors from the center point
                                fx, fy = (box_l / 2) * cos_t, (box_l / 2) * sin_t
                                lx, ly = -(box_w / 2) * sin_t, (box_w / 2) * cos_t
                                
                                # Calculate the 4 corners of the rotated polygon
                                corners = [
                                    (ox + fx + lx, oy + fy + ly), # Front-Left
                                    (ox + fx - lx, oy + fy - ly), # Front-Right
                                    (ox - fx - lx, oy - fy - ly), # Back-Right
                                    (ox - fx + lx, oy - fy + ly), # Back-Left
                                ]
                                
                                poly = patches.Polygon(corners, closed=True, linewidth=1.5, 
                                                       edgecolor='gray', facecolor='lightgray', alpha=0.6)
                                ax_spatial.add_patch(poly)
                                
                            ax_spatial.plot([], [], 's', color='lightgray', markeredgecolor='gray', label='Other Agents (t=0)')

                # --- 2. Ground Truth ---
                gt_local = targets[i, :, :2]
                gt_global = local_to_global(gt_local, origin, theta)
                gt_x, gt_y = gt_global[:, 0], gt_global[:, 1]
                
                gt_vx, gt_vy = np.diff(gt_x), np.diff(gt_y)
                gt_speed = np.sqrt(gt_vx**2 + gt_vy**2) * 10.0
                t_future = np.arange(1, len(gt_speed) + 1)
                
                ax_spatial.plot(gt_x, gt_y, 'g-', label='Focal Ground Truth', linewidth=2)
                ax_vel.plot(t_future, gt_speed, 'g-', marker='s', markersize=4, label='Focal GT Speed')

                # --- 3. Predictions & Metrics ---
                best_ade = float('inf')
                best_fde = float('inf')

                if len(preds.shape) > 3:  
                    num_modes = preds.shape[1]
                    for m in range(num_modes):
                        pred_local = preds[i, m, :, :2]
                        pred_global = local_to_global(pred_local, origin, theta)
                        
                        errors = np.linalg.norm(pred_global - gt_global, axis=-1)
                        mode_ade = np.mean(errors)
                        mode_fde = errors[-1]
                        
                        if mode_ade < best_ade:
                            best_ade = mode_ade
                            best_fde = mode_fde
                        
                        pred_x, pred_y = pred_global[:, 0], pred_global[:, 1]
                        pred_vx, pred_vy = np.diff(pred_x), np.diff(pred_y)
                        pred_speed = np.sqrt(pred_vx**2 + pred_vy**2) * 10.0
                        
                        if m == 0: 
                            ax_spatial.plot(pred_x, pred_y, 'r--', linewidth=2, label='Primary Pred')
                            ax_vel.plot(t_future, pred_speed, 'r--', marker='^', markersize=4, label='Primary Pred Speed')
                        else: 
                            ax_spatial.plot(pred_x, pred_y, 'r--', alpha=0.3)
                            ax_vel.plot(t_future, pred_speed, 'r--', alpha=0.3)
                else: 
                    pred_local = preds[i, :, :2]
                    pred_global = local_to_global(pred_local, origin, theta)
                    
                    errors = np.linalg.norm(pred_global - gt_global, axis=-1)
                    best_ade = np.mean(errors)
                    best_fde = errors[-1]
                    
                    pred_x, pred_y = pred_global[:, 0], pred_global[:, 1]
                    ax_spatial.plot(pred_x, pred_y, 'r--', label='Focal Prediction', linewidth=2)
                    
                    pred_vx, pred_vy = np.diff(pred_x), np.diff(pred_y)
                    pred_speed = np.sqrt(pred_vx**2 + pred_vy**2) * 10.0
                    ax_vel.plot(t_future, pred_speed, 'r--', marker='^', markersize=4, label='Pred Speed')

                total_ade += best_ade
                total_fde += best_fde
                total_samples += 1

                # --- 4. Plot Formatting ---
                ax_spatial.set_title(f'Global Spatial Traj | Sample {i}\nminADE: {best_ade:.3f}m | minFDE: {best_fde:.3f}m')
                ax_spatial.set_xlabel('Global X Coordinate (m)')
                ax_spatial.set_ylabel('Global Y Coordinate (m)')
                ax_spatial.grid(True, linestyle=':', alpha=0.6)
                ax_spatial.legend(loc='best')
                ax_spatial.axis('equal') 
                
                ax_vel.set_title(f'Velocity Profile | Sample {i}')
                ax_vel.set_xlabel('Time Step (t=0 is present)')
                ax_vel.set_ylabel('Speed (m/s)')
                ax_vel.grid(True, linestyle=':', alpha=0.6)
                ax_vel.legend(loc='best')

                plt.suptitle(f'Batch {batch_idx}', fontsize=16)
                plt.tight_layout()
                file_name = f'viz_batch_{batch_idx:04d}_sample_{i:02d}.png'
                plt.savefig(os.path.join(viz_dir, file_name), bbox_inches='tight', dpi=150)
                plt.close(fig)
                                
    if total_samples > 0:
        avg_ade = total_ade / total_samples
        avg_fde = total_fde / total_samples
        print("="*40)
        print("EVALUATION RESULTS (GLOBAL COORDINATES)")
        print(f"Total Samples: {total_samples}")
        print(f"Average minADE: {avg_ade:.4f} meters")
        print(f"Average minFDE: {avg_fde:.4f} meters")
        print("="*40)

    print("[*] Visualization complete!")

if __name__ == "__main__":
    main()