import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import sys

isaaclab_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, isaaclab_root)

def plot_multiple(all_results, dt=0.1, center=np.array([0, 0])):
    # Make sure output folder exists
    os.makedirs("plots", exist_ok=True)


    euler_idx = [0, 1, 2]        # roll, pitch, yaw
    pos_idx = [3, 4, 5]          # x, y, z
    vel_idx = [6, 7, 8]          # vx, vy, vz
    ang_vel_idx = [9, 10, 11]    # wx, wy, wz
    moments_idx = [3, 4, 5]      # mx, my, mz
    forces_idx = [0, 1, 2]      # tx, ty, tz
    rewards_idx = [0]            # rewards

    groups = {
        "Position": pos_idx,
        "Velocity": vel_idx,
        "Euler Angles": euler_idx,
        "Angular Velocities": ang_vel_idx,
        "Moments": moments_idx,
        "Forces": forces_idx,
        "Rewards": rewards_idx,
        "3D Plot": pos_idx,
    }

    for group_name, indices in groups.items():

        if group_name == "3D Plot":
            fig = plt.figure(figsize=(4, 6)) 
            ax = fig.add_subplot(111, projection='3d')
            ax.set_box_aspect((1, 1, 3))
            ax.view_init(elev=30, azim=45)
            plt.tight_layout()

            for run_idx, run in enumerate(all_results):
                states = run['states']  # shape [T, state_dim]
                if states.shape[0] > 500:
                    continue
                # Extract positions
                x = states[:, 4]
                y = states[:, 5]
                z = states[:, 6]

                ax.plot(x + center[0], y + center[1], z, label=f'Run {run_idx+1}')

            ax.locator_params(axis='x', nbins=3)
            ax.locator_params(axis='y', nbins=3)
            # for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            #     axis.pane.set_visible(False)
            #     axis.set_tick_params(pad=6)
            ax.set_xlabel('X [m]')
            ax.set_ylabel('Y [m]')
            ax.zaxis.set_rotate_label(False) 
            ax.set_zlabel('Z [m]', rotation=90)
            # ax.set_title('3D Trajectory Plot')
            ax.set_ylim([center[1]-30, center[1]+30])
            ax.set_xlim([center[0]-30, center[0]+30])
            plt.savefig(f"plots/3D_Trajectory.png", dpi=300)
            
            plt.close()
            continue
        elif group_name == "Rewards":
            fig, ax = plt.subplots(figsize=(8, 6))
            for run_idx, run in enumerate(all_results):
                rewards = run['rewards']  # shape [T, ]
                if rewards.shape[0] > 500:
                    continue
                timesteps = np.arange(rewards.shape[0]) * dt
                ax.plot(timesteps[0:-1], rewards[0:-1], label=f'Run {run_idx+1}')
                ax.set_xlabel('Time [s]')
                ax.set_ylabel(f'{group_name}')
                ax.grid(True)
            # plt.suptitle(group_name)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(f"plots/{group_name}.png", dpi=300)
            plt.close()
        else:
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))  # 1 row, 3 columns
            axes = axes.flatten()

            for run_idx, run in enumerate(all_results):
                states = run['states']  # shape [T, state_dim]
                if states.shape[0] > 500:
                    continue
                # states[:, 4] += 10  # shift x
                # states[:, 5] += -10  # shift y
                actions = run['actions']  # shape [T, action_dim]
                rewards = run['rewards']  # shape [T, ]

                # Compute Euler angles from quaternion if needed
                if states.shape[1] >= 4:
                    w = states[:, 0]
                    x = states[:, 1]
                    y = states[:, 2]
                    z = states[:, 3]

                    t0 = 2.0 * (w * x + y * z)
                    t1 = 1.0 - 2.0 * (x * x + y * y)
                    roll = np.arctan2(t0, t1)

                    t2 = 2.0 * (w * y - z * x)
                    t2 = np.clip(t2, -1.0, 1.0)
                    pitch = np.arcsin(t2)

                    t3 = 2.0 * (w * z + x * y)
                    t4 = 1.0 - 2.0 * (y * y + z * z)
                    yaw = np.arctan2(t3, t4)

                    euler_angles = np.stack([roll*180/np.pi, pitch*180/np.pi, yaw*180/np.pi], axis=1)
                    if states.shape[1] in [8, 14]:
                        states = np.concatenate((euler_angles, states[:, 4:]), axis=1)
                    states[:, -3:] = states[:, -3:] * 180/np.pi  # angular velocities to deg/s

                timesteps = np.arange(states.shape[0]) * dt

                for i, idx in enumerate(indices):
                    axis = 'X' if i == 0 else 'Y' if i == 1 else 'Z'
                    unit = '[m]' if group_name == "Position" else '[m/s]' if group_name == "Velocity" else '[deg]' if group_name == "Euler Angles" else '[deg/s]' if group_name == "Angular Velocities" else '[N]' if group_name == "Forces" else '[Nm]' if group_name == "Moments" else ''
                    if group_name == "Forces" or group_name == "Moments":
                        axes[i].plot(timesteps, actions[:, idx])
                    # elif group_name == "Rewards":
                    #     axes[i].plot(timesteps, rewards)
                    else:
                        axes[i].plot(timesteps, states[:, idx])
                    axes[i].set_xlabel('Time [s]')
                    axes[i].set_ylabel(f'{axis} {group_name} {unit}')
                    axes[i].grid(True)

            # plt.suptitle(group_name)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(f"plots/{group_name}.png", dpi=300)
            plt.close()

def plot_landing(all_results, dt=0.1, center=np.array([0, 0])):
    import matplotlib.patches as patches

    pos_idx = [4, 5]   # x, y

    # Collect all landing positions to compute axis limits
    # landing_points = []
    # for run in all_results:
    #     loc = run["states"][-1, pos_idx] + center  # shift landing zone center to (10,10)
    #     landing_points.append(loc)

    # landing_points = np.array(landing_points)
    # max_val = float(np.max(np.abs(landing_points)))   # highest |x| or |y|
    # axis_limit = max_val + 0.5         # always ≥3, add margin

    fig, ax = plt.subplots(figsize=(8, 6))

    g_count = 0
    y_count = 0
    o_count = 0
    r_count = 0
    for run_idx, run in enumerate(all_results):
        # if run["states"].size > 500:
        #             continue
        loc = run["states"][-1, pos_idx]
        if np.linalg.norm(run["states"][-1,7:9]) > 0.3 and np.linalg.norm(loc) < 2.0:
            ax.plot(loc[0] + center[0], loc[1] + center[1], marker='o', color='orange',fillstyle='none', markersize=8, markeredgewidth=1.5)
            print(f"Run {run_idx+1} - X Velocity: {run['states'][-1,7]:.3f} m/s, Y Velocity: {run['states'][-1,8]:.3f} m/s, Z Velocity: {run['states'][-1,9]:.3f} m/s, Norm Velocity: {np.linalg.norm(run['states'][-1,7:9]):.3f} m/s")
            o_count += 1
        elif np.linalg.norm(run["states"][-1,7:9]) < 0.3 and np.linalg.norm(loc) > 2.0:
            ax.plot(loc[0] + center[0], loc[1] + center[1], marker='o', color='yellow',fillstyle='none', markersize=8, markeredgewidth=1.5)
            y_count += 1
        elif np.linalg.norm(run["states"][-1,7:9]) > 0.3 and  np.linalg.norm(loc) > 2.0:
            ax.plot(loc[0] + center[0], loc[1] + center[1], marker='x', color='red')
            r_count += 1
        else:
            ax.plot(loc[0] + center[0], loc[1] + center[1], marker='o', color='green',fillstyle='none', markersize=8, markeredgewidth=1.5)
            g_count += 1
    print(f"Green: {g_count}, Yellow: {y_count}, Orange: {o_count}, Red: {r_count}")

    # Target site
    ax.plot(center[0], center[1], 'b*', markersize=15)

    # Light grey landing-zone circle
    landing_zone = patches.Circle(
        (center[0],center[1]),
        radius=2.0,
        linewidth=1,
        alpha=0.5,
        color='lightgrey',
    )
    ax.add_patch(landing_zone)

    # Dotted reference lines at X=0 and Y=0
    ax.axhline(center[1], color='black', linestyle=':', linewidth=1)
    ax.axvline(center[0], color='black', linestyle=':', linewidth=1)

    # Axes limits (auto-expanded but minimum ±3)
    ax.set_xlim(left=center[0]-5, right=center[0]+5)
    ax.set_ylim(bottom=center[1]-5, top=center[1]+5)

    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_aspect('equal')

    plt.savefig("plots/Landing_Zone.png", dpi=300)
    plt.close()

def main():
    # tensor = torch.load("logs/IsaacLab/lander_6dof_direct/20251128_143813/play_results_20251216_205122.pt", weights_only=False, map_location="cpu") # origin
    tensor = torch.load("logs/IsaacLab/lander_6dof_direct/20251128_143813/play_results_20251218_014641.pt", weights_only=False, map_location="cpu") # offset 5,-5
    all_results = tensor

    dt = 0.1

    center = np.array([5, -5])
    plot_multiple(all_results, dt, center)
    plot_landing(all_results, dt, center)

if __name__ == "__main__":
    main()