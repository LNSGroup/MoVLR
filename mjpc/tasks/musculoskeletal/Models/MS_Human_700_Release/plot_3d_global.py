import torch
import matplotlib.pyplot as plt
import numpy as np
import io
import matplotlib
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d.axes3d as p3
from textwrap import wrap
import imageio


def scaling(data):
    smpl_kinetic_chain = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 12], [0, 14, 17, 19, 21], [0, 13, 16, 18, 20]]
    lengths = [
        [0.1236801353330437, 0.3993810723986291, 0.4001249804748512, 0.1375286849307577], 
        [0.1236801353330437, 0.3993810723986291, 0.4001249804748512, 0.1375286849307577], 
        [0.49682246931739826], 
        [0.44129287331515255, 0.13776026286632892, 0.32226146821983404, 0.2582364948564625], 
        [0.44129287331515255, 0.13776026286632892, 0.32226146821983404, 0.2582364948564625]
        ]
    unused_indices = [3, 6, 9, 15]

    def scale(frame, smpl_kinetic_chain, lengths):
        new_frame = np.zeros(frame.shape)
        new_frame[0] = frame[0]


        for i in range(len(smpl_kinetic_chain)):
            v = []

            for j in range(len(smpl_kinetic_chain[i])-1):
                diff = frame[smpl_kinetic_chain[i][j+1]] - frame[smpl_kinetic_chain[i][j]]
                v.append(diff / np.linalg.norm(diff))
                new_frame[smpl_kinetic_chain[i][j+1]] = new_frame[smpl_kinetic_chain[i][j]] + lengths[i][j] * v[j]


        new_frame[unused_indices] = frame[0]
        return new_frame


    for i in range(data.shape[0]):
        frame = data[i]
        # print(frame.shape)
        data[i] = scale(frame, smpl_kinetic_chain, lengths)
    
    return data

def plot_3d_motion(args, figsize=(10, 10), fps=120, radius=4):
    matplotlib.use('Agg')

    joints, out_name, title = args
    
    title_sp = title.split(' ')
    if len(title_sp) > 20:
        title = '\n'.join([' '.join(title_sp[:10]), ' '.join(title_sp[10:20]), ' '.join(title_sp[20:])])
    elif len(title_sp) > 10:
        title = '\n'.join([' '.join(title_sp[:10]), ' '.join(title_sp[10:])])

    data = joints.copy().reshape(len(joints), -1, 3)
    scaling(data)
    
    nb_joints = joints.shape[1]
    # smpl_kinetic_chain = [
    #     [0, 11, 12, 13, 14, 15], [0, 16, 17, 18, 19, 20], [0, 1, 2, 3, 4],
    #     [3, 5, 6, 7], [3, 8, 9, 10]
    # ] if nb_joints == 21 else [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10],
    #                            [0, 3, 6, 9, 12, 15], [9, 14, 17, 19, 21],
    #                            [9, 13, 16, 18, 20]]
    smpl_kinetic_chain = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 12], [0, 14, 17, 19, 21], [0, 13, 16, 18, 20]]

    
    limits = 1000 if nb_joints == 21 else 2

    MINS = data.min(axis=0).min(axis=0)
    MAXS = data.max(axis=0).max(axis=0)

    colors = [
        'red', 'blue', 'black', 'red', 'blue', 'darkblue', 'darkblue',
        'darkblue', 'darkblue', 'darkblue', 'darkred', 'darkred', 'darkred',
        'darkred', 'darkred'
    ]
    frame_number = data.shape[0]

    height_offset = MINS[1]
    data[:, :, 1] -= height_offset
    trajec = data[:, 0, [0, 2]]

    data[..., 0] -= data[:, 0:1, 0]
    data[..., 2] -= data[:, 0:1, 2]

    def update(index):
        def init():
            ax.set_xlim3d([-radius / 2, radius / 2])
            ax.set_ylim3d([0, radius])
            ax.set_zlim3d([0, radius])
            ax.grid(b=False)

        def plot_xzPlane(minx, maxx, miny, minz, maxz):
            ## Plot a plane XZ
            verts = [[minx, miny, minz], [minx, miny, maxz],
                     [maxx, miny, maxz], [maxx, miny, minz]]
            xz_plane = Poly3DCollection([verts])
            xz_plane.set_facecolor((0.5, 0.5, 0.5, 0.5))
            ax.add_collection3d(xz_plane)

        fig = plt.figure(figsize=(480 / 96., 320 / 96.),
                         dpi=96) if nb_joints == 21 else plt.figure(
                             figsize=(10, 10), dpi=96)
        # fig.tight_layout()
        if title is not None:
            wraped_title = '\n'.join(wrap(title, 40))
            fig.suptitle(wraped_title, fontsize=16)
        ax = p3.Axes3D(fig, auto_add_to_figure=False)
        fig.add_axes(ax)

        init()

        # ax.lines = []
        # ax.collections = []
        ax.view_init(elev=110, azim=-90)
        ax.dist = 7.5
        #         ax =
        plot_xzPlane(MINS[0] - trajec[index, 0], MAXS[0] - trajec[index, 0], 0,
                     MINS[2] - trajec[index, 1], MAXS[2] - trajec[index, 1])
        #         ax.scatter(data[index, :22, 0], data[index, :22, 1], data[index, :22, 2], color='black', s=3)

        if index > 1:
            ax.plot3D(trajec[:index, 0] - trajec[index, 0],
                      np.zeros_like(trajec[:index, 0]),
                      trajec[:index, 1] - trajec[index, 1],
                      linewidth=1.0,
                      color='blue')
        #             ax = plot_xzPlane(ax, MINS[0], MAXS[0], 0, MINS[2], MAXS[2])

        for i, (chain, color) in enumerate(zip(smpl_kinetic_chain, colors)):
            #             print(color)
            if i < 5:
                linewidth = 4.0
            else:
                linewidth = 2.0
            ax.plot3D(data[index, chain, 0],
                      data[index, chain, 1],
                      data[index, chain, 2],
                      linewidth=linewidth,
                      color=color)
        #         print(trajec[:index, 0].shape)

        plt.axis('off')

        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

        if out_name is not None:
            plt.savefig(out_name, dpi=96)
            plt.close()

        else:
            io_buf = io.BytesIO()
            fig.savefig(io_buf, format='raw', dpi=96)
            io_buf.seek(0)
            # print(fig.bbox.bounds)
            arr = np.reshape(np.frombuffer(io_buf.getvalue(), dtype=np.uint8),
                             newshape=(int(fig.bbox.bounds[3]),
                                       int(fig.bbox.bounds[2]), -1))
            io_buf.close()
            plt.close()
            return arr

    out = []
    for i in range(frame_number):
        out.append(update(i))
    out = np.stack(out, axis=0)
    return torch.from_numpy(out)


def draw_to_batch(smpl_joints_batch, title_batch=None, outname=None):

    batch_size = len(smpl_joints_batch)
    out = []
    for i in range(batch_size):
        out.append(
            plot_3d_motion([
                smpl_joints_batch[i], None,
                title_batch[i] if title_batch is not None else None
            ]))
        if outname is not None:
            imageio.mimsave(outname[i], np.array(out[-1]), duration=50)
    out = torch.stack(out, axis=0)
    return out
