import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from plotting import trajplot
import math

def trajAni(x:np.ndarray,
        box_lim=2000,
        step=10,
        interval=5
        ):
    fig = plt.figure()
    ax = fig.add_axes(rect=[0, 0, 1, 1],
                      projection='3d',
                      xlim=(-box_lim, box_lim),
                      ylim=(-box_lim, box_lim),
                      zlim=(-box_lim, box_lim)
                     )
    ax.set_xlabel("radially")
    ax.set_ylabel("track")
    ax.set_zlabel("angular momentum")
    ax.set_aspect("equal")
    total = x.shape[0]
    x = x.reshape([total, -1, 6])

    n_frame = math.ceil(total/step)

    ax, objs = trajplot.mulDeTrajPlot(x, ax=ax, cur_pos=True)
    trajs = objs["trajs"]
    curs = objs["curs"]
    n_obj = len(trajs)

    colors = ["blue"] + ["red" for _ in range(n_obj-1)]

    def animate(frame_number):
        n = frame_number*step
        for i in range(n_obj):
            trajs[i].remove()
            trajs[i] = ax.plot3D(x[:n,i,0],x[:n,i,1],x[:n,i,2], color=colors[i])[0]
            curs[i].remove()
            curs[i] = ax.scatter3D(x[n,i,0],x[n,i,1],x[n,i,2], color=colors[i])
            

    ani = animation.FuncAnimation(fig, animate, frames=n_frame, interval=interval)

    return ani

from plotting.utils import sphere
def safeSphereAni(x:np.ndarray,
        box_lim=2000.,
        rad=500.,
        step=10,
        interval=5
        ):
    fig = plt.figure()
    ax = fig.add_axes(rect=[0, 0, 1, 1],
                      projection='3d',
                      xlim=(-box_lim, box_lim),
                      ylim=(-box_lim, box_lim),
                      zlim=(-box_lim, box_lim)
                     )
    ax.set_xlabel("radially")
    ax.set_ylabel("track")
    ax.set_zlabel("angular momentum")
    ax.set_aspect("equal")
    total = x.shape[0]
    x = x.reshape([total, -1, 6])

    n_frame = math.ceil(total/step)

    ax, objs = trajplot.mulDeSafeSpherePlot(x[:1], ax=ax, rad=rad)
    trajs = objs["trajs"]
    curs = objs["curs"]
    spheres = objs["spheres"]
    n_debris = len(spheres)


    def animate(frame_number):
        n = frame_number*step
        trajs[0].remove()
        trajs[0] = ax.plot3D(x[:n,0,0],x[:n,0,1],x[:n,0,2], color="blue")[0]
        curs[0].remove()
        curs[0] = ax.scatter3D(x[n, 0, 0], x[n, 0, 1], x[n, 0, 2], color="blue")
        for i in range(n_debris):
            pass
            spheres[i].x_data
            # spheres[i].remove()
            # spheres[i] = sphere(x[n,i+1,:3], ax=ax, rad=rad, color="red")[1]
            

    ani = animation.FuncAnimation(fig, animate, frames=n_frame, interval=interval)

    return ani