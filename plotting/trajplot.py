import numpy as np
import matplotlib.pyplot as plt

def mulDeTrajPlot(x:np.ndarray,
                  ax:plt.Axes = None,
                  trajectory_color = "darkblue",
                  point_color = "green",
                  cur_pos = True,
                  box_lim = 2000.,
                  ext = 0):
    n_step = x.shape[0]
    if ext:
        x = x[:, :-ext]
    x = x.reshape([n_step, -1, 6])
    n_obj = x.shape[1]
    if ax is None:
        ax1 = plt.axes(projection="3d", xlim=(-box_lim, box_lim), ylim=(-box_lim, box_lim), zlim=(-box_lim, box_lim))
        
    else:
        ax1 = ax
    ax1.set_xlabel("radially")
    ax1.set_ylabel("track")
    ax1.set_zlabel("angular momentum")
    ax1.set_aspect('equal')
    
    objs = {
        "start_points": [],
        "trajs": [],
        "curs": [],
    }

    for i in range(n_obj):
        color = trajectory_color if i==0 else "red"
        objs["start_points"].append( ax1.scatter3D(x[0,i,0],x[0,i,1],x[0,i,2], color=point_color) )
        objs["trajs"].append( ax1.plot3D(x[:,i,0], x[:,i,1], x[:,i,2], color=color)[0] )
        if cur_pos:
            objs["curs"].append( ax1.scatter3D(x[-1,i,0],x[-1,i,1],x[-1,i,2], color="red") )
    return ax1, objs


from plotting.utils import sphere
def mulDeSafeSpherePlot(x:np.ndarray,
                        ax:plt.Axes = None,
                        trajectory_color = "darkblue",
                        sphere_color = "red",
                        rad = 500.,
                        quiver = True,
                        box_lim = 2000.,
                        ext = 0):
    n_step = x.shape[0]
    if ext:
        x = x[:, :-ext]
    x = x.reshape([n_step, -1, 6])
    n_obj = x.shape[1]
    if ax is None:
        ax1 = plt.axes(projection="3d", xlim=(-box_lim, box_lim), ylim=(-box_lim, box_lim), zlim=(-box_lim, box_lim))
        
    else:
        ax1 = ax
    ax1.set_xlabel("radially")
    ax1.set_ylabel("track")
    ax1.set_zlabel("angular momentum")
    ax1.set_aspect('equal')
    
    objs = {
        "trajs": [],
        "curs": [],
        "spheres": [],
        "quivers": []
    }

    for i in range(n_obj):
        if i==0:
            objs["trajs"].append( ax1.plot3D(x[:,i,0], x[:,i,1], x[:,i,2], color=trajectory_color)[0] )
            objs["curs"].append( ax1.scatter3D(x[-1,i,0],x[-1,i,1],x[-1,i,2], color=trajectory_color) )
        else:
            objs["spheres"].append( sphere(x[-1,i,:3], rad=rad, color=sphere_color, ax=ax1)[1] )
            if quiver:
                objs["quivers"].append( ax1.quiver(*x[-1,i,:6], color="blue", length=10))
    return ax1, objs