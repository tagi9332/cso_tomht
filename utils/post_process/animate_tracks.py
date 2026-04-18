import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML


def animate_tracks(meas_df: pd.DataFrame, tracked_df: pd.DataFrame, output_path: str, display_plots: bool = False):
    """Creates a GIF showing raw detections per frame and active tracks."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Set dynamic limits based on the data
    all_x = pd.concat([meas_df['x'], tracked_df['x']])
    all_y = pd.concat([meas_df['y'], tracked_df['y']])
    
    ax.set_xlim(all_x.min() - 10, all_x.max() + 10)
    ax.set_ylim(all_y.min() - 10, all_y.max() + 10)
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    
    time_steps = sorted(meas_df['time'].unique())
    colors = plt.cm.tab20.colors
    
    # Persistent objects
    raw_scatter = ax.scatter([], [], c='gray', s=20, alpha=0.5, label='Current Detections')
    lines = {}
    
    def update(frame_time):
        ax.set_title(f"TOMHT Tracking - Frame {frame_time}")
        
        # Update raw detections for this specific frame
        current_meas = meas_df[meas_df['time'] == frame_time]
        if not current_meas.empty:
            raw_scatter.set_offsets(current_meas[['x', 'y']].to_numpy())
        else:
            raw_scatter.set_offsets(np.empty((0, 2)))
            
        # Update track histories up to this frame
        current_tracks = tracked_df[tracked_df['time'] <= frame_time]
        active_ids_this_frame = tracked_df[tracked_df['time'] == frame_time]['id'].unique()
        
        for track_id in current_tracks['id'].unique():
            trk_history = current_tracks[current_tracks['id'] == track_id]
            
            # Create a line if it doesn't exist
            if track_id not in lines:
                color = colors[int(track_id) % len(colors)]
                line, = ax.plot([], [], '-', color=color, linewidth=2)
                lines[track_id] = line
                
            # Update the line data
            lines[track_id].set_data(trk_history['x'], trk_history['y'])
            
            # Fade out tracks that are no longer active in the current frame
            if track_id not in active_ids_this_frame:
                lines[track_id].set_alpha(0.3)
            else:
                lines[track_id].set_alpha(1.0)
                
        return [raw_scatter] + list(lines.values())

    ani = animation.FuncAnimation(
        fig, update, frames=time_steps, interval=200, blit=False
    )
    
    ani.save(output_path, writer='pillow')
    
    # Optionally display the animation in the Jupyter notebook
    if display_plots:
        return HTML(ani.to_jshtml())


    plt.close()
    print(f"Saved animation to {output_path}")
