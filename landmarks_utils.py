import os
import json
import cv2
import numpy as np
from tqdm import tqdm

import torch
import face_alignment

import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import Video, HTML


def save_landmarks(landmarks, save_path):
    
    if not save_path:
        return
        
    if isinstance(landmarks, list):
        landmarks_dict = {}
        for i, frame_landmarks in enumerate(landmarks):
            landmarks_dict[i] = frame_landmarks.tolist()
    else:
        landmarks_dict = {0: landmarks.tolist()}
        
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(landmarks_dict, f, ensure_ascii=False, indent=4)

        
def load_landmarks(load_path):
    
    with open(load_path) as f:
        landmarks_dict = json.load(f)
    
    num_frames = len(landmarks_dict)
    if num_frames==1:
        landmarks = np.array(landmarks_dict['0'])
    else:
        landmarks = []
        for i in range(num_frames):
            landmarks.append(np.array(landmarks_dict[str(i)]))
    
    return landmarks


def load_video(video_path, output_size=(270, 480)):
    
    cap = cv2.VideoCapture(video_path)
    frames_list = []
    while True:
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.cvtColor(cv2.resize(frame, output_size), cv2.COLOR_BGR2RGB)
        frames_list.append(frame)
    print(f'Video {video_path} loaded: {len(frames_list)} farmes with shape {np.shape(frames_list[0])}')
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f'{fps} frames per second')
    
    return frames_list, fps, Video(video_path, width=output_size[0])


def draw_landmarks(image, landmarks, plot=True, save_path=None):
    
    plot_style = {'marker': 'o', 
                  'markersize': 4, 
                  'linestyle': '-', 
                  'lw': 2}
    
    landmark_types = {'face': list(range(0, 17)),
                  'eyebrow1': list(range(17, 22)),
                  'eyebrow2': list(range(22, 27)),
                  'nose': list(range(27, 31)),
                  'nostril': list(range(31, 36)),
                  'eye1': list(range(36, 42)) + [36],
                  'eye2': list(range(42, 48)) + [42],
                  'lips': list(range(48, 60)) + [48],
                  'teeth': list(range(60, 68)) + [60]
                 }
    
    type_colors = {'face': (0.682, 0.780, 0.909, 0.5),
                  'eyebrow1': (1.0, 0.498, 0.055, 0.4),
                  'eyebrow2': (1.0, 0.498, 0.055, 0.4),
                  'nose': (0.345, 0.239, 0.443, 0.4),
                  'nostril': (0.345, 0.239, 0.443, 0.4),
                  'eye1': (0.596, 0.875, 0.541, 0.3),
                  'eye2': (0.596, 0.875, 0.541, 0.3),
                  'lips': (0.596, 0.875, 0.541, 0.3),
                  'teeth': (0.596, 0.875, 0.541, 0.4)
                  }

    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(image)
    ax.axis('off')

    if landmarks is not None:
        
        for landmark_type in landmark_types:
            type_list = landmark_types[landmark_type]
            type_array = np.stack([landmarks[idx] for idx in type_list])
            ax.plot(type_array[:, 0], type_array[:, 1], color=type_colors[landmark_type], **plot_style)
            
        ax = fig.add_subplot(1, 2, 2, projection='3d')
        surf = ax.scatter(landmarks[:, 0], landmarks[:, 1], landmarks[:, 2], color='tab:orange', edgecolor='tab:blue')

        for landmark_type in landmark_types:
            type_list = landmark_types[landmark_type]
            type_array = np.stack([landmarks[idx] for idx in type_list])
            ax.plot3D(type_array[:, 0], type_array[:, 1], type_array[:, 2], color='tab:blue')

        ax.view_init(elev=100., azim=90.)
        ax.tick_params(labelbottom=False, labeltop=False, labelleft=False, labelright=False)
        ax.set_xlim(ax.get_xlim()[::-1])
    
    if save_path:
        fig.savefig(save_path, bbox_inches='tight')
        
    if plot:
        plt.show()
    else:
        plt.close(fig)
        
    
def animated_frames(frames, figsize=(10,8), fps=10):
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off')
    im = ax.imshow(frames[0])

    def animate(i):
        im.set_array(frames[i])
        return [im,]

    ani = animation.FuncAnimation(fig, animate, frames=len(frames),
                                  interval=1000/fps, blit=True)
    plt.close()

    return ani


def animated_landmarks(video_frames, video_landmarks, figsize=(10, 8), fps=30):
    
    plot_style = {'marker': 'o', 
                  'markersize': 4, 
                  'linestyle': '-', 
                  'lw': 2}
    
    landmark_types = {'face': list(range(0, 17)),
                  'eyebrow1': list(range(17, 22)),
                  'eyebrow2': list(range(22, 27)),
                  'nose': list(range(27, 31)),
                  'nostril': list(range(31, 36)),
                  'eye1': list(range(36, 42)) + [36],
                  'eye2': list(range(42, 48)) + [42],
                  'lips': list(range(48, 60)) + [48],
                  'teeth': list(range(60, 68)) + [60]
                 }
    
    type_colors = {'face': (0.682, 0.780, 0.909, 0.5),
                  'eyebrow1': (1.0, 0.498, 0.055, 0.4),
                  'eyebrow2': (1.0, 0.498, 0.055, 0.4),
                  'nose': (0.345, 0.239, 0.443, 0.4),
                  'nostril': (0.345, 0.239, 0.443, 0.4),
                  'eye1': (0.596, 0.875, 0.541, 0.3),
                  'eye2': (0.596, 0.875, 0.541, 0.3),
                  'lips': (0.596, 0.875, 0.541, 0.3),
                  'teeth': (0.596, 0.875, 0.541, 0.4)
                  }

    fig = plt.figure(figsize=(10, 8))
    ax1 = fig.add_subplot(1, 2, 1)
    im = ax1.imshow(video_frames[0])
    ax1.axis('off')

    plots_2d = {}
    for landmark_type in landmark_types:
        type_list = landmark_types[landmark_type]
        type_array = np.stack([video_landmarks[0][idx] for idx in type_list])
        plots_2d[landmark_type] = ax1.plot(type_array[:, 0], type_array[:, 1], 
                                           color=type_colors[landmark_type], **plot_style)[0]

    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    surf = ax2.scatter(video_landmarks[0][:, 0], video_landmarks[0][:, 1], video_landmarks[0][:, 2], 
                       color='tab:orange', edgecolor='tab:blue')

    plots_3d = {}
    for landmark_type in landmark_types:
        type_list = landmark_types[landmark_type]
        type_array = np.stack([video_landmarks[0][idx] for idx in type_list])
        plots_3d[landmark_type] = ax2.plot3D(type_array[:, 0], type_array[:, 1], type_array[:, 2], color='tab:blue')[0]
    
    ax2.view_init(elev=100., azim=90.)
    ax2.tick_params(labelbottom=False, labeltop=False, labelleft=False, labelright=False)
    ax2.set_xlim(ax2.get_xlim()[::-1])
    
    handles = [im, surf, *plots_2d.values(), *plots_2d.values()]
    
    def animate(i):
        im.set_array(video_frames[i])
        if video_landmarks[i].any():
            surf._offsets3d = (video_landmarks[i][:, 0], video_landmarks[i][:, 1], video_landmarks[i][:, 2])  
            for landmark_type in landmark_types:
                type_list = landmark_types[landmark_type]
                type_array = np.stack([video_landmarks[i][idx] for idx in type_list])
                plots_2d[landmark_type].set_data(type_array[:, 0], type_array[:, 1])
                plots_3d[landmark_type].set_data_3d(type_array[:, 0], type_array[:, 1], type_array[:, 2])
        else:
            surf._offsets3d = ([], [], [])
            for landmark_type in landmark_types:
                plots_2d[landmark_type].set_data([], [])
                plots_3d[landmark_type].set_data_3d([], [], [])
                
        return handles

    ani = animation.FuncAnimation(fig, animate, frames=len(video_frames),
                                  interval=1000/fps, blit=True)
    plt.close()
    
    return ani
        

def save_video(video_frames, video_preds, fps, output_path):
    
    ani = animated_landmarks(video_frames, video_preds, fps=fps)
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=fps, metadata=dict(artist='StanfordFacialNerveCenter'))
    print(f'output video: {output_path}')
    ani.save(output_path, writer=writer)
    
    return Video(output_path)


def fa_image_pred(image, save_json_path=None, save_plot_path=None):
    
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, device='cuda', flip_input=False)
    preds = fa.get_landmarks(image)[-1]
    save_landmarks(preds, save_path=save_json_path)
    draw_landmarks(image, preds, plot=True, save_path=save_plot_path)
    
    return preds


def fa_video_pred(in_frames, save_json_path=None, save_frames_dir=None, batch_size=32):
    
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._3D, device='cuda', flip_input=False)
    num_frames = len(in_frames)
    num_batchs = int(np.ceil(num_frames / batch_size))
    preds = []
    with tqdm(total=num_frames) as pbar:
        for i in range(num_batchs):
            batch_frames = in_frames[i*batch_size:(i+1)*batch_size]
            batch = np.stack(batch_frames)
            batch = batch.transpose(0, 3, 1, 2)
            batch = torch.Tensor(batch).cuda()
            batch_preds = fa.get_landmarks_from_batch(batch)
            preds.extend(batch_preds)
            pbar.update(batch_size)
        
    save_landmarks(preds, save_path=save_json_path)
    
    if save_frames_dir:
        os.makedirs(save_frames_dir, exist_ok=True)
        for j, pred in enumerate(preds):
            if not pred.any(): pred = None
            save_path = os.path.join(save_frames_dir, 'frame{}'.format(j))
            draw_landmarks(in_frames[j], pred, False, save_path)
    
    return preds
