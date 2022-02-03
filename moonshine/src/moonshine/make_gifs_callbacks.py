import logging
import tempfile

import matplotlib.pyplot as plt
import wandb

logger = logging.getLogger(__file__)


def log_reconstruction_gifs(scenario, batch, outputs, prefix):
    video_format = 'gif'
    fps = 60
    x_reconstruction = outputs['x_reconstruction']
    input_anim = scenario.example_to_gif(batch)
    input_gif_filename = tempfile.mktemp(suffix=f'.{video_format}')
    input_anim.save(input_gif_filename, writer='imagemagick', fps=fps)
    plt.close(input_anim._fig)
    reconstruction_dict = scenario.flat_vector_to_example_dict(batch, x_reconstruction)
    reconstruction_anim = scenario.example_to_gif(reconstruction_dict)
    reconstruction_gif_filename = tempfile.mktemp(suffix=f'.{video_format}')
    reconstruction_anim.save(reconstruction_gif_filename, writer='imagemagick', fps=fps)
    plt.close(reconstruction_anim._fig)
    wandb.log({
        f'{prefix}_input_gif':          wandb.Video(input_gif_filename, fps=fps, format=video_format),
        f'{prefix}_reconstruction_gif': wandb.Video(reconstruction_gif_filename, fps=fps, format=video_format),
    })
