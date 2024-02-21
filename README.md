# LiDAR Diffusion

Installing python packages pre-requisites:

`sudo apt install build-essential python3-dev libopenblas-dev`

`pip3 install -r requirements.txt`

Installing MinkowskiEngine:

`pip3 install -U MinkowskiEngine==0.5.4 --install-option="--blas=openblas" -v --no-deps`

To setup the code run the following command on the code main directory:

`pip3 install -U -e .`

# Diffusion Scene Completion Pipeline

For training the diffusion model, the configurations are defined in `config/config.yaml`, and the training can be started with:

`python3 train.py`

For training the refinement network, the configurations are defined in `config/config_refine.yaml`, and the training can be started with:

`python3 train_refine.py`

For running the scene completion inference we provide a pipeline where both the diffusion and refinement network are loaded and used to complete the scene from an input scan. You can run the pipeline with the command:

`python3 tools/diff_completion_pipeline.py --ckpt CHECKPOINT_PATH -T DENOISING_STEPS -s CONDITIONING_WEIGHT`

Upon acceptance of the paper, we will release the noise predictor and refinement network together with the code.
