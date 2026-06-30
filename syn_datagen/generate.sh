PYTHONPATH=$HOME/miniconda3/envs/layeredflow_all/lib/python3.10/site-packages \
../blender/build/install/blender \
-noaudio --background \
--python render.py -- \
--data --sample_view --scene_id 0 --num_images 1 --generate_gt
