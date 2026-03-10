source export.sh
source processor/sai.sh ${1} 0.15
python processor/unproj_scene.py ${1} --depth_trunc 4.0 --voxel_size 0.01 --sdf_trunc 0.5 --correct_convention
