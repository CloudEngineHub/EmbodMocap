seq_path=$1
view_name=$2
down_scale=$3
vertical=$4
if [ -z "$vertical" ]; then
    vertical=0
fi
vf="scale=iw/${down_scale}:ih/${down_scale}"
if [ "$vertical" = "1" ]; then
    vf="${vf},transpose=1"
fi
mkdir ${seq_path}/${view_name}/images

ffmpeg -i ${seq_path}/${view_name}/data.mov -r 30 -vf "${vf}" -q:v 5 -start_number 0 ${seq_path}/${view_name}/images/${view_name}_%04d.jpg
