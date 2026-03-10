export QT_QPA_PLATFORM=${QT_QPA_PLATFORM:-offscreen}
scene_path=$1
if [ -d "$scene_path/colmap/images" ]; then
    rm -r $scene_path/colmap/images
fi
if [ -d "$scene_path/colmap/dense" ]; then
    rm -r $scene_path/colmap/dense
fi
if [ -f "$scene_path/colmap/database.db" ]; then
    rm $scene_path/colmap/database.db
fi

 mkdir $scene_path/colmap/images
 mkdir $scene_path/colmap/dense
 cp $scene_path/images/frame_*.jpg $scene_path/colmap/images/
 
 colmap feature_extractor \
     --database_path $scene_path/colmap/database.db \
     --image_path $scene_path/colmap/images
 
 colmap exhaustive_matcher \
     --database_path $scene_path/colmap/database.db
 
 colmap point_triangulator \
     --database_path $scene_path/colmap/database.db \
     --image_path $scene_path/colmap/images \
     --input_path $scene_path/colmap/sparse/0 \
     --output_path $scene_path/colmap/sparse/0 \
     --clear_points 1

## added by claude
 colmap bundle_adjuster \
     --input_path $scene_path/colmap/sparse/0 \
     --output_path $scene_path/colmap/sparse/0 \
     --BundleAdjustment.refine_extrinsics 0
 
#  colmap image_undistorter \
#      --image_path $scene_path/colmap/images \
#      --input_path $scene_path/colmap/sparse/0 \
#      --output_path $scene_path/colmap/dense
 
#  colmap patch_match_stereo \
#      --workspace_path $scene_path/colmap/dense
 
#  colmap stereo_fusion \
#      --workspace_path $scene_path/colmap/dense\
#      --output_path $scene_path/colmap/dense/fused.ply