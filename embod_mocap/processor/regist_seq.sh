export QT_QPA_PLATFORM=${QT_QPA_PLATFORM:-offscreen}
scene_path=$1
view_path=$2
focal=$3
cx=$4
cy=$5
vocab_tree_path=$6
 
 if [ -z "$scene_path/colmap" ]; then
   echo "Error: No dataset path provided."
   echo "Usage: $0 <dataset_path>"
   exit 1
 fi
 
 colmap model_converter \
     --input_path $scene_path/colmap/sparse/0 \
     --output_path $scene_path/colmap/sparse/0 \
     --output_type BIN
 
 rm -r $view_path/colmap
 mkdir -p $view_path/colmap
 
 cp  $scene_path/colmap/database.db $view_path/colmap/database.db
 
 colmap feature_extractor \
     --database_path $view_path/colmap/database.db \
     --image_path $view_path/images \
     --image_list_path $view_path/image-list.txt \
     --ImageReader.single_camera 1 \
     --ImageReader.camera_model SIMPLE_PINHOLE \
     --ImageReader.camera_params "${focal},${cx},${cy}"
 

#  colmap exhaustive_matcher --database_path $view_path/colmap/database.db
 colmap sequential_matcher \
     --database_path $view_path/colmap/database.db \
     --SequentialMatching.overlap 10 \
     --SequentialMatching.loop_detection 0

 colmap vocab_tree_matcher \
     --database_path $view_path/colmap/database.db \
     --VocabTreeMatching.vocab_tree_path $vocab_tree_path \
     --VocabTreeMatching.match_list_path $view_path/image-list.txt

#  colmap vocab_tree_matcher \
#      --database_path $view_path/colmap/database.db \
#      --VocabTreeMatching.vocab_tree_path home/ubuntu/programs/colmap/vocab_tree_faiss_flickr100K_words32K.bin \
#      --VocabTreeMatching.match_list_path $view_path/image-list.txt

colmap image_registrator \
    --database_path $view_path/colmap/database.db \
     --input_path $scene_path/colmap/sparse/0 \
     --output_path $view_path/colmap 

#  colmap bundle_adjuster \
#      --input_path $view_path/colmap/ \
#      --output_path $view_path/colmap/\
#      --BundleAdjustment.max_num_iterations 100 \
#      --BundleAdjustment.function_tolerance 1e-9
     
 colmap model_converter \
     --input_path $view_path/colmap \
     --output_path $view_path/colmap \
     --output_type TXT
