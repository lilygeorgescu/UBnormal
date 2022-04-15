## Useful scripts for using UBnormal 

### Data set split
To <b>train</b> your model use the following files:
- normal_training_video_names.txt
- abnormal_training_video_names.txt

To <b>validate</b> your model use the following files: 
- normal_validation_video_names.txt
- abnormal_validation_video_names.txt

To <b>test</b> your model use the following files: 
- normal_test_video_names.txt
- abnormal_test_video_names.txt



### Object ids
We release the list of object ids that are not persons in ```object_names_per_video.pkl``` with the following format:
```
format:
{
    video_name: {object_id_1: object_name_1, object_id_2: object_name_3}
}

real example:
{

    'normal_scene_29_scenario_6': {2: 'bicycle', 4: 'bicycle', 6: 'motorcycle'}

}
```
If an object id is not found in ```object_names_per_video.pkl```, it means that the id is assigned to a person.

### Scripts
- ```compute_auc_score.py``` 
    - we used this script to compute the frame-level AUC score.
    
- ```create_tracks_for_tbdc_rbdc.py```
    - we used this script to create the ground-truth tracks in order to compute TBDC and RBDC.
    - this script must be run for each split (train/val/test).
- ```write_frame_level_ground_truth.py```
    - we used this script to compute the frame-level ground-truth.
    - this script must be run for each split (train/val/test).
   
- ```statistics.py```
    - we used this script to compute the statistics of our data set.
    
    
### Ground-truth format

Annotations folder:

- each video has a corresponding folder named ```{video_name}_annotations```.

- for each frame in each video there is a map named ```{video_name}_{frame_index}_gt.png```

- the objects are marked in the map with the corresponding id (0 denotes the background).

    - e.g. if you want to extract the contour of the person with the id ```3``` from the video named ```video_name```,
     you should go to the ```{video_name}_annotations``` folder, read all the maps and searched for the number ```3```.
     
- each abnormal video has a txt file named ```{video_name}_tracks.txt``` that specifies the anomalies with the following format:

    - [object_id,start_frame_idx,end_frame_idx]
- in order to create the frame-level ground-truth the ```{video_name}_tracks.txt``` file is used.
- in order to create the ground-truth tracks to compute TBDC and RBDC both ```{video_name}_tracks.txt``` and the maps from ```{video_name}_annotations``` are used.

 
    
            
            
