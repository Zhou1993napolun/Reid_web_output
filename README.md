# Clothes_detection_ARM

## This project is suitable for both ARM and x86 environment.



### Some of the files were not uploaded because they were too large:

```shell
./clo_yolo/obj_4000.weights ./deep_sort/deep/checkpoint
```

The download link of these files are:

For the `./clo_yolo/obj_4000.weights` the link is : `https://pan.baidu.com/s/1AT7tL9vNzFKGN4cQpKWusQ` And the password is : `1234`

or you can try this link to download the file : `update soon`



For the `./deep_sort/deep/checkpoint` the link is : `https://pan.baidu.com/s/1ZI_UVUsPC9NKPFmjyF4MiQ` And the password is : `1234`

or you can try this link to download the file : `update soon`





**The python version of these project is 3.7**

After download these weights file and put it to the correct place, we should run this command to setup our environment.

```shell
pip install requirement.txt
```

Then run this command to test that all the dependencies are installed, and if there are no installed libraries, install them directly using the `pip` command.

```shell
python3 reid_clothes.py --video_path 'your video path'
```





Finally, we only should use these command to run the project.

```shell
python3 reid_clothes.py --video_path 'your video path'
```

or

```shell
python3 no_reid_clothes.py.py --video_path 'your video path'
```

For the camera as input option, we will add it in a later optimization.
