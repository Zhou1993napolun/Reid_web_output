# Reid With Web Output

## This project is suitable for both ARM and x86 environment.

### Some of the files were not uploaded because they were too large:

```shell
./deep_sort
```

The download link of these files are:

For the `./deep_sort` the link is : `https://pan.bnu.edu.cn/l/6FDc00   `



or you can try this link to download the file : `update soon`



### Start the project

**The python version of these project is 3.7**

After download these weights file and put it to the correct place, we should run this command to setup our environment.

**For example**

1、Create a virtual environment 

```shell
conda create -n reid_flask python=3.7
```

2、Cd to your work file

```
cd your_work_file_path
```

3、Install the require libraries

```shell
pip install requirement.txt
```

4、Update the new upsampling file

```shell
cp upsampling.py your_virtual_environment_path/lib/lib/python3.7/site-packages/torch/nn/modules/upsampling.py
```

5、Run this command to start the Project

```shell
python3 flask_with_output_test.py --camera 'your camera path'
```

6、And then, we can use this device or other devices on the same LAN to log in to the website to check the output of the project. (The default IP port is 8080, which we can change in the file)

Example

```shell
192.168.2.66:8080/video_feed
```




