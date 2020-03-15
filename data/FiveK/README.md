# TextFiveK Dataset

## Data Format

### File Tree
The data is saved in `/home/jingshi/project/LDIE/data/FiveK` in server `10.9.226.94`.

The file tree is 

```shell
FiveK
├── FiveK.json
└── images
    ├── 0001O.jpg
    ├── 0001A.jpg
    └── ...
```

### Data Structure

For images, each file in the directory is an image.

For `FiveK.json`, the data is stored as a list of dictionary, and each dictionary stores one image pair with language request annotations. One example of the dictionary is 

```python
{'input': '0001O.jpg',
 'output': '0001A.jpg', 
 'request': 'Image needsTo be soften and sharpened'}
```

- input: the name of input image. For example, for the above input image name, you can find its relative path at `FiveK/images/0001O.jpg`.
- output: the name of output image. The relative path can be find in the same way as input.
- request: the language request annotated by workers.
