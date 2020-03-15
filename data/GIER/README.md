# GIER Dataset

## Data Format

### File Tree

The data is saved in `/home/jingshi/project/LDIE/data/IER2` in server `10.9.226.94`.

The file tree is 

```shell
IER2
├── IER2.json
├── url_dict.json
├── images
    ├── lHT34_lHT34.jpg
    ├── lHT34_jwtyK.jpg
    └── ...
└── masks
    ├── lHT34_lHT34_mask.json
    ├── 37nv66_37nv66_mask.json
    └── ... 
```

### Data Structure

For images, each file in the directory is an image.

The image URL is stored in `url_dict.json`.

For masks, each file, e.g., `lHT34_lHT34_mask.json`, is a list of non-overlapping masks for the input image, encoded using RLE.

For `IER2.json`, the data is stored as a list of dictionary, and each dictionary stores one image pair with all annotations. One example of the dictionary is 

```python
{'input': 'lHT34_lHT34.jpg',
 'output': 'lHT34_jwtyK.jpg', 
 'segment': 'lHT34/lHT34.jpg', 
 'palette': 'lHT34/lHT34plt.jpg',
 'request': 'Two focus points...',
 'detailed_request': 'One focus point on the guy on the far',
 'dataset': 'zhopped',
 'workerId': 'ANOTWUSBR3VB1', 
 'operator': {'brightness': {'local': False, 'ids': []},  'saturation': {'local': False, 'ids': []}, 'tint':     {'local': True, 'ids': [1,3,4]}}, 
 'expert_summary': ['turn the image black and white', 'the on the left is very dark, the one one right has more light', 'grayscale and vinaginette alot'], 
'amateur_summary': ['turn the image black and white', 'the on the left is very dark']}
```

- input: the name of input image. For example, for the above input image name, you can find its relative path at `IER2/images/lHT34_lHT34.jpg` by replacing `/` with `_` in the name.

- output: the name of output image. The relative path can be find in the same way as input.

- segment: deprecated.
- palette: deprecated.

- request: the original user's request crawled from web.
- detailed request: the original user's detailed request crawled from web.
- dataset: either from "zhopped" or "reddit".
- workerId: deprecated.
- operator: it is a dictionary composed of
  - Operation name (brightenss) : it is the operation name that is applied to this image pair. It has more annotations as 
    - local: True or False, indicating local or global operation.
    - ids: a list of index, where you can find the real mask at `IER2/masks/lHT34_lHT34_mask.json`. Each id is exactly the index for the mask list. If local is false, then the ids is an empty list.
  
- expert_summary: a list of requests annotated by expert.
- amateur_summary: a list of requests annotated by amateur.