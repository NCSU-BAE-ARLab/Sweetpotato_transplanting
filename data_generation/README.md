<p align="center">
  <img src="https://github.com/NCSU-BAE-ARLab/Sweetpotato_transplanting/blob/main/assets/data_generation.png" width="80%" />
</p>


### Dataset generation
```bash
python synthesize.py
```
Please make sure to set the path to the input and output folders, before running the script.

### Convert to coco format

```bash
python labelme2coco_v1.py
```

### To visualize 
```bash
python visualize_coco.py
```
