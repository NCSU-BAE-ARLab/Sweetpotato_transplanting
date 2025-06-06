### How to run the code?

We are using pytorch and torchvision's built-in implementations of deep learning models. 
The version of some of the main packages in our code environemnt is as below.

```bash
python 3.8.10
pytorch 2.0.0+nv23.05
torchvision 0.15.2a0
```
### Train

```bash
python train_fast.py
```

### Evaluation
To evaluate the model with a particular trained weights
```bash
python validate.py
```
To evaluate the model with trained weights in all epoches
```bash
python validate_all.py
```
