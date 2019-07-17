## Splitter

This scripts help to split images that contain multiple joined photos.
It's a naive approach and not optimized.

To use it place your input images in the **input_to_split** folder and run **main.py**.
The splitted images will be outputed to the **splitted** folder.

The algorithm is the following :

For each pair of **adjacent** line/column the **mean difference** between the **adjacents pixels** is computed.
Afterwards the **total mean** of this value is computed.
Then for each column/line the **difference** between its **mean value** and the **total mean** is computed.
If this **difference** is greater than an **arbitrary threshold**, it will detect the column/line as a **split** between two photos on the same image.

Afterwards all the **splits** are filtered and only the one that seems big enought are kept.

You can try to tune it using thoses parameters :

* MEAN_DISTANCE = Threshold were the mean value of a column or line is considered to be a split.
* MIN_DIM_PCT = The minimum dimension (with or height) in percent of the source image to consider a split to be valid.
* FILL_INSTEAD_OF_STRETCH = Images are resize to 512*512, tell to fill the background with black instead of stretching them.

### Todo

* Add command line arguments
* Optimize 
* Use a dynamic threshold instead of a fixed one