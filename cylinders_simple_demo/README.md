# The Dataset

The un-augmented dataset can be downloaded here. It contains several hundred trajectories of state/action information.

https://drive.google.com/file/d/1FgE1qNDeKJzN_fraKHjUmkFSzetHnRLS/view?usp=sharing

So that it has the folder structure:

```
cylinders_simple_demo
 |- data
    |- gz and pkl files...
    |- test/val/train split text files
```

The data set consists of pkl files, which contain some meta-data as well as filenames of the rest of the data (the .pkl.gz files). The pkl and pkl.gz files each contain a single dictionary, which in turn contains the data. Most of the data is float32 numpy arrays. Each example is a single trajectory of length 50.

# Running the example

1. Generate the augmented dataset


    ./scripts/generate_augmented_dataset -h  # help info
    ./scripts/generate_augmented_dataset ./data hparams/cylinders.hjson ./data_aug


2. train the dynamics model


    ./scripts/train-test_dynamics train -h  # see training options
    ./scripts/train-test_dynamics train ./data_aug hparams/propnet.hjson -n with_augmentation

3. evaluate the dynamics model 
 
  
    ./scripts/train-test_dynamics eval -h  # see eval options
    ./scripts/train-test_dynamics train ./data with_augmentation_xxx # full unique model name from the above training run

# Making a comparison to no augmentation

In this case, you can just train the dynamics model on the original dataset, instead of the augmented one. You can then compare the two using the same validation set (use the un-augmented one for this).


# Example Results

## Augmentations
Here are some visualizations of the first few augmented examples. On my computer, I can generate XXX augmentations per second.

## Training Curves

## Evaluation Metrics

Testing

| Method | mean position error |
|---|---|
| Without Augmentation | xx.xxx |
| With Augmentation | xx.xxx |
