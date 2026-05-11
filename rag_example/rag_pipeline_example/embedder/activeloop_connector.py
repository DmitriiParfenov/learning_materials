import deeplake


dataset_path = 'hub://activeloop/mnist-train'
ds = deeplake.load(dataset_path)
print(ds.summary())