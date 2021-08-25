
train_net.py --> make_dataset.py
--> train_loader = make_data_loader(cfg, is_train=True)
  --> cfg --> "dataset_name"
  --> define a transform function (to adjust the shape of the image) --> "transform"

--> make_dataset( "dataset_name" & "transform")
  --> according the data_name --> "img_dir" & "label_dir"
  --> _data_factory --> locate the process methods
  --> "dataset" (can return the "inp" and the "out")

--> make_data_Sampler("dataset", shuffle=True)
  --> torch.utils.data.sampler.RandomSampler(data) -- "T or F"
  --> "sampler"

--> make_batch_sampler("sampler", )
  --> torch.utils.data.sampler.BatchSampler(sampler, batch_size, drop_last)

