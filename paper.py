plots = not evolve and not opt.noplots  # create plots
cuda = device.type != "cpu"
init_seeds(opt.seed + 1 + RANK, deterministic=True)
with torch_distributed_zero_first(LOCAL_RANK):
data_dict = data_dict or check_dataset(data)  # check if None
train_path, val_path = data_dict["train"], data_dict["val"]
nc = 1 if single_cls else int(data_dict["nc"])  # number of classes
names = {0: "item"} if single_cls and len(data_dict["names"]) != 1 else data_dict["names"]  # class names
is_coco = isinstance(val_path, str) and val_path.endswith("coco/val2017.txt")  # COCO dataset