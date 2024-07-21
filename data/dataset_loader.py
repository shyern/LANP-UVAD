import importlib
from data.base_dataset import BaseDataset
from torch.utils.data.dataloader import DataLoader


def find_dataset_using_name(dataset_name):
    # Given the option --dataset [datasetname],
    # the file "datasets/datasetname_dataset.py"
    # will be imported.
    if dataset_name == 'shanghaitech':
        perx = 'sh'
    elif dataset_name == 'ucf-crime':
        perx = 'ucf'
    else:
        perx = ''
    dataset_filename = "data." + "dataset_" + perx

    datasetlib = importlib.import_module(dataset_filename)

    # In the file, the class called DatasetNameDataset() will
    # be instantiated. It has to be a subclass of BaseDataset,
    # and it is case-insensitive.
    dataset = None
    target_dataset_name = 'Dataset_' + perx.upper()
    for name, cls in datasetlib.__dict__.items():
        if name.lower() == target_dataset_name.lower() \
                and issubclass(cls, BaseDataset):
            dataset = cls

    if dataset is None:
        raise ValueError("In %s.py, there should be a subclass of BaseDataset "
                         "with class name that matches %s in lowercase." %
                         (dataset_filename, target_dataset_name))

    return dataset


def CreateDataset(args, logger):
    Dataset = find_dataset_using_name(args.dataset)

    dataset_test = Dataset()
    dataset_test.initialize(args, is_train=False)

    dataset_train = Dataset()
    dataset_train.initialize(args, is_train=True, is_normal=False)
    logger.info(dataset_train.logger_info)

    dataset_cluster = Dataset()
    dataset_cluster.initialize(args, is_train=True, is_normal=False)

    dataset_train_eval = Dataset()
    dataset_train_eval.initialize(args, sample_type="uniform", is_train=False, eval_train=True)

    test_loader = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, drop_last=False)
    train_loader = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    train_loader_cluster = DataLoader(dataset_cluster, batch_size=1, shuffle=False, num_workers=0, pin_memory=True, drop_last=False)
    train_eval_loader = DataLoader(dataset_train_eval, batch_size=1, shuffle=False, pin_memory=True, drop_last=False)

    logger.info("dataset [%s] was created" % (args.dataset))
    

    return test_loader, train_loader, train_eval_loader, train_loader_cluster