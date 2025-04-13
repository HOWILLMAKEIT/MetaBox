import os
import json
import xgboost as xgb


def get_data(mode,surrogates_dir,root_dir):
    train_set,vali_set,test_set=None,None,None
    if mode == "v3-test":
        train_set,vali_set,test_set,bo_initializations=load_data(root_dir, only_test=True)
    elif mode == "v3-train-augmented":
        train_set,vali_set,test_set,bo_initializations=load_data(root_dir, only_test=False, augmented_train=True)
    elif mode in ["v1", "v2", "v3"]:
        train_set,vali_set,test_set,bo_initializations=load_data(root_dir, version = mode, only_test=False)
    else:
        raise ValueError("Provide a valid mode")

    surrogates_file = surrogates_dir+"summary-stats.json"
    if os.path.isfile(surrogates_file):
        with open(surrogates_file) as f:
            surrogates_stats = json.load(f)

    return train_set,vali_set,test_set,bo_initializations,surrogates_stats

def load_data( rootdir="", version = "v3", only_test = True, augmented_train = False):
    
        """
        Loads data with some specifications.
        Inputs:
            * root_dir: path to directory with the benchmark data.
            * version: name indicating what HPOB version to use. Options: v1, v2, v3).
            * Only test: Whether to load only testing data (valid only for version v3).  Options: True/False
            * augmented_train: Whether to load the augmented train data (valid only for version v3). Options: True/False

        """

        print("Reading data...")
        meta_train_augmented_path = os.path.join(rootdir, "meta-train-dataset-augmented.json")
        meta_train_path = os.path.join(rootdir, "meta-train-dataset.json")
        meta_test_path = os.path.join(rootdir,"meta-test-dataset.json")
        meta_validation_path = os.path.join(rootdir, "meta-validation-dataset.json")
        bo_initializations_path = os.path.join(rootdir, "bo-initializations.json")

        with open(meta_test_path, "rb") as f:
            meta_test_data = json.load(f)

        with open(bo_initializations_path, "rb") as f:
            bo_initializations = json.load(f)

        meta_train_data = None
        meta_validation_data = None
        
        if not only_test:
            if augmented_train or version=="v1":
                with open(meta_train_augmented_path, "rb") as f:
                    meta_train_data = json.load(f)
            else:
                with open(meta_train_path, "rb") as f:
                    meta_train_data = json.load(f)
            with open(meta_validation_path, "rb") as f:
                meta_validation_data = json.load(f)

        if version != "v3":
            temp_data = {}
            for search_space in meta_train_data.keys():
                temp_data[search_space] = {}

                for dataset in meta_train_data[search_space].keys():
                    temp_data[search_space][dataset] =  meta_train_data[search_space][dataset]

                if search_space in meta_test_data.keys():
                    for dataset in meta_test_data[search_space].keys():
                        temp_data[search_space][dataset] = meta_test_data[search_space][dataset]

                    for dataset in meta_validation_data[search_space].keys():
                        temp_data[search_space][dataset] = meta_validation_data[search_space][dataset]

            meta_train_data = None
            meta_validation_data = None
            meta_test_data = temp_data

        search_space_dims = {}

        for search_space in meta_test_data.keys():
            dataset = list(meta_test_data[search_space].keys())[0]
            X = meta_test_data[search_space][dataset]["X"][0]
            search_space_dims[search_space] = len(X)

        return meta_train_data,meta_validation_data,meta_test_data,bo_initializations

def get_bst(surrogates_dir,search_space_id,dataset_id,surrogates_stats):
    surrogate_name='surrogate-'+search_space_id+'-'+dataset_id
    bst_surrogate = xgb.Booster()
    bst_surrogate.load_model(surrogates_dir+surrogate_name+'.json')

    y_min = surrogates_stats[surrogate_name]["y_min"]
    y_max = surrogates_stats[surrogate_name]["y_max"]
    assert y_min is not None, 'y_min is None!!'

    return bst_surrogate,y_min,y_max
