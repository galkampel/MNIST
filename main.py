import argparse
import json
from dataset import *
from torch.utils.data import DataLoader
import torch.optim as optim
from utils import *
from torchvision.transforms import transforms
from sklearn.model_selection import StratifiedKFold
from models.model import *
from train import Trainer
import optuna
from scipy import stats


# TODO (check)
#    i. check save/load models, plot
#   ii. load model and plot train-validation accuracy on best model
#  iii. save best params (from trial)
#   iv. check flow
#    v. objective function for optuna (cross-validation)
#   vi. fix run best model
#   main: if train- use trial (optuna+objective), if test, use best params (run_best_model)
#   print test accuracy (mean +- sd) (done)
#   load model


def get_arguments(arg_list=None):
    parser = argparse.ArgumentParser(description="Model parameters")
    parser.add_argument("--params_foldername", type=str, default="input", choices=["input"])
    parser.add_argument("--params_filename", type=str, default="trial_standard",
                        choices=["trial_standard", "trial_contrastive", "trial_triplet", "run_best_standard",
                                 "run_best_contrastive", "run_best_triplet"])
    return parser.parse_args(arg_list)


def set_data_loader(dataset, idxes, batch_size, criterion_type, shuffle=False):
    if criterion_type == 'contrastive':
        data_set = MNISTContrastiveDataset(dataset, idxes, sim_dist_dict, seed=SEED)
    elif criterion_type == 'triplet':
        data_set = MNISTTripletDataset(dataset, idxes, sim_dist_dict, seed=SEED)
    else:
        data_set = dataset
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=shuffle)
    return data_loader


def set_model_params(trial):
    params = {}
    if MODEL_ID == 1:
        n_layers = 2
        params['n_layers'] = n_layers
    elif MODEL_ID == 2:
        hidden_features = trial.suggest_categorical('hidden_features', [512, 256])
        params['hidden_features'] = hidden_features
        num_ps = 3 if hidden_features == 512 else 2
        for i in range(1, num_ps + 1):
            p = trial.suggest_float(f"dropout_l{i}", 0.2, 1.0)
            params[f"dropout_l{i}"] = p
    return params


def objective(trial):
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
    lr = trial.suggest_float("lr", 1e-3, 1.0, log=True)  # [1e-5, 1e-1]
    apply_wd = trial.suggest_categorical("apply_wd", [True, False])
    wd = 0.0
    if apply_wd:
        wd = trial.suggest_float("wd", 1e-4, 1e-1, log=True)
    # margin can be a hyperparameter, (TripletLoss)
    criterion = criterion_dict[CRITERION_TYPE]
    criterion_cls = nn.CrossEntropyLoss() if APPLY_CLS else None
    model_params = set_model_params(trial)
    val_accs = []
    for fold, (val_idxes, train_idxes) in enumerate(skf.split(data, targets), 1):  # original: (train_idxes, val_idxes)
        print(f'tuning fold {fold}')
        train_loader = set_data_loader(training_set, train_idxes, batch_size, CRITERION_TYPE, shuffle=True)
        val_loader = set_data_loader(training_set, val_idxes, batch_size, CRITERION_TYPE)
        model = ConvNet(**model_params) if MODEL_ID == 1 else MLPNet(**model_params)
        model = model.to(DEVICE)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        trainer = Trainer(model, optimizer, criterion, DEVICE, CRITERION_TYPE, criterion_cls, NORMALIZE)
        best_val_acc = 0
        for epoch in range(1, EPOCHS+1):
            trainer.train_epoch(train_loader)
            train_acc = trainer.evaluate_epoch(train_loader)
            val_acc = trainer.evaluate_epoch(val_loader)
            if VERBOSE:
                print(f'epoch {epoch}:\ttrain accuracy = {train_acc:.4f}\tvalidation accuracy = {val_acc:.4f}')
            if best_val_acc < val_acc:
                best_val_acc = val_acc

        val_accs.append(best_val_acc)

    return np.mean(val_accs)


def run_best_model(run_params, start_epoch, trainers):  # start_epoch=1, trainers=None (or list of trainers)
    batch_size = run_params.get('batch_size', 32)
    lr = run_params.get('lr', 1e-3)
    wd = run_params.get('wd', 0.0)
    criterion = criterion_dict[CRITERION_TYPE]
    criterion_cls = nn.CrossEntropyLoss() if APPLY_CLS else None
    model_params = run_params["model_params"]
    model_params_str = ''.join(f'{name}={val}_' for name, val in model_params.items())
    folds_dict = {}
    model_name = f'{CRITERION_TYPE}_{"ConvNet" if MODEL_ID == 1 else "MLPNet"}'
    best_model_name = f'{model_name}_model_cls={APPLY_CLS}_{model_params_str}lr={lr}_wd={wd}'
    best_epoch_lst = []
    test_acc_folds_lst = []
    print(f'best_model_name = {best_model_name}')
    for fold, (val_idxes, train_idxes) in enumerate(skf.split(data, targets), 1):
        print(f'fold {fold}:')
        train_loader = set_data_loader(training_set, train_idxes, batch_size, CRITERION_TYPE, shuffle=True)
        val_loader = set_data_loader(training_set, val_idxes, batch_size, CRITERION_TYPE)
        test_loader = set_data_loader(test_set, val_idxes, batch_size, criterion_type='standard')
        if trainers is None:
            model = ConvNet(** model_params) if MODEL_ID == 1 else MLPNet(** model_params)
            model = model.to(DEVICE)
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
            trainer = Trainer(model, optimizer, criterion, DEVICE, CRITERION_TYPE, criterion_cls, NORMALIZE)
        else:
            trainer = trainers[fold]
        acc_dict = {'train': [], 'validation': []}
        best_acc = 0.0
        best_epoch = 0
        test_acc_lst = []
        for epoch in range(start_epoch, EPOCHS+1):
            print(f'training epoch {epoch}')
            trainer.train_epoch(train_loader)
            trn_acc = trainer.evaluate_epoch(train_loader)
            val_acc = trainer.evaluate_epoch(val_loader)
            test_acc = trainer.eval_epoch_standard(test_loader)
            acc_dict['train'].append(trn_acc)
            acc_dict['validation'].append(val_acc)
            test_acc_lst.append(test_acc)
            if val_acc > best_acc:
                best_acc = val_acc
                best_epoch = epoch
                if run_params.get('save_model', False):
                    trainer.save_checkpoint(epoch, fold, acc_dict, best_model_name)
        # fold_dict = {"train-val_accuracy": acc_dict, "test_accuarcy": test_acc_lst}
        test_acc_folds_lst.append(test_acc_lst)

        folds_dict[fold] = acc_dict
        best_epoch_lst.append(best_epoch)
    test_dict = {'accuracy': np.array(test_acc_folds_lst), 'epoch': best_epoch_lst}
    return folds_dict, test_dict, best_model_name


def plot_best_model_results(folds_dict, model_name, folder_plot='plots'):
    trn_mat = []
    val_mat = []
    N = len(folds_dict)
    for i in range(1, N+1):
        trn_mat.append(folds_dict[i]['train'])
        val_mat.append(folds_dict[i]['validation'])
    trn_arr = np.array(trn_mat).mean(0)
    val_arr = np.array(val_mat).mean(0)
    acc_dict = {'train': trn_arr, 'validation': val_arr}
    plot_accuracy(acc_dict, folder_plot, model_name, save_model=True)


def save_best_trial(best_trial, save_folder_name='results'):
    save_folder = os.path.join(os.getcwd(), save_folder_name)
    os.makedirs(save_folder, exist_ok=True)
    filename = f'{CRITERION_TYPE}_model={MODEL_ID}'
    file_path = os.path.join(save_folder, f'{filename}.json')
    save_params = {'model_params': best_trial.params, 'best_value': best_trial.values}
    with open(file_path, 'w') as json_save_file:
        json.dump(save_params, json_save_file)


def load_best_models(params):
    trainers = []
    model_name = params.get("model_name", "")
    min_epoch = 1
    model_params = params.get("model_params", {})
    for fold in range(1, N_FOLDS+1):
        model = ConvNet(** model_params) if MODEL_ID == 1 else MLPNet(** model_params)
        model = model.to(DEVICE)
        criterion = criterion_dict[CRITERION_TYPE]
        criterion_cls = nn.CrossEntropyLoss() if APPLY_CLS else None
        optimizer = optim.Adam(model.parameters(), lr=params.get("lr", 1e-3), weight_decay=params.get("wd", 0.0))
        trainer = Trainer(model, optimizer, criterion, DEVICE, CRITERION_TYPE, criterion_cls, NORMALIZE)
        trainer.load_model(f'{model_name}_fold={fold}')
        trainers.append(trainer)
    return trainers, model_name, min_epoch


def main(params):
    if exe_type == 'study':
        study = optuna.create_study(study_name=f'study_{CRITERION_TYPE}', direction="maximize")
        study.optimize(objective, n_trials=n_trials)
        best_trial = study.best_trial
        print(f'best trial values = {best_trial.values}')  # best value acc?
        print(f'best trial params:\n{best_trial.params}')  # best params
        save_best_trial(best_trial)
    elif exe_type == 'run_best':
        folds_dict, test_dict = {}, {}
        best_model_name = ""
        start_epoch = 1
        trainers = None
        if params.get("load_model", False):
            trainers, best_model_name, start_epoch = load_best_models(params)
        if params.get("train_model", False):
            folds_dict, test_dict, best_model_name = run_best_model(params, start_epoch, trainers)
        if params.get("plot_model", False):
            plot_best_model_results(folds_dict, best_model_name)
        if VERBOSE:
            epoch_mode = stats.mode(test_dict['epoch']).mode.item()
            acc_folds = test_dict['accuracy'][:, epoch_mode]
            print(f'epoch mode = {epoch_mode}')
            print(f'test accuracy = {np.mean(acc_folds)} (+/- {np.std(acc_folds)})')


if __name__ == '__main__':
    root_dir = os.getcwd()
    args = get_arguments()
    full_path = os.path.join(root_dir, args.params_foldername, f'{args.params_filename}.json')
    exe_type = 'study' if 'trial' in args.params_filename else 'run_best'
    with open(full_path) as json_file:
        exe_params = json.load(json_file)
    SEED = exe_params.get("seed", 0)
    MODEL_ID = exe_params.get("model_id", 2)  # 1- ConvNet model, 2- MLPNet
    EPOCHS = exe_params.get("epochs", 20)  # 10, 20, 30
    cuda_device = exe_params.get("cuda_device", 0)
    DEVICE = f"cuda:{cuda_device}" if torch.cuda.is_available() else "cpu"
    CRITERION_TYPE = exe_params.get("criterion_type", 'standard')
    APPLY_CLS = exe_params.get("apply_cls", False)
    # e.g. 5 -> ~12,000 for training and ~48,000 for validation
    N_FOLDS = exe_params.get("n_splits", 10)
    random_state = exe_params.get("random_state", 1)
    VERBOSE = exe_params.get("verbose", True)
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=random_state)
    torch.manual_seed(SEED)
    transform = transforms.Compose([transforms.ToTensor()])
    training_set = load_MNIST_dataset(transform, train=True, root=root_dir)
    test_set = load_MNIST_dataset(transform, train=False, root=root_dir)
    data = training_set.data.numpy()
    targets = training_set.targets.numpy()
    sim_dist_dict = create_similarity_dist_dict(CRITERION_TYPE)
    criterion_dict = {'contrastive': ContrastiveLoss(), 'triplet': TripletLoss(), 'standard': nn.CrossEntropyLoss()}
    NORMALIZE = exe_params.get("normalize", False)
    n_trials = exe_params.get("n_trials", 20)
    best_model_params = exe_params.get("run_params", {})
    main(best_model_params)
