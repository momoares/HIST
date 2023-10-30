import torch
import torch.nn as nn
import torch.optim as optim
import os
import copy
import json
import argparse
import datetime
import collections
import numpy as np
import pandas as pd
from tqdm import tqdm
import qlib
# regiodatetimeG_CN, REG_US]
from qlib.config import REG_US, REG_CN
# provider_uri = "~/.qlib/qlib_data/us_data"  # target_dir
provider_uri = "~/.qlib/qlib_data/cn_data"  # target_dir
qlib.init(provider_uri=provider_uri, region=REG_CN)
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from torch.utils.tensorboard import SummaryWriter
from qlib.contrib.model.pytorch_gru import GRUModel
from qlib.contrib.model.pytorch_lstm import LSTMModel
from qlib.contrib.model.pytorch_gats import GATModel
from qlib.contrib.model.pytorch_sfm import SFM_Model
from qlib.contrib.model.pytorch_alstm import ALSTMModel
from qlib.contrib.model.pytorch_transformer import Transformer
from model import MLP, HIST
from utils import metric_fn, mse
from dataloader import DataLoader

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

EPS = 1e-12


def get_model(model_name):

    if model_name.upper() == 'MLP':
        return MLP

    if model_name.upper() == 'LSTM':
        return LSTMModel

    if model_name.upper() == 'GRU':
        return GRUModel
    
    if model_name.upper() == 'GATS':
        return GATModel

    if model_name.upper() == 'SFM':
        return SFM_Model

    if model_name.upper() == 'ALSTM':
        return ALSTMModel
    
    if model_name.upper() == 'TRANSFORMER':
        return Transformer

    if model_name.upper() == 'HIST':
        return HIST

    raise ValueError('unknown model name `%s`'%model_name)

'''
average_params函数接受一组字典作为输入，假定它们具有相同的键，
然后计算每个键的值的平均值，并返回结果的平均字典。如果输入列表只包含一个参数集，
它将直接返回该参数集。如果参数具有不同的键，它将引发ValueError异常。
'''
def average_params(params_list):
    assert isinstance(params_list, (tuple, list, collections.deque))
    n = len(params_list)
    if n == 1:
        return params_list[0]
    new_params = collections.OrderedDict()
    keys = None
    for i, params in enumerate(params_list):
        if keys is None:
            keys = params.keys()
        for k, v in params.items():
            if k not in keys:
                raise ValueError('the %d-th model has different params'%i)
            if k not in new_params:
                new_params[k] = v / n
            else:
                new_params[k] += v / n
    return new_params


#预测和实际值的比较
def loss_fn(pred, label, args):
    mask = ~torch.isnan(label)
    #这个 mask 被用于筛选掉 label 中不是 NaN（非数字）的元素。
    return mse(pred[mask], label[mask])


global_log_file = None

#输出时间 在每个输出之前打印时间
def pprint(*args):
    # print with UTC+8 time
    time = '['+str(datetime.datetime.utcnow()+
                   datetime.timedelta(hours=8))[:19]+'] -'
    print(time, *args, flush=True)

    if global_log_file is None:
        return
    with open(global_log_file, 'a') as f:
        print(time, *args, flush=True, file=f)


global_step = -1

def train_epoch(epoch, model, optimizer, train_loader, writer, args, stock2concept_matrix = None):

    global global_step

    model.train()

    for i, slc in tqdm(train_loader.iter_batch(), total=train_loader.batch_length):
        global_step += 1
        feature, label, market_value , stock_index, _ = train_loader.get(slc)

        #根据模型不同输入不同
        if args.model_name == 'HIST':
            pred = model(feature, stock2concept_matrix[stock_index], market_value)
        else:
            pred = model(feature)
        loss = loss_fn(pred, label, args)#lossfn 计算loss的函数

        optimizer.zero_grad()#清空模型参数的梯度
        loss.backward()#backwards 计算梯度
        torch.nn.utils.clip_grad_value_(model.parameters(), 3.)#裁剪梯度 防止爆炸
        optimizer.step()#更新模型参数


def test_epoch(epoch, model, test_loader, writer, args, stock2concept_matrix=None, prefix='Test'):

    model.eval()

    losses = []
    preds = []

    for i, slc in tqdm(test_loader.iter_daily(), desc=prefix, total=test_loader.daily_length):

        feature, label, market_value, stock_index, index = test_loader.get(slc)
        #从测试数据加载器 test_loader 中获取一个数据切片，包括
        #feature（特征）、label（标签）、market_value（市值）、stock_index（股票索引）、以及 index。
        
        with torch.no_grad():#以确保在评估过程中不会进行梯度计算。
            if args.model_name == 'HIST':
                pred = model(feature, stock2concept_matrix[stock_index], market_value)
            else:
                pred = model(feature)

            #计算损失值并把损失值存到列表中
            loss = loss_fn(pred, label, args)
            preds.append(pd.DataFrame({ 'score': pred.cpu().numpy(), 'label': label.cpu().numpy(), }, index=index))

        losses.append(loss.item())

    
    #evaluate 计算各种评分
    preds = pd.concat(preds, axis=0)
    precision, recall, ic, rank_ic = metric_fn(preds)
    scores = ic
    # scores = (precision[3] + precision[5] + precision[10] + precision[30])/4.0
    # scores = -1.0 * mse

    writer.add_scalar(prefix+'/Loss', np.mean(losses), epoch)
    writer.add_scalar(prefix+'/std(Loss)', np.std(losses), epoch)
    writer.add_scalar(prefix+'/'+args.metric, np.mean(scores), epoch)
    writer.add_scalar(prefix+'/std('+args.metric+')', np.std(scores), epoch)

    return np.mean(losses), scores, precision, recall, ic, rank_ic


#使用训练好的模型进行推断（inference）或预测
def inference(model, data_loader, stock2concept_matrix=None):

    # model.eval() 来禁用模型中的一些训练相关特性，以确保在推断过程中不会进行梯度计算
    model.eval()

    preds = []
    for i, slc in tqdm(data_loader.iter_daily(), total=data_loader.daily_length):
        #数据进行了切片
        #data_loader.iter_daily() 用于获取每个数据切片
        #，然后通过 data_loader.get(slc) 获取每个切片的数据。
        
        feature, label, market_value, stock_index, index = data_loader.get(slc)
        with torch.no_grad():
            if args.model_name == 'HIST':
                pred = model(feature, stock2concept_matrix[stock_index], market_value)
                #hist需要额外的输入数据
            else:
                pred = model(feature)
                
            preds.append(pd.DataFrame({ 'score': pred.cpu().numpy(), 'label': label.cpu().numpy(),  }, index=index))
            #preds 列表包含了每个数据切片的模型预测结果和标签，可以用于后续的分析、可视化或存储。
    preds = pd.concat(preds, axis=0)
    return preds

#加载数据
#总的来说，这个函数负责设置用于训练和评估机器学习模型的数据流水线，使用金融或时间序列数据。它很可能是更大代码库的一部分，其中还包括了机器学习模型和训练循环等其他组件。
def create_loaders(args):

    start_time = datetime.datetime.strptime(args.train_start_date, '%Y-%m-%d')
    end_time = datetime.datetime.strptime(args.test_end_date, '%Y-%m-%d')
    train_end_time = datetime.datetime.strptime(args.train_end_date, '%Y-%m-%d')

    hanlder = {'class': 'Alpha360', 'module_path': 'qlib.contrib.data.handler', 'kwargs': {'start_time': start_time, 'end_time': end_time, 'fit_start_time': start_time, 'fit_end_time': train_end_time, 'instruments': args.data_set, 'infer_processors': [{'class': 'RobustZScoreNorm', 'kwargs': {'fields_group': 'feature', 'clip_outlier': True}}, {'class': 'Fillna', 'kwargs': {'fields_group': 'feature'}}], 'learn_processors': [{'class': 'DropnaLabel'}, {'class': 'CSRankNorm', 'kwargs': {'fields_group': 'label'}}], 'label': ['Ref($close, -1) / $close - 1']}}
    segments =  { 'train': (args.train_start_date, args.train_end_date), 'valid': (args.valid_start_date, args.valid_end_date), 'test': (args.test_start_date, args.test_end_date)}
    dataset = DatasetH(hanlder,segments)

    df_train, df_valid, df_test = dataset.prepare( ["train", "valid", "test"], col_set=["feature", "label"], data_key=DataHandlerLP.DK_L,)
    import pickle5 as pickle
    with open(args.market_value_path, "rb") as fh:
        df_market_value = pickle.load(fh)
    #df_market_value = pd.read_pickle(args.market_value_path)
    df_market_value = df_market_value/1000000000
    stock_index = np.load(args.stock_index, allow_pickle=True).item()

    start_index = 0
    slc = slice(pd.Timestamp(args.train_start_date), pd.Timestamp(args.train_end_date))
    df_train['market_value'] = df_market_value[slc]
    df_train['market_value'] = df_train['market_value'].fillna(df_train['market_value'].mean())
    df_train['stock_index'] = 733
    df_train['stock_index'] = df_train.index.get_level_values('instrument').map(stock_index).fillna(733).astype(int)

    train_loader = DataLoader(df_train["feature"], df_train["label"], df_train['market_value'], df_train['stock_index'], batch_size=args.batch_size, pin_memory=args.pin_memory, start_index=start_index, device = device)

    slc = slice(pd.Timestamp(args.valid_start_date), pd.Timestamp(args.valid_end_date))
    df_valid['market_value'] = df_market_value[slc]
    df_valid['market_value'] = df_valid['market_value'].fillna(df_train['market_value'].mean())
    df_valid['stock_index'] = 733
    df_valid['stock_index'] = df_valid.index.get_level_values('instrument').map(stock_index).fillna(733).astype(int)
    start_index += len(df_valid.groupby(level=0).size())

    valid_loader = DataLoader(df_valid["feature"], df_valid["label"], df_valid['market_value'], df_valid['stock_index'], pin_memory=True, start_index=start_index, device = device)
    
    slc = slice(pd.Timestamp(args.test_start_date), pd.Timestamp(args.test_end_date))
    df_test['market_value'] = df_market_value[slc]
    df_test['market_value'] = df_test['market_value'].fillna(df_train['market_value'].mean())
    df_test['stock_index'] = 733
    df_test['stock_index'] = df_test.index.get_level_values('instrument').map(stock_index).fillna(733).astype(int)
    start_index += len(df_test.groupby(level=0).size())

    test_loader = DataLoader(df_test["feature"], df_test["label"], df_test['market_value'], df_test['stock_index'], pin_memory=True, start_index=start_index, device = device)

    return train_loader, valid_loader, test_loader


def main(args):
    seed = np.random.randint(1000000)
    np.random.seed(seed)
    torch.manual_seed(seed)
    suffix = "%s_dh%s_dn%s_drop%s_lr%s_bs%s_seed%s%s"%(
        args.model_name, args.hidden_size, args.num_layers, args.dropout,
        args.lr, args.batch_size, args.seed, args.annot
    )

    output_path = args.outdir
    if not output_path:
        output_path = './output/' + suffix
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if not args.overwrite and os.path.exists(output_path+'/'+'info.json'):
        print('already runned, exit.')
        return

    writer = SummaryWriter(log_dir=output_path)

    global global_log_file
    global_log_file = output_path + '/' + args.name + '_run.log'

    pprint('create loaders...')
    train_loader, valid_loader, test_loader = create_loaders(args)

    stock2concept_matrix = np.load(args.stock2concept_matrix) 
    if args.model_name == 'HIST':
        stock2concept_matrix = torch.Tensor(stock2concept_matrix).to(device)

    all_precision = []
    all_recall = []
    all_ic = []
    all_rank_ic = []
    for times in range(args.repeat):
        pprint('create model...')
        if args.model_name == 'SFM':
            model = get_model(args.model_name)(d_feat = args.d_feat, output_dim = 32, freq_dim = 25, hidden_size = args.hidden_size, dropout_W = 0.5, dropout_U = 0.5, device = device)
        elif args.model_name == 'ALSTM':
            model = get_model(args.model_name)(args.d_feat, args.hidden_size, args.num_layers, args.dropout, 'LSTM')
        elif args.model_name == 'Transformer':
            model = get_model(args.model_name)(args.d_feat, args.hidden_size, args.num_layers, dropout=0.5)
        elif args.model_name == 'HIST':
            model = get_model(args.model_name)(d_feat = args.d_feat, num_layers = args.num_layers, K = args.K)
        else:
            model = get_model(args.model_name)(d_feat = args.d_feat, num_layers = args.num_layers)
        
        model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        best_score = -np.inf
        best_epoch = 0
        stop_round = 0
        best_param = copy.deepcopy(model.state_dict())
        params_list = collections.deque(maxlen=args.smooth_steps)
        for epoch in range(args.n_epochs):
            pprint('Running', times,'Epoch:', epoch)

            pprint('training...')
            train_epoch(epoch, model, optimizer, train_loader, writer, args, stock2concept_matrix)
            torch.save(model.state_dict(), output_path+'/model.bin.e'+str(epoch))
            torch.save(optimizer.state_dict(), output_path+'/optimizer.bin.e'+str(epoch))

            params_ckpt = copy.deepcopy(model.state_dict())
            params_list.append(params_ckpt)
            avg_params = average_params(params_list)
            model.load_state_dict(avg_params)

            pprint('evaluating...')
            train_loss, train_score, train_precision, train_recall, train_ic, train_rank_ic = test_epoch(epoch, model, train_loader, writer, args, stock2concept_matrix, prefix='Train')
            val_loss, val_score, val_precision, val_recall, val_ic, val_rank_ic = test_epoch(epoch, model, valid_loader, writer, args, stock2concept_matrix, prefix='Valid')
            test_loss, test_score, test_precision, test_recall, test_ic, test_rank_ic = test_epoch(epoch, model, test_loader, writer, args, stock2concept_matrix, prefix='Test')

            pprint('train_loss %.6f, valid_loss %.6f, test_loss %.6f'%(train_loss, val_loss, test_loss))
            pprint('train_score %.6f, valid_score %.6f, test_score %.6f'%(train_score, val_score, test_score))
            # pprint('train_mse %.6f, valid_mse %.6f, test_mse %.6f'%(train_mse, val_mse, test_mse))
            # pprint('train_mae %.6f, valid_mae %.6f, test_mae %.6f'%(train_mae, val_mae, test_mae))
            pprint('train_ic %.6f, valid_ic %.6f, test_ic %.6f'%(train_ic, val_ic, test_ic))
            pprint('train_rank_ic %.6f, valid_rank_ic %.6f, test_rank_ic %.6f'%(train_rank_ic, val_rank_ic, test_rank_ic))
            pprint('Train Precision: ', train_precision)
            pprint('Valid Precision: ', val_precision)
            pprint('Test Precision: ', test_precision)
            pprint('Train Recall: ', train_recall)
            pprint('Valid Recall: ', val_recall)
            pprint('Test Recall: ', test_recall)
            model.load_state_dict(params_ckpt)

            if val_score > best_score:
                best_score = val_score
                stop_round = 0
                best_epoch = epoch
                best_param = copy.deepcopy(avg_params)
            else:
                stop_round += 1
                if stop_round >= args.early_stop:
                    pprint('early stop')
                    break

        pprint('best score:', best_score, '@', best_epoch)
        model.load_state_dict(best_param)
        torch.save(best_param, output_path+'/model.bin')

        pprint('inference...')
        res = dict()
        for name in ['train', 'valid', 'test']:

            pred= inference(model, eval(name+'_loader'), stock2concept_matrix)
            pred.to_pickle(output_path+'/pred.pkl.'+name+str(times))

            precision, recall, ic, rank_ic = metric_fn(pred)

            pprint(('%s: IC %.6f Rank IC %.6f')%(
                        name, ic.mean(), rank_ic.mean()))
            pprint(name, ': Precision ', precision)
            pprint(name, ': Recall ', recall)
            res[name+'-IC'] = ic
            # res[name+'-ICIR'] = ic.mean() / ic.std()
            res[name+'-RankIC'] = rank_ic
            # res[name+'-RankICIR'] = rank_ic.mean() / rank_ic.std()
        
        all_precision.append(list(precision.values()))
        all_recall.append(list(recall.values()))
        all_ic.append(ic)
        all_rank_ic.append(rank_ic)

        pprint('save info...')
        writer.add_hparams(
            vars(args),
            {
                'hparam/'+key: value
                for key, value in res.items()
            }
        )

        info = dict(
            config=vars(args),
            best_epoch=best_epoch,
            best_score=res,
        )
        default = lambda x: str(x)[:10] if isinstance(x, pd.Timestamp) else x
        with open(output_path+'/info.json', 'w') as f:
            json.dump(info, f, default=default, indent=4)
    pprint(('IC: %.4f (%.4f), Rank IC: %.4f (%.4f)')%(np.array(all_ic).mean(), np.array(all_ic).std(), np.array(all_rank_ic).mean(), np.array(all_rank_ic).std()))
    precision_mean = np.array(all_precision).mean(axis= 0)
    precision_std = np.array(all_precision).std(axis= 0)
    N = [1, 3, 5, 10, 20, 30, 50, 100]
    for k in range(len(N)):
        pprint (('Precision@%d: %.4f (%.4f)')%(N[k], precision_mean[k], precision_std[k]))

    pprint('finished.')
'''
训练循环部分：

如果验证集的性能 (val_score) 比历史最佳性能 (best_score) 更好，更新 best_score、stop_round（停止轮数）、best_epoch（最佳时代） 和 best_param（最佳模型参数）。
否则，递增 stop_round，如果 stop_round 达到指定的早期停止轮数 (args.early_stop)，则打印 "early stop" 并跳出循环。
打印最佳性能 (best_score) 和其出现的时代 (best_epoch)。

加载最佳模型参数并将其保存到文件中。

进行模型的推断操作，对训练、验证和测试数据集进行预测，然后计算一些性能指标，如精度、召回率、信息系数 (IC) 和排名信息系数 (Rank IC)。这些指标会被记录在名为 res 的字典中。

将预测结果保存到文件（'pred.pkl' 文件中），并打印每个数据集（train、valid、test）的 IC 和 Rank IC。

将性能指标和参数配置记录到 TensorBoard 的事件文件中，以便进行可视化。

创建一个包含实验信息的字典 info，包括参数配置、最佳时代和性能指标。

计算 IC 和 Rank IC 的平均值和标准差，并将它们打印出来。

计算 Precision@N 指标（N=1, 3, 5, 10, 20, 30, 50, 100）的平均值和标准差，并将它们打印出来。

最后，打印 "finished" 表示训练和评估过程已完成。

'''

class ParseConfigFile(argparse.Action):

    def __call__(self, parser, namespace, filename, option_string=None):

        if not os.path.exists(filename):
            raise ValueError('cannot find config at `%s`'%filename)

        with open(filename) as f:
            config = json.load(f)
            for key, value in config.items():
                setattr(namespace, key, value)


def parse_args():

    parser = argparse.ArgumentParser()

    # model
    parser.add_argument('--model_name', default='HIST')
    parser.add_argument('--d_feat', type=int, default=6)
    parser.add_argument('--hidden_size', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--K', type=int, default=1)

    # training
    parser.add_argument('--n_epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--early_stop', type=int, default=30)
    parser.add_argument('--smooth_steps', type=int, default=5)
    parser.add_argument('--metric', default='IC')
    parser.add_argument('--loss', default='mse')
    parser.add_argument('--repeat', type=int, default=10)

    # data
    parser.add_argument('--data_set', type=str, default='csi300')
    parser.add_argument('--pin_memory', action='store_false', default=True)
    parser.add_argument('--batch_size', type=int, default=-1) # -1 indicate daily batch
    parser.add_argument('--least_samples_num', type=float, default=1137.0) 
    parser.add_argument('--label', default='') # specify other labels
    parser.add_argument('--train_start_date', default='2007-01-01')
    parser.add_argument('--train_end_date', default='2014-12-31')
    parser.add_argument('--valid_start_date', default='2015-01-01')
    parser.add_argument('--valid_end_date', default='2016-12-31')
    parser.add_argument('--test_start_date', default='2017-01-01')
    parser.add_argument('--test_end_date', default='2020-12-31')

    # other
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--annot', default='')
    parser.add_argument('--config', action=ParseConfigFile, default='')
    parser.add_argument('--name', type=str, default='csi300_HIST')

    # input for csi 300
    parser.add_argument('--market_value_path', default='./data/csi300_market_value_07to20.pkl')
    parser.add_argument('--stock2concept_matrix', default='./data/csi300_stock2concept.npy')
    parser.add_argument('--stock_index', default='./data/csi300_stock_index.npy')

    parser.add_argument('--outdir', default='./output/csi300_HIST')
    parser.add_argument('--overwrite', action='store_true', default=False)

    args = parser.parse_args()

    return args


if __name__ == '__main__':

    args = parse_args()
    main(args)
