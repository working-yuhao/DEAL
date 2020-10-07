from argparse import ArgumentParser

def make_args(input = None):
    parser = ArgumentParser()
    # general
    parser.add_argument('--comment', dest='comment', default='0', type=str,
                        help='comment')
    parser.add_argument('--task', dest='task', default='link', type=str,
                        help='link: link prediction; node: node classification')

    parser.add_argument('--dataset', dest='dataset', default='All', type=str,
                        help='All; Cora; grid; communities; ppi')

    parser.add_argument('--gpu', dest='gpu', action='store_true',
                        help='whether use gpu')

    parser.add_argument('--cpu', dest='gpu', action='store_false',
                        help='whether use cpu')

    parser.add_argument('--sa', dest='strong_A', action='store_true',
                        help='use Strong Alignment')

    parser.add_argument('--wa', dest='strong_A', action='store_false',
                        help='use Weak Alignment')

    parser.add_argument('--cuda', dest='cuda', default='0', type=str)

    parser.add_argument('--res_dir', dest='res_dir', default=None, type=str)

    parser.add_argument('--ps', dest='ps', default='', type=str)

    # dataset
    parser.add_argument('--train_ratio', dest='train_ratio', default=0.2, type=float,
                        help='The ratio between the training dataset size and node numbers')

    parser.add_argument('--dropout', dest='dropout', action='store_true',
                        help='whether dropout, default 0.3')
    parser.add_argument('--dropout_no', dest='dropout', action='store_false',
                        help='whether dropout, default 0.3')

    parser.add_argument('--batch_size', dest='batch_size', default=8, type=int) # implemented via accumulating gradient
    parser.add_argument('--layer_num', dest='layer_num', default=2, type=int)
    parser.add_argument('--feature_dim', dest='feature_dim', default=64, type=int)
    parser.add_argument('--hidden_dim', dest='hidden_dim', default=64, type=int)
    parser.add_argument('--output_dim', dest='output_dim', default=64, type=int)

    parser.add_argument('--lr', dest='lr', default=5e-2, type=float)
    parser.add_argument('--epoch_num', dest='epoch_num', default=3000, type=int)
    parser.add_argument('--repeat_num', dest='repeat_num', default=10, type=int) # 10
    parser.add_argument('--epoch_log', dest='epoch_log', default=10, type=int)

    parser.add_argument('--gamma', dest='gamma', default=2, type=float)

    parser.add_argument('--approximate', dest='approximate', default=-1, type=int,
                        help='k-hop shortest path distance. -1 means exact shortest path') # -1, 2

    parser.add_argument('--use_order', dest='use_order', default=False, type=str2bool, 
                        help='whether use Order Strategy, default False')

    parser.add_argument('--ind', dest='inductive', default=False, type=str2bool, 
                        help='Inductive mode, default False')

    parser.add_argument('--cache', dest='cache', default=True, type=str2bool, 
                        help='If use cache, default True')

    parser.add_argument('--remove_link_ratio', dest='remove_link_ratio', default=0.2, type=float)

    parser.add_argument('--rm_feature', dest='rm_feature', default=False, type=str2bool, 
                        help='If use cache, default False')

    parser.add_argument('--mode', dest='train_mode', default='cos', type=str,
                        help='cos, dot, all, pdist, default cos')
    parser.add_argument('--loss', dest='loss', default=None, type=str,
                        help='loss function options: default, etc.')

    parser.add_argument('--attr_model', dest='attr_model', default='Emb', type=str,
                        help='Attribute embedding model, Emb, SAGE, GAT ... , default Emb')

    parser.add_argument('--bce', dest='BCE_mode', default=True, type=str2bool, 
                help='If use BCE_mode, default True')

    parser.set_defaults(
                dataset='CiteSeer', #Cora...
                gpu=True,
                layer_num=2,
                lr=1e-2, 
                repeat_num=1,
                loss = 'default',
                epoch_num = 5000,
                epoch_log = 2,
                task='link',
                train_mode = 'cos',
                train_ratio = 0.1,
        )

    if input is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(input)
    return args

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')