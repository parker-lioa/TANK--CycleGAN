



def train_opt(parser):

    # special options

    parser.add_argument('--amp_train', action='store_true')

    # path

    parser.add_argument('--model_path', type=str, default='./models')
    parser.add_argument('--save_period', type=int, default=100)
    parser.add_argument('--save_image_period', type=int, default=20)
    parser.add_argument('--log_path', type=str, default='./logs')
    parser.add_argument('--data_path1', type=str, default='./datasets/A')
    parser.add_argument('--data_path2', type=str, default='./datasets/B')
    parser.add_argument('--pre_train', action='store_true')
    parser.add_argument('--pre_epoch', type=int, default=0)

    # model-parameters

    parser.add_argument('--A_dim', type=int, default=1)
    parser.add_argument('--B_dim', type=int, default=1)

    # hyper-parameters

    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--loss', type=str, default='lsgan')
    parser.add_argument('--n_critic', type=int, default=1)
    parser.add_argument('--clip_value', type=float, default=0.01)
    parser.add_argument('--cycle_weight', type=float, default=10)
    parser.add_argument('--idt_weight', type=float, default=0)