import argparse
from model.utils.config import cfg, cfg_from_file, cfg_from_list


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--log_ckpt_name', dest='log_ckpt_name',
                        help='saved log and ckpt dir name',
                        default='cs2cs_fg', type=str)
    parser.add_argument('--lc', dest='lc',
                        help='whether use context vector for pixel level',
                        default= True)
    parser.add_argument('--gc', dest='gc',
                        help='whether use context vector for global level',
                        default= True)
    parser.add_argument('--LA_ATT', dest='LA_ATT',
                        help='whether to use local attention',
                        default=True)
    parser.add_argument('--MID_ATT', dest='MID_ATT',
                        help='whether to use middle attention',
                        default=True)
    parser.add_argument('--dataset', dest='dataset',
                        help='source training dataset',
                        default='cs_combine_fg', type=str)
    parser.add_argument('--dataset_t', dest='dataset_t',
                        help='target training dataset',
                        default='cs_fg_combine', type=str)
    parser.add_argument('--net', dest='net',
                        help='vgg16, res101 res50',
                        default='vgg16', type=str)
    parser.add_argument(
        "--teacher_alpha",
        dest="teacher_alpha",
        help="Teacher EMA alpha (decay)",
        default=0.99,
        type=float,
    )
    parser.add_argument(
        "--student_load_name",
        dest="student_load_name",
        help="path to load student models",
        default="models",
        type=str,
    )
    parser.add_argument(
        "--teacher_load_name",
        dest="teacher_load_name",
        help="path to load teacher models",
        default="models",
        type=str,
    )
    parser.add_argument(
        "--th",
        dest="threshold",
        help="threshold for pseudo label",
        default=0.6,
        type=float,
    )
    parser.add_argument(
        "--th_l",
        dest="threshold_low",
        help="threshold for pseudo label",
        default=0.2,
        type=float,
    )
    parser.add_argument(
        "--person_th",
        dest="person_th",
        help="threshold for pseudo label",
        default=0.6,
        type=float,
    )


    parser.add_argument('--alpha', dest='alpha',
                        help='trade-off parameter',
                        default=0.1, type=float)
    parser.add_argument('--la1', dest='lambda1',
                        help='trade-off parameter',
                        default=0.5, type=float)
    parser.add_argument('--la2', dest='lambda2',
                        help='trade-off parameter',
                        default=0.1, type=float)

    parser.add_argument('--beta', dest='beta',
                        help='trade-off parameter between detection loss and domain-alignment loss. Used for Car datasets',
                        default=0.1, type=float)

    ##########################################################################
    ##########################################################################
    parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                        help='step to do learning rate decay, unit is epoch',
                        default=[5], type=int)
    parser.add_argument('--start_epoch', dest='start_epoch',
                        help='starting epoch',
                        default=1, type=int)
    parser.add_argument('--epochs', dest='max_epochs',
                        help='number of epochs to train',
                        default=6, type=int)
    parser.add_argument('--gamma', dest='gamma',
                        help='value of gamma',
                        default=3, type=float)
    parser.add_argument('--disp_interval', dest='disp_interval',
                        help='number of iterations to display',
                        default=100, type=int)
    parser.add_argument('--checkpoint_interval', dest='checkpoint_interval',
                        help='number of iterations to display',
                        default=10000, type=int)

    parser.add_argument('--save_dir', dest='save_dir',
                        help='directory to save models', default="models",
                        type=str)
    parser.add_argument('--load_name', dest='load_name',
                        help='path to load models', default="",
                        type=str)
    parser.add_argument('--nw', dest='num_workers',
                        help='number of worker to load data',
                        default=0, type=int)
    parser.add_argument('--cuda', dest='cuda',
                        help='whether use CUDA',
                        default= True)

    parser.add_argument('--detach', dest='detach',
                        help='whether use detach',
                        action='store_false')
    parser.add_argument('--ef', dest='ef',
                        help='whether use exponential focal loss',
                        action='store_true')
    parser.add_argument('--ls', dest='large_scale',
                        help='whether use large imag scale',
                        default=False,
                        action='store_true')
    parser.add_argument('--mGPUs', dest='mGPUs',
                        help='whether use multiple GPUs',
                        action='store_true')
    parser.add_argument('--bs', dest='batch_size',
                        help='batch_size',
                        default=1, type=int)
    parser.add_argument('--cag', dest='class_agnostic',
                        help='whether perform class_agnostic bbox regression',
                        action='store_true')

    # config optimization
    parser.add_argument('--o', dest='optimizer',
                        help='training optimizer',
                        default="sgd", type=str)
    parser.add_argument('--lr', dest='lr',
                        help='starting learning rate',
                        default=0.001, type=float)
    parser.add_argument('--eta', dest='eta',
                        help='trade-off parameter between detection loss and domain-alignment loss. Used for Car datasets',
                        default=0.1, type=float)
    parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                        help='learning rate decay ratio',
                        default=0.1, type=float)
    parser.add_argument('--s', dest='session',
                        help='training session',
                        default=1, type=int)
    parser.add_argument('--r', dest='resume',
                        help='resume checkpoint or not',
                        default=False, type=bool)
    parser.add_argument('--checksession', dest='checksession',
                        help='checksession to load model',
                        default=1, type=int)
    parser.add_argument('--checkepoch', dest='checkepoch',
                        help='checkepoch to load model',
                        default=1, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint',
                        help='checkpoint to load model',
                        default=0, type=int)
    # log and diaplay
    parser.add_argument('--use_tfb', dest='use_tfboard',
                        help='whether use tensorboard',
                        default= True)
    parser.add_argument('--image_dir', dest='image_dir',
                        help='directory to load images for demo',
                        default="images")

    parser.add_argument('--log_file',
                        help='log_dir',
                        default="temp.txt")
    parser.add_argument('--up_th', dest='up_th',
                        help='directory to load images for demo',
                        default=85) #80
    parser.add_argument('--down_th', dest='down_th',
                        help='directory to load images for demo',
                        default=50) #40

    args = parser.parse_args()
    return args


def set_dataset_args(args, test=False):
    if not test:
        if args.dataset == "hos":
            args.imdb_name = "hos_2007_trainval"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
        elif args.dataset == "inb":
            args.imdb_name = "inb_2007_trainval"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
        elif args.dataset == "voc_0712":
            args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
        elif args.dataset == "cs":
            args.imdb_name = "cs_2007_train_s"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
        elif args.dataset == "cs_combine_fg":
            args.imdb_name = "cs_2007_train_combine_fg"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
        elif args.dataset == "sim":
            args.imdb_name = "sim10k_2012_trainval"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
        elif args.dataset == "sim_combine":
            args.imdb_name = "sim10k_2012_trainval_combine"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
        elif args.dataset == "cs_fg":
            args.imdb_name = "cs_fg_2007_train_t"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
        elif args.dataset == "cs_fg_error":
            args.imdb_name = "cs_fg_2007_error"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
        elif args.dataset == "cs_rtts":
            args.imdb_name = "cs_rtts_2007_train_t"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
        elif args.dataset == "cs_car":
            args.imdb_name = "cs_car_2007_train"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
        elif args.dataset == "kitti_car":
            args.imdb_name = "kitti_car_2007_trainval"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
        elif args.dataset == "kitti":
            args.imdb_name = "kitti_2007_trainval"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
        elif args.dataset == "kitti_car_combine":
            args.imdb_name = "kitti_car_2007_trainval_combine"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']

        elif args.dataset == "cs_fg_similar":
            args.imdb_name = "cs_fg_similar_2007_train"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
        elif args.dataset == "cs_fg_error":
            args.imdb_name = "cs_fg_similar_2007_error"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
        elif args.dataset == "cs_fg_disimilar":
            args.imdb_name = "cs_fg_disimilar_2007_train"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']

        elif args.dataset == "cs_fg_disimilar_trained":
            args.imdb_name = "cs_fg_disimilar_2007_trained"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
        elif args.dataset == "cs_fg_similar_trained":
            args.imdb_name = "cs_fg_similar_2007_trained"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']

        elif args.dataset == "cs_similar":
            args.imdb_name = "cs_similar_2007_train"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
        elif args.dataset == "cs_disimilar":
            args.imdb_name = "cs_disimilar_2007_train"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']

        elif args.dataset == "cs_car_similar":
            args.imdb_name = "cs_car_similar_2007_train"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
        elif args.dataset == "cs_car_disimilar":
            args.imdb_name = "cs_car_disimilar_2007_train"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']

        elif args.dataset == "cs_car_kitti_similar":
            args.imdb_name = "cs_car_kitti_similar_2007_train"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
        elif args.dataset == "cs_car_kitti_disimilar":
            args.imdb_name = "cs_car_kitti_disimilar_2007_train"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']

        elif args.dataset == "cs_car_sim10k_similar":
            args.imdb_name = "cs_car_sim10k_similar_2007_train"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
        elif args.dataset == "cs_car_sim10k_disimilar":
            args.imdb_name = "cs_car_sim10k_disimilar_2007_train"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']

        elif args.dataset == "bdd100k":
            args.imdb_name = "bdd100k_train"
            args.imdbval_name = "bdd100k_val"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']

        elif args.dataset == "bdd100k_similar":
            args.imdb_name = "bdd100k_similar_train"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']
        elif args.dataset == "bdd100k_disimilar":
            args.imdb_name = "bdd100k_disimilar_train"
            args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']
        ############################ target ################################################

        if args.dataset_t == "hos":
            args.imdb_name_target = "hos_2007_trainval"
        elif args.dataset_t == "inb":
            args.imdb_name_target = "inb_2007_trainval"
        elif args.dataset_t == "clipart":
            args.imdb_name_target = "clipart_2007_train"
        elif args.dataset_t == "cs":
            args.imdb_name_target = "cs_2007_train"
        elif args.dataset_t == "cs_fg":
            args.imdb_name_target = "cs_fg_2007_train_t"
        elif args.dataset_t == "cs_rtts":
            args.imdb_name_target = "cs_rtts_2007_train_t"
        elif args.dataset_t == "cs_fg_error":
            args.imdb_name_target = "cs_fg_2007_error"
        elif args.dataset_t == "cs_fg_combine":
            args.imdb_name_target = "cs_fg_2007_train_combine"
        elif args.dataset_t == "cs_car":
            args.imdb_name_target = "cs_car_2007_train"
        elif args.dataset_t == "cs_car_combine":
            args.imdb_name_target = "cs_car_2007_train_combine"
        elif args.dataset_t == "cs_car_combine_kt":
            args.imdb_name_target = "cs_car_2007_train_combine_kt"
        elif args.dataset_t == "kitti_car":
            args.imdb_name_target = "kitti_car_2007_trainval"
        elif args.dataset_t == "kitti":
            args.imdb_name_target = "kitti_2007_trainval"

        elif args.dataset_t == "cs_fg_disimilar":
            args.imdb_name_target = "cs_fg_disimilar_2007_train"
        elif args.dataset_t == "cs_fg_similar":
            args.imdb_name_target = "cs_fg_similar_2007_train"

        elif args.dataset_t == "cs_fg_disimilar_trained":
            args.imdb_name_target = "cs_fg_disimilar_2007_trained"
        elif args.dataset_t == "cs_fg_similar_trained":
            args.imdb_name_target = "cs_fg_similar_2007_trained"

        elif args.dataset_t == "cs_disimilar":
            args.imdb_name_target = "cs_disimilar_2007_train"
        elif args.dataset_t == "cs_similar":
            args.imdb_name_target = "cs_similar_2007_train"

        elif args.dataset_t == "cs_car_disimilar":
            args.imdb_name_target = "cs_car_disimilar_2007_train"
        elif args.dataset_t == "cs_car_similar":
            args.imdb_name_target = "cs_car_similar_2007_train"

        elif args.dataset_t == "cs_car_kitti_disimilar":
            args.imdb_name_target = "cs_car_kitti_disimilar_2007_train"
        elif args.dataset_t == "cs_car_kitti_similar":
            args.imdb_name_target = "cs_car_kitti_similar_2007_train"

        elif args.dataset_t == "cs_car_sim10k_disimilar":
            args.imdb_name_target = "cs_car_sim10k_disimilar_2007_train"
        elif args.dataset_t == "cs_car_sim10k_similar":
            args.imdb_name_target = "cs_car_sim10k_similar_2007_train"

        elif args.dataset_t == "bdd100k":
            args.imdb_name_target = "bdd100k_train"

        elif args.dataset_t == "bdd100k_similar":
            args.imdb_name_target = "bdd100k_similar_train"
        elif args.dataset_t == "bdd100k_disimilar":
            args.imdb_name_target = "bdd100k_disimilar_train"

    else:
        if args.dataset_t == "hos":
            args.imdb_name_target = "hos_2007_test"
            args.set_cfgs_target = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
        elif args.dataset_t == "cs_fg":
            args.imdb_name_target = "cs_fg_2007_test_t"
            args.set_cfgs_target = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
        elif args.dataset_t == "inb":
            args.imdb_name_target = "inb_2007_trainval"
            args.set_cfgs_target = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
        elif args.dataset_t == "clipart":
            args.imdb_name_target = "clipart_2007_val"
            args.set_cfgs_target = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
        elif args.dataset_t == "cs_fg":
            args.imdb_name_target = "cs_fg_2007_test_t"
            args.imdb_name_target_train = "cs_fg_2007_train"
            args.set_cfgs_target = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
        elif args.dataset_t == "cs_rtts":
            args.imdb_name_target = "cs_rtts_2007_test_t"
            args.imdb_name_target_train = "cs_rtts_2007_train"
            args.set_cfgs_target = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
        elif args.dataset_t == "cs_fg_fn":
            args.imdb_name_target = "cs_fg_2007_fn_test"
            args.imdb_name_target_train = "cs_fg_2007_train"
            args.set_cfgs_target = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
        elif args.dataset_t == "cs_fg_error":
            args.imdb_name_target = "cs_fg_2007_error"
            args.imdb_name_target_train = "cs_fg_2007_train"
            args.set_cfgs_target = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
        elif args.dataset_t == "cs_fg_combine":
            args.imdb_name_target = "cs_fg_2007_val"
            args.set_cfgs_target = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
        elif args.dataset_t == "cs_car":
            args.imdb_name_target = "cs_car_2007_val"
            args.set_cfgs_target = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
        elif args.dataset_t == "cs_car_combine":
            args.imdb_name_target = "cs_car_2007_val"
            args.set_cfgs_target = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
        elif args.dataset_t == "cs_car_combine_kt":
            args.imdb_name_target = "cs_car_2007_val"
            args.set_cfgs_target = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
        elif args.dataset_t == "cs":
            args.imdb_name_target = "cs_2007_test_s"
            args.set_cfgs_target = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
        elif args.dataset_t == "kitti_car":
            args.imdb_name_target = "kitti_car_2007_trainval"
            args.set_cfgs_target = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
        elif args.dataset_t == "kitti":
            args.imdb_name_target = "kitti_2007_trainval"
            args.set_cfgs_target = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
        elif args.dataset_t == "voc_0712":
            args.imdb_name_target = "voc_2012_val"
            args.set_cfgs_target = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']

        elif args.dataset_t == "cs_fg_disimilar":
            args.imdb_name_target = "cs_fg_disimilar_2007_train"
            args.set_cfgs_target = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
        elif args.dataset_t == "cs_fg_similar":
            args.imdb_name_target = "cs_fg_similar_2007_train"
            args.set_cfgs_target = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']

        elif args.dataset_t == "cs_fg_disimilar_trained":
            args.imdb_name_target = "cs_fg_disimilar_2007_trained"
            args.set_cfgs_target = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
        elif args.dataset_t == "cs_fg_similar_trained":
            args.imdb_name_target = "cs_fg_similar_2007_trained"
            args.set_cfgs_target = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']

        elif args.dataset_t == "cs_disimilar":
            args.imdb_name_target = "cs_disimilar_2007_train"
            args.set_cfgs_target = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
        elif args.dataset_t == "cs_similar":
            args.imdb_name_target = "cs_similar_2007_train"
            args.set_cfgs_target = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']

        elif args.dataset_t == "cs_car_disimilar":
            args.imdb_name_target = "cs_car_disimilar_2007_train"
            args.set_cfgs_target = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
        elif args.dataset_t == "cs_car_similar":
            args.imdb_name_target = "cs_car_similar_2007_train"
            args.set_cfgs_target = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']

        elif args.dataset_t == "cs_car_kitti_disimilar":
            args.imdb_name_target = "cs_car_kitti_disimilar_2007_train"
            args.set_cfgs_target = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
        elif args.dataset_t == "cs_car_kitti_similar":
            args.imdb_name_target = "cs_car_kitti_similar_2007_train"
            args.set_cfgs_target = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']

        elif args.dataset_t == "cs_car_sim10k_disimilar":
            args.imdb_name_target = "cs_car_sim10k_disimilar_2007_train"
            args.set_cfgs_target = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
        elif args.dataset_t == "cs_car_sim10k_similar":
            args.imdb_name_target = "cs_car_sim10k_similar_2007_train"
            args.set_cfgs_target = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']

        elif args.dataset_t == "bdd100k":
            args.imdb_name_target = "bdd100k_val"
            args.set_cfgs_target = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']

        elif args.dataset_t == "bdd100k_similar":
            args.imdb_name_target = "bdd100k_similar_train"
            args.set_cfgs_target = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']
        elif args.dataset_t == "bdd100k_disimilar":
            args.imdb_name_target = "bdd100k_disimilar_train"
            args.set_cfgs_target = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']

    args.cfg_file = "./cfgs/{}_ls.yml".format(args.net) if args.large_scale else "./cfgs/{}.yml".format(args.net)


    return args
