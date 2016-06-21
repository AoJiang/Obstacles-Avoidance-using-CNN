import find_mxnet
import mxnet as mx
import logging
import os
import numpy as np
import cv2
from time import time
def data_loader(args, data_shape, kv):
    flat = False if len(data_shape) == 3 else True
    print "Data_shape:" + str(data_shape)
    #print flat

    train = mx.io.ImageRecordIter(
        mean_r  = 118.659101056,
        mean_g  = 114.800727896,
        mean_b  = 114.864786915,
        #num_part = 5,
        path_imgrec='rgb_tr_train.rec',
        #path_imgrec = args.data_dir + args.trainfile,
        #mean_img    = args.data_dir + "mean.bin",
        data_shape  = data_shape,
        batch_size  = args.batch_size,
        #rand_crop   = True,
        #rand_mirror = True,
      #  flat        = flat,
        seed        = int(time()),
        #max_img_size  = 100,
        #max_shear_ratio = 0.3,
        #fill_value  = 200,
        #max_random_contrast = 10,
        #max_rotate_angle = 20,
        #inter_method  = 10,
        shuffle     = True)

    test = mx.io.ImageRecordIter(
        mean_r=118.659101056,
        mean_g=114.800727896,
        mean_b=114.864786915,
        # num_part = 5,
        path_imgrec='rgb_te_train.rec',
        # path_imgrec = args.data_dir + args.trainfile,
        # mean_img    = args.data_dir + "mean.bin",
        data_shape=data_shape,
        batch_size=args.batch_size,
        # rand_crop   = True,
        # rand_mirror = True,
        #  flat        = flat,
        seed=int(time()),
        # max_img_size  = 100,
        # max_shear_ratio = 0.3,
        # fill_value  = 200,
        # max_random_contrast = 10,
        # max_rotate_angle = 20,
        # inter_method  = 10,
        shuffle=True)

    '''
    val = mx.io.ImageRecordIter(
        path_imgrec = args.data_dir + "train.rec",
        #mean_img    = args.data_dir + "mean.bin",
        rand_crop   = False,
        rand_mirror = False,
        data_shape  = data_shape,
        batch_size  = args.batch_size)
    '''
    #train = np.array([])
    return (train,test)

    #return (train, val)
def fit(args, network,  data_shape, batch_end_callback=None):
    # kvstore
    kv = mx.kvstore.create(args.kv_store)

    # logging
    head = '%(asctime)-15s Node[' + str(kv.rank) + '] %(message)s'
    if 'log_file' in args and args.log_file is not None:
        log_file = args.log_file
        log_dir = args.log_dir
        log_file_full_name = os.path.join(log_dir, log_file)
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        logger = logging.getLogger()
        handler = logging.FileHandler(log_file_full_name)
        formatter = logging.Formatter(head)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        logger.info('start with arguments %s', args)
    else:
        logging.basicConfig(level=logging.DEBUG, format=head)
        logging.info('start with arguments %s', args)

    # load model
    model_prefix = args.model_prefix
    #if model_prefix is not None:
    #    model_prefix += "-%d" % (kv.rank)
    model_args = {}
    if args.load_epoch is not None:
        assert model_prefix is not None
        tmp = mx.model.FeedForward.load(model_prefix, args.load_epoch)
        model_args = {'arg_params' : tmp.arg_params,
                      'aux_params' : tmp.aux_params,
                      'begin_epoch' : args.begin_epoch}
    # save model
    save_model_prefix = args.save_model_prefix
    if save_model_prefix is None:
        save_model_prefix = model_prefix
    checkpoint = None if save_model_prefix is None else mx.callback.do_checkpoint(save_model_prefix)

    # data
    (train,test) = data_loader(args, data_shape, kv)

    # train
    devs = [mx.cpu(0)] if args.gpus is None else [
        mx.gpu(int(i)) for i in args.gpus.split(',')]

    epoch_size = args.num_examples / args.batch_size

    if args.kv_store == 'dist_sync':
        epoch_size /= kv.num_workers
        model_args['epoch_size'] = epoch_size

    if 'lr_factor' in args and args.lr_factor < 1:
        model_args['lr_scheduler'] = mx.lr_scheduler.FactorScheduler(
            step = max(int(epoch_size * args.lr_factor_epoch), 1),
            factor = args.lr_factor)

    if 'clip_gradient' in args and args.clip_gradient is not None:
        model_args['clip_gradient'] = args.clip_gradient

    # disable kvstore for single device
    if 'local' in kv.type and (
            args.gpus is None or len(args.gpus.split(',')) is 1):
        kv = None

    model = mx.model.FeedForward(
        ctx                = devs,
        symbol             = network,
        num_epoch          = args.num_epochs,
        learning_rate      = args.lr,
        momentum           = 0.9,
        wd                 = 0.00001,

        optimizer          = mx.optimizer.SGD(clip_gradient=2),
        initializer        = mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2.34),
        #initializer        = mx.init.Normal(sigma=0.001),
        **model_args)

    #eval_metrics = ['accuracy']
    eval_metrics = ['acc','ce']
    ## TopKAccuracy only allows top_k > 1
    #for top_k in [1,5]:
    #    eval_metrics.append(mx.metric.create('top_k_accuracy', top_k = top_k))

    if batch_end_callback is not None:
        if not isinstance(batch_end_callback, list):
            batch_end_callback = [batch_end_callback]
    else:
        batch_end_callback = []
    batch_end_callback.append(mx.callback.Speedometer(args.batch_size, 10))

    model.fit(
        X                  = train,
        eval_data          = test,
        eval_metric        = eval_metrics,
        kvstore            = kv,
        batch_end_callback = batch_end_callback,
        epoch_end_callback = checkpoint)

    return model

def predict(model, image, data_shape):
    print image.shape
    image = cv2.resize(image,(data_shape[1],data_shape[2]),interpolation=cv2.INTER_CUBIC)
    image = np.reshape(image, (1, data_shape[0], data_shape[1], data_shape[2]))
    dataiter = mx.io.NDArrayIter(data = image)
    ans = model.predict(dataiter)
    print type(ans)
    print ans
    return np.argmax(ans)
	#ans = model.predict(eval_data=train)
