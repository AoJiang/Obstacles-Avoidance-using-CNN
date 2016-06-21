import mxnet as mx
def get_mlp(num_class=3):
    """
    multi-layer perceptron
    """
    data = mx.symbol.Variable('data')
    fc1  = mx.symbol.FullyConnected(data = data, name='fc1', num_hidden=128)
    act1 = mx.symbol.Activation(data = fc1, name='relu1', act_type="relu")
    fc2  = mx.symbol.FullyConnected(data = act1, name = 'fc2', num_hidden = 64)
    act2 = mx.symbol.Activation(data = fc2, name='relu2', act_type="relu")
    fc3  = mx.symbol.FullyConnected(data = act2, name='fc3', num_hidden=num_class)
    mlp  = mx.symbol.SoftmaxOutput(data = fc3, name = 'softmax')
    return mlp

def get_lenet(num_class=3):
    """
    LeCun, Yann, Leon Bottou, Yoshua Bengio, and Patrick
    Haffner. "Gradient-based learning applied to document recognition."
    Proceedings of the IEEE (1998)
    """
    data = mx.symbol.Variable('data')
    # first conv
    conv1 = mx.symbol.Convolution(data=data, kernel=(5,5), num_filter=20)
    tanh1 = mx.symbol.Activation(data=conv1, act_type="tanh")
    pool1 = mx.symbol.Pooling(data=tanh1, pool_type="max",
                              kernel=(2,2), stride=(2,2))
    # second conv
    conv2 = mx.symbol.Convolution(data=pool1, kernel=(5,5), num_filter=50)
    tanh2 = mx.symbol.Activation(data=conv2, act_type="tanh")
    pool2 = mx.symbol.Pooling(data=tanh2, pool_type="max",
                              kernel=(2,2), stride=(2,2))
    # first fullc
    flatten = mx.symbol.Flatten(data=pool2)
    fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=500)
    tanh3 = mx.symbol.Activation(data=fc1, act_type="tanh")
    # second fullc
    fc2 = mx.symbol.FullyConnected(data=tanh3, num_hidden=num_class)
    # loss
    lenet = mx.symbol.SoftmaxOutput(data=fc2, name='softmax')
    return lenet


def get_mynet(num_class=3):
    """
    LeCun, Yann, Leon Bottou, Yoshua Bengio, and Patrick
    Haffner. "Gradient-based learning applied to document recognition."
    Proceedings of the IEEE (1998)
    """
    data = mx.symbol.Variable('data')
    # first conv
    conv1 = mx.symbol.Convolution(data=data, kernel=(5,5), num_filter=50)
    tanh1 = mx.symbol.Activation(data=conv1, act_type="tanh")
    pool1 = mx.symbol.Pooling(data=tanh1, pool_type="sum", kernel=(2,2), stride=(1,1))
    # second conv
    conv2 = mx.symbol.Convolution(data=pool1, kernel=(4,5), num_filter=40)
    tanh2 = mx.symbol.Activation(data=conv2, act_type="relu")
    pool2 = mx.symbol.Pooling(data=tanh2, pool_type="sum", kernel=(2,2), stride=(1,1))
    '''
    conv3 = mx.symbol.Convolution(data=pool2, kernel=(3,3), num_filter=80)
    tanh3 = mx.symbol.Activation(data=conv2, act_type="relu")
    pool3 = mx.symbol.Pooling(data=tanh2, pool_type="sum", kernel=(2,2), stride=(2,2))
    '''
    '''
    conv4 = mx.symbol.Convolution(data=pool3, kernel=(4, 4), num_filter=40)
    tanh4 = mx.symbol.Activation(data=conv4, act_type="tanh")
    pool4 = mx.symbol.Pooling(data=tanh4, pool_type="avg", kernel=(4, 4), stride=(2, 2))
    '''
    # first fullc
    flatten = mx.symbol.Flatten(data=pool2)
    fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=50)
    tanh3 = mx.symbol.Activation(data=fc1, act_type="relu")
    drop3 = mx.symbol.Dropout(data=tanh3, p=0.5, name="drop6")
    '''
    fc2 = mx.symbol.FullyConnected(data=tanh3, num_hidden=10)
    tanh4 = mx.symbol.Activation(data=fc2, act_type="relu")
    drop4 = mx.symbol.Dropout(data=tanh4, p=0.8, name="drop7")
    '''
    # second fullc
    fc2 = mx.symbol.FullyConnected(data=drop3, num_hidden=num_class)
    #return fc2
    # loss
    lenet = mx.symbol.SoftmaxOutput(data=fc2, name='softmax')
    return lenet

def get_alexnet(num_classes = 3):
    input_data = mx.symbol.Variable(name="data")
    # stage 1
    conv1 = mx.symbol.Convolution(
        data=input_data, kernel=(5, 5), stride=(2, 2), num_filter=96)
    relu1 = mx.symbol.Activation(data=conv1, act_type="relu")
    pool1 = mx.symbol.Pooling(
        data=relu1, pool_type="max", kernel=(3, 3), stride=(2,2))
    lrn1 = mx.symbol.LRN(data=pool1, alpha=0.0001, beta=0.75, knorm=1, nsize=5)
    # stage 2
    conv2 = mx.symbol.Convolution(
        data=lrn1, kernel=(5, 5), pad=(2, 2), num_filter=256)
    relu2 = mx.symbol.Activation(data=conv2, act_type="relu")
    pool2 = mx.symbol.Pooling(data=relu2, kernel=(3, 3), stride=(2, 2), pool_type="max")
    lrn2 = mx.symbol.LRN(data=pool2, alpha=0.0001, beta=0.75, knorm=1, nsize=5)
    # stage 3
    conv3 = mx.symbol.Convolution(
        data=lrn2, kernel=(3, 3), pad=(1, 1), num_filter=384)
    relu3 = mx.symbol.Activation(data=conv3, act_type="relu")
    conv4 = mx.symbol.Convolution(
        data=relu3, kernel=(3, 3), pad=(1, 1), num_filter=384)
    relu4 = mx.symbol.Activation(data=conv4, act_type="relu")
    conv5 = mx.symbol.Convolution(
        data=relu4, kernel=(3, 3), pad=(1, 1), num_filter=256)
    relu5 = mx.symbol.Activation(data=conv5, act_type="relu")
    pool3 = mx.symbol.Pooling(data=relu5, kernel=(3, 3), stride=(2, 2), pool_type="max")
    # stage 4
    flatten = mx.symbol.Flatten(data=pool3)
    fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=4096)
    relu6 = mx.symbol.Activation(data=fc1, act_type="relu")
    dropout1 = mx.symbol.Dropout(data=relu6, p=0.5)
    # stage 5
    fc2 = mx.symbol.FullyConnected(data=dropout1, num_hidden=4096)
    relu7 = mx.symbol.Activation(data=fc2, act_type="relu")
    dropout2 = mx.symbol.Dropout(data=relu7, p=0.5)
    # stage 6
    fc3 = mx.symbol.FullyConnected(data=dropout2, num_hidden=num_classes)
    softmax = mx.symbol.SoftmaxOutput(data=fc3, name='softmax')
    return softmax

def ConvFactory(data, num_filter, kernel, stride=(1,1), pad=(0, 0), name=None, suffix=''):
    conv = mx.symbol.Convolution(data=data, num_filter=num_filter, kernel=kernel, stride=stride, pad=pad, name='conv_%s%s' %(name, suffix))
    act = mx.symbol.Activation(data=conv, act_type='relu', name='relu_%s%s' %(name, suffix))
    return act

def InceptionFactory(data, num_1x1, num_3x3red, num_3x3, num_d5x5red, num_d5x5, pool, proj, name):
    # 1x1
    c1x1 = ConvFactory(data=data, num_filter=num_1x1, kernel=(1, 1), name=('%s_1x1' % name))
    # 3x3 reduce + 3x3
    c3x3r = ConvFactory(data=data, num_filter=num_3x3red, kernel=(1, 1), name=('%s_3x3' % name), suffix='_reduce')
    c3x3 = ConvFactory(data=c3x3r, num_filter=num_3x3, kernel=(3, 3), pad=(1, 1), name=('%s_3x3' % name))
    # double 3x3 reduce + double 3x3
    cd5x5r = ConvFactory(data=data, num_filter=num_d5x5red, kernel=(1, 1), name=('%s_5x5' % name), suffix='_reduce')
    cd5x5 = ConvFactory(data=cd5x5r, num_filter=num_d5x5, kernel=(5, 5), pad=(2, 2), name=('%s_5x5' % name))
    # pool + proj
    pooling = mx.symbol.Pooling(data=data, kernel=(3, 3), stride=(1, 1), pad=(1, 1), pool_type=pool, name=('%s_pool_%s_pool' % (pool, name)))
    cproj = ConvFactory(data=pooling, num_filter=proj, kernel=(1, 1), name=('%s_proj' %  name))
    # concat
    concat = mx.symbol.Concat(*[c1x1, c3x3, cd5x5, cproj], name='ch_concat_%s_chconcat' % name)
    return concat

def get_googlenet(num_classes = 1000):
    data = mx.sym.Variable("data")
    conv1 = ConvFactory(data, 64, kernel=(7, 7), stride=(2,2), pad=(3, 3), name="conv1")
    pool1 = mx.sym.Pooling(conv1, kernel=(3, 3), stride=(2, 2), pool_type="max")
    conv2 = ConvFactory(pool1, 64, kernel=(1, 1), stride=(1,1), name="conv2")
    conv3 = ConvFactory(conv2, 192, kernel=(3, 3), stride=(1, 1), pad=(1,1), name="conv3")
    pool3 = mx.sym.Pooling(conv3, kernel=(3, 3), stride=(2, 2), pool_type="max")

    in3a = InceptionFactory(pool3, 64, 96, 128, 16, 32, "max", 32, name="in3a")
    in3b = InceptionFactory(in3a, 128, 128, 192, 32, 96, "max", 64, name="in3b")
    pool4 = mx.sym.Pooling(in3b, kernel=(3, 3), stride=(2, 2), pool_type="max")
    in4a = InceptionFactory(pool4, 192, 96, 208, 16, 48, "max", 64, name="in4a")
    in4b = InceptionFactory(in4a, 160, 112, 224, 24, 64, "max", 64, name="in4b")
    in4c = InceptionFactory(in4b, 128, 128, 256, 24, 64, "max", 64, name="in4c")
    in4d = InceptionFactory(in4c, 112, 144, 288, 32, 64, "max", 64, name="in4d")
    in4e = InceptionFactory(in4d, 256, 160, 320, 32, 128, "max", 128, name="in4e")
    pool5 = mx.sym.Pooling(in4e, kernel=(3, 3), stride=(2, 2), pool_type="max")
    in5a = InceptionFactory(pool5, 256, 160, 320, 32, 128, "max", 128, name="in5a")
    in5b = InceptionFactory(in5a, 384, 192, 384, 48, 128, "max", 128, name="in5b")
    pool6 = mx.sym.Pooling(in5b, kernel=(7, 7), stride=(1,1), pool_type="avg")
    flatten = mx.sym.Flatten(data=pool6)
    fc1 = mx.sym.FullyConnected(data=flatten, num_hidden=num_classes)
    softmax = mx.symbol.SoftmaxOutput(data=fc1, name='softmax')
    return softmax

def get_dpn(num_classes=3):
    data = mx.symbol.Variable('data')
    #up = mx.symbol.UpSampling(data=data, scale=2, sample_type='bilinear', num_args=2)
    conv1 = mx.symbol.Convolution(data=data, kernel=(8, 8), stride=(4, 4), num_filter=32)
    batn1 = mx.symbol.BatchNorm(data=conv1, eps=0.001, momentum=0.9)
    #relu1 = mx.sym.LeakyReLU(data=batn1, upper_bound=1,  act_type='rrelu')
    relu1 = mx.symbol.Activation(data=batn1, act_type='relu')
    conv2 = mx.symbol.Convolution(data=relu1, kernel=(4,4), stride=(2, 2), num_filter=64)
    batn2 = mx.symbol.BatchNorm(data=conv2, eps=0.001, momentum=0.9)
    #relu2 = mx.symbol.LeakyReLU(data=batn2, upper_bound=1, act_type='rrelu')
    relu2 = mx.symbol.Activation(data=batn2, act_type='relu')
    conv3 = mx.symbol.Convolution(data=relu2, kernel=(3,3), stride=(1, 1), num_filter=64)
    batn3 = mx.symbol.BatchNorm(data=conv3, eps=0.001, momentum=0.9)
    #relu3 = mx.symbol.LeakyReLU(data=batn3, upper_bound=10, act_type='rrelu')
    relu3 = mx.symbol.Activation(data=batn3, act_type='relu')
    # conv4 = mx.symbol.Convolution(data=relu3, kernel=(3, 3), stride=(1, 1), num_filter=64)
    # batn4 = mx.symbol.BatchNorm(data=conv4, eps=0.001, momentum=0.9)
    #relu4 = mx.symbol.LeakyReLU(data=batn4, upper_bound=1, act_type='rrelu')
    # relu4 = mx.symbol.Activation(data=batn4, act_type='relu')
    flatten = mx.symbol.Flatten(data=relu3)
    fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=512)
    batn5 = mx.symbol.BatchNorm(data=fc1, eps=0.001, momentum=0.9)
    #act1 = mx.symbol.LeakyReLU(data=batn5, upper_bound=1, act_type='rrelu')
    act1 = mx.symbol.Activation(data=batn5, act_type='relu')
    drop1 = mx.symbol.Dropout(data=act1, p=0.5)
    fc2 = mx.symbol.FullyConnected(data=drop1, num_hidden = num_classes)

    softmax = mx.symbol.SoftmaxOutput(data=fc2, name='softmax')
    #softmax = mx.symbol.LinearRegressionOutput(data=fc2, label=data['label'], name='linear')
    return softmax

def get_depNet(num_classes):
    data = mx.symbol.Variable('data')
    conv1 = mx.symbol.Convolution(data=data, kernel=(11, 11), stride=(4,4), num_filter=96)
    batn1 = mx.symbol.BatchNorm(data=conv1, eps=0.001, momentum=0.9)
    relu1 = mx.symbol.Activation(data=batn1, act_type='relu')

    conv2 = mx.symbol.Convolution(data=relu1, kernel=(5, 5), stride=(1, 1), num_filter=256)
    batn2 = mx.symbol.BatchNorm(data=conv2, eps=0.001, momentum=0.9)
    relu2 = mx.symbol.Activation(data=batn2, act_type='relu')

    conv3 = mx.symbol.Convolution(data=relu2, kernel=(3, 3), stride=(1, 1), num_filter=384)
    batn3 = mx.symbol.BatchNorm(data=conv3, eps=0.001, momentum=0.9)
    relu3 = mx.symbol.Activation(data=batn3, act_type='relu')

    conv4 = mx.symbol.Convolution(data=relu3, kernel=(3, 3), stride=(1, 1), num_filter=384)
    batn4 = mx.symbol.BatchNorm(data=conv1, eps=0.001, momentum=0.9)
    relu4 = mx.symbol.Activation(data=batn1, act_type='relu')

    conv5 = mx.symbol.Convolution(data=data, kernel=(3, 3), stride=(1, 1), num_filter=256)
    batn5 = mx.symbol.BatchNorm(data=conv1, eps=0.001, momentum=0.9)
    relu5 = mx.symbol.Activation(data=batn1, act_type='relu')

    flatten = mx.symbol.Flatten(data=relu5)
    fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=4096)
    batn5 = mx.symbol.BatchNorm(data=fc1, eps=0.001, momentum=0.9)
    #act1 = mx.symbol.LeakyReLU(data=batn5, upper_bound=1, act_type='rrelu')
    act1 = mx.symbol.Activation(data=batn5, act_type='relu')
    drop1 = mx.symbol.Dropout(data=act1, p=0.5)
    fc2 = mx.symbol.FullyConnected(data=drop1, num_hidden = num_classes)

    softmax = mx.symbol.SoftmaxOutput(data=fc2, name='softmax')

    return softmax