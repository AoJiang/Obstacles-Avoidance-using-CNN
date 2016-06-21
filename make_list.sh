#python /home/lcc/mxnet/tools/make_list.py /home/lcc/ardrone_python/data/rgb /home/lcc/ardrone_python/data/rgb/train --recursive=True --exts=.jpg #--chunks=2 #--train_ratio=0.5
#python /home/lcc/mxnet/tools/im2rec.py data/rgb/train_0 data/rgb  --encoding=.jpg --quality=100 --resize=84 #--train_ratio=0.5 #--color=0
#python /home/lcc/mxnet/tools/im2rec.py data/rgb/train_1 data/rgb  --encoding=.jpg --quality=100 --resize=84 #--train_ratio=0.5 #--color=0
#python /home/lcc/mxnet/tools/im2rec.py data/rgb/train data/rgb  --encoding=.jpg --quality=100 --resize=84 #--train_ratio=0.5 #--color=0

#python /home/lcc/mxnet/tools/make_list.py /home/lcc/ardrone_python/data/hsv /home/lcc/ardrone_python/data/hsv/train --recursive=True --exts=.jpg #--chunks=2 #--train_ratio=0.5
#python /home/lcc/mxnet/tools/im2rec.py data/hsv/train_0 data/hsv  --encoding=.jpg --quality=100 --resize=84 #--train_ratio=0.5 --color=0
#python /home/lcc/mxnet/tools/im2rec.py data/hsv/train_1 data/hsv  --encoding=.jpg --quality=100 --resize=84 #--train_ratio=0.5 --color=0
#python /home/lcc/mxnet/tools/im2rec.py data/hsv/train data/hsv  --encoding=.jpg --quality=100 --resize=84 --train_ratio=0.5 #--color=0

#python /home/lcc/mxnet/tools/make_list.py /home/lcc/ardrone_python/data/hsv_train /home/lcc/ardrone_python/data/hsv_train/train --recursive=True --exts=.jpg #--chunks=2 #--train_ratio=0.5
#python /home/lcc/mxnet/tools/im2rec.py data/hsv_train/train data/hsv_train  --encoding=.jpg --quality=100 --resize=84 #--train_ratio=0.5 #--color=0

#python /home/lcc/mxnet/tools/make_list.py rgb_test /home/lcc/ardrone_python/data/rgb_test/train --recursive=True --exts=.jpg #--chunks=2 #--train_ratio=0.5
#python /home/lcc/mxnet/tools/im2rec.py rgb_test data/rgb_test  --encoding=.jpg --quality=100 --resize=84 --recursive=True --list=True #--train_ratio=0.5 #--color=0
#python /home/lcc/mxnet/tools/im2rec.py rgb_test data/rgb_test  --encoding=.jpg --quality=100 --resize=84 --recursive=True #--list=True #--train_ratio=0.5 #--color=0

python /home/lcc/mxnet/tools/im2rec.py rgb_tr data/rgb_tr/  --encoding=.jpg --quality=100 --resize=180 --recursive=True --list=True #--chunk=2 #--train_ratio=0.5 #--color=0
#python /home/lcc/mxnet/tools/im2rec.py rgb_tr_0_train data/rgb/  --encoding=.jpg --quality=100 --resize=84 --recursive=True #--list=False #--train_ratio=0.5 #--color=0
#python /home/lcc/mxnet/tools/im2rec.py rgb_tr_1_train data/rgb/  --encoding=.jpg --quality=100 --resize=84 --recursive=True #--list=False #--train_ratio=0.5 #--color=0
python /home/lcc/mxnet/tools/im2rec.py rgb_tr_train data/rgb_tr/  --encoding=.jpg --quality=100 --resize=180 --recursive=True

python /home/lcc/mxnet/tools/im2rec.py rgb_te data/rgb_te/  --encoding=.jpg --quality=100 --resize=180 --recursive=True --list=True #--chunk=2 #--train_ratio=0.5 #--color=0
#python /home/lcc/mxnet/tools/im2rec.py rgb_tr_0_train data/rgb/  --encoding=.jpg --quality=100 --resize=84 --recursive=True #--list=False #--train_ratio=0.5 #--color=0
#python /home/lcc/mxnet/tools/im2rec.py rgb_tr_1_train data/rgb/  --encoding=.jpg --quality=100 --resize=84 --recursive=True #--list=False #--train_ratio=0.5 #--color=0
python /home/lcc/mxnet/tools/im2rec.py rgb_te_train data/rgb_te/  --encoding=.jpg --quality=100 --resize=180 --recursive=True

#python /home/lcc/mxnet/tools/make_list.py /home/lcc/ardrone_python/data/test /home/lcc/ardrone_python/data/test/train --recursive=True --exts=.jpg #--chunks=2 #--train_ratio=0.5
#python /home/lcc/mxnet/tools/im2rec.py data/hsv/train_0 data/test  --encoding=.jpg --quality=100 --resize=84 #--train_ratio=0.5 --color=0
#python /home/lcc/mxnet/tools/im2rec.py data/hsv/train_1 data/test  --encoding=.jpg --quality=100 --resize=84 #--train_ratio=0.5 --color=0
#python /home/lcc/mxnet/tools/im2rec.py data/test/train data/test  --encoding=.jpg --quality=100 --resize=84 #--train_ratio=0.5 #--color=0

#python /home/lcc/mxnet/tools/make_list.py /home/lcc/ardrone_python/data/pca /home/lcc/ardrone_python/data/pca/train --recursive=True --exts=.jpg
#python /home/lcc/mxnet/tools/im2rec.py data/pca/train data/pca  --encoding=.jpg --quality=100 --resize=160 --train_ratio=0.8

#python /home/lcc/mxnet/tools/make_list.py /home/lcc/ardrone_python/data/edge /home/lcc/ardrone_python/data/edge/train --recursive=True --exts=.jpg
#python /home/lcc/mxnet/tools/im2rec.py data/edge/train data/edge  --encoding=.jpg --quality=100 --resize=160 --color=0 --train_ratio=0.8

#python /home/lcc/mxnet/tools/make_list.py /home/lcc/ardrone_python/data/gray /home/lcc/ardrone_python/data/gray/train --recursive=True --exts=.jpg
#python /home/lcc/mxnet/tools/im2rec.py data/edge/train data/gray  --encoding=.jpg --quality=100 --resize=160 --color=0 --train_ratio=0.8

#python /home/lcc/mxnet/tools/make_list.py /home/lcc/ardrone_python/data/randcrop /home/lcc/ardrone_python/data/randcrop/train --recursive=True --exts=.jpg
#python /home/lcc/mxnet/tools/im2rec.py data/randcrop/train data/randcrop  --encoding=.jpg --quality=100 --resize=160 --color=0 --train_ratio=0.8

#python /home/lcc/mxnet/tools/make_list.py /home/lcc/ardrone_python/data/pca_all /home/lcc/ardrone_python/data/pca_all/train --recursive=True --exts=.jpg #--chunks=2 #--train_ratio=0.5
#python /home/lcc/mxnet/tools/im2rec.py data/rgb/train_0 data/rgb  --encoding=.jpg --quality=100 --resize=84 #--train_ratio=0.5 #--color=0
#python /home/lcc/mxnet/tools/im2rec.py data/rgb/train_1 data/rgb  --encoding=.jpg --quality=100 --resize=84 #--train_ratio=0.5 #--color=0
#python /home/lcc/mxnet/tools/im2rec.py pca_all data/pca_all/  --num_thread=8 --encoding=.jpg --quality=100 --resize=84 --recursive=True #--list=True #--train_ratio=0.8 --test_ratio=0.2 #--color=0