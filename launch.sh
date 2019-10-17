#!\bin\bash
python train.py --layers 28 --widen-factor 10 --batchnorm True --fixup False --name bn1block_2 -d 0:2 --epochs 300
python train.py --layers 28 --widen-factor 10 --batchnorm False --fixup True --name fixup1block_2 -d 0:2 --epochs 300

