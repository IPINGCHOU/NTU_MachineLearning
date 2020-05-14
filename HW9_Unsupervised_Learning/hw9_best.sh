# wget -P $2 https://github.com/IPINGCHOU/NTU_MachineLearning/releases/download/hw9/baseline.pth
# wget -P $2 https://github.com/IPINGCHOU/NTU_MachineLearning/releases/download/hw9/improved_2.pth
# wget -P $2 https://github.com/IPINGCHOU/NTU_MachineLearning/releases/download/hw9/unet.pth

python3 hw9_best.py $1 $2 $3
