from pylab import *

label = np.load('00label.npy').astype('float32')
r = np.load('UNET02-3800.npy').astype('float32')
print(label.shape)
print(r.shape)
###########################################
#     input    crack      Nc
#crack         TP         WN
#   nc         WP         TN
###########################################

thre = 0.4
TP = len(where(r[where(label==1)]>=thre)[0])
WP = len(where(r[where(label==1)]< thre)[0])
WN = len(where(r[where(label==0)]>=thre)[0])
TN = len(where(r[where(label==0)]< thre)[0])

print(TP, WN)
print(WP, TN)