import os

file = os.listdir('C:\\Users\shiyx\Desktop\drive-download-20190807T220028Z-001 (1)')

for i in range(len(file)):
    path = 'C:\\Users\shiyx\Desktop\drive-download-20190807T220028Z-001 (1)\\'+file[i]
    os.rename(path, 'C:\\Users\shiyx\Desktop\drive-download-20190807T220028Z-001 (1)\\%02d.jpg'%i)
