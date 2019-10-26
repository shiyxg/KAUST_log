from pylab import *
from PIL import Image
ori_label = np.load('UNET02-3800.npy')
a = np.zeros_like(ori_label)
a[:,:] = ori_label
group=[]
threshold = 0.2
a[where(a<threshold)]=0
a[where(a>=threshold)]=1


def calculate(i, j):
    x = []
    y = []
    if i<0 or i>=a.shape[0] or j<0 or j>=a.shape[1]:
        return x,y

    if a[i,j] == 1:
        a[i,j]=0.5
        x.append(i)
        y.append(j)
        print('\r', i, j, end='', flush=True)
        x_left,y_left = calculate(i-1,j)
        x_right,y_right = calculate(i+1,j)
        x_up,y_up = calculate(i,j-1)
        x_down,y_down = calculate(i,j+1)

        x = x+x_left+x_right+x_up+x_down
        y = y+y_left+y_right+y_up+y_down

        return x,y
    else:
        return x,y


def loop(i,j):
    x,y = [],[]
    # in the array a， 1 means this is a crack, and no trace have been there, 2 means it has been or is traced, 0 means it is not traced
    if a[i,j] == 1:
        trace = [[i,j, 0,0,0,0]]
        x.append(i)
        y.append(j)
        while 1:
            if len(trace)==0:
                # when there is no point to tracing
                break
            i,j, u, r, d, l = trace[-1] # use the end of trace line as the staring point to continue tracing
            print('\r', 'traceline:%05d'%len(trace), i,j,u,r,d,l, end='')
            if u == 0:
                # when the up point has not been traced
                a[i,j] = 2
                new_i, new_j = i, j-1
                u=1
            elif r == 0:
                # when the right point has not been traced
                a[i,j]=2
                new_i, new_j = i+1, j
                r=1
            elif d == 0:
                # when the down point has not been traced
                a[i, j] = 2
                new_i, new_j = i, j +1
                d=1
            elif l == 0:
                # when the left point has not been traced
                a[i, j] = 2
                new_i, new_j = i-1, j
                l=1
            else:
                # when all directions have been traced
                trace.pop()
                continue
            trace[-1] = [i,j,u,r,d,l]
            if new_i < 0 or new_i >= a.shape[0] or new_j < 0 or new_j >= a.shape[1]:
                # when new one out of range, stop
                continue
            if a[new_i, new_j]==2 or a[new_i, new_j]==0:
                # when new one is not crack or being traced, stop
                continue
            x.append(new_i)
            y.append(new_j)
            # add new point as the starting point to trace.
            trace.append([new_i,new_j, 0,0,0,0])

    return x,y


for i in range(a.shape[0]):
    for j in range(a.shape[1]):
        if a[i,j]==1:
            ########
            #(i,j),   (i+1,j)
            #(i,j+1), (i+1,j+1)
            ########
            # the depth of stack has limitation ,so to big images, it will break down
            # x,y = calculate(i,j)
            x,y = loop(i,j)
            assert len(x)==len(y)
            if len(x)>700:
                group.append([x, y])
                print('')
                print('in new points group:',len(group), ',add points:',len(x))

c = np.zeros_like(a)
for i in range(len(group)):
    c[group[i][0], group[i][1]]=1
c = c*ori_label
save('UNET02-3800-big.npy', c)
c[where(c==0)]=None
c[0,0]=0
figure()
imshow(np.load('00gray.npy'), cmap='gray')
imshow(c, cmap='bwr')
figure()
imshow(np.load('00gray.npy'), cmap='gray')
ori_label[where(ori_label<0.1)]=None
ori_label[0,0]=0
imshow(ori_label, cmap='bwr')
show()