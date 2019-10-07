from pylab import *
from PIL import Image

def pick():
    pic = Image.open('vug.jpg')
    # pic = pic.resize([2736//2, 3648//2])
    pic_data = np.array(pic)

    ph,pw,pc = pic_data.shape
    figure(1)
    imshow(pic_data)
    h,w,c,num = [256,256,3,100]

    samples = np.zeros([h,w,c,num])
    index = []
    for i in range(num):
        index_h = int(np.random.random()*(ph-h-2))
        index_w = int(np.random.random()*(pw-w-2))
        should_continue = False
        if len(index) !=0:
            for k in index:
                kh, kw, _ = k
                if (kh-h < index_h < kh+h) and ( kw-w < index_w < kw+w):
                    should_continue = True
                    break

        if should_continue:
            continue
        index.append([index_h, index_w, i])
        samples[:,:,:,i] = pic_data[index_h:(index_h+h), index_w:(index_w+w), :]
        plot([index_w, index_w, index_w+w, index_w+w, index_w], [index_h, index_h+h, index_h+h, index_h, index_h], 'b')
        text(x=index_w, y=index_h, s = len(index), color='r')

    result = np.zeros([h, w, c, len(index)])
    for i in range(len(index)):
        result[:,:,:, i] = samples[:,:,:,index[i][2]]

    np.save('samples/all.npy', np.array({'index':index, 'data':result}))

    for i in range(len(index)):
        c = Image.fromarray(result[:,:,:,i].astype('int8'), mode='RGB')
        c.save('samples/%s.jpg'%i)
    show()

def replot():
    pic = Image.open('vug.jpg')
    pic_data = np.array(pic)
    ph, pw, pc = pic_data.shape
    figure(1)
    imshow(pic_data)
    h, w, c, num = [256, 256, 3, 100]

    result = np.load('samples/all.npy').tolist()
    samples = result['data']
    index = result['index'][0:21]
    for i in range(len(index)):
        index_h, index_w, _ = index[i]
        samples[:, :, :, i] = pic_data[index_h:(index_h + h), index_w:(index_w + w), :]
        plot([index_w, index_w, index_w + w, index_w + w, index_w],
             [index_h, index_h + h, index_h + h, index_h, index_h], 'b')
        text(x=index_w, y=index_h, s=i, color='r')
    show()

    i = 0
    while 1:
        a = input('which one do you like to show:')
        if a=='':
            a=i + 1
        i = int(a)

        imshow(pic_data)
        index_h, index_w, _ = index[i]
        samples[:, :, :, i] = pic_data[index_h:(index_h + h), index_w:(index_w + w), :]
        plot([index_w, index_w, index_w + w, index_w + w, index_w],
             [index_h, index_h + h, index_h + h, index_h, index_h], 'b')
        text(x=index_w, y=index_h, s=i, color='r')

        show()

def get_label():

    label = np.zeros([21, 256,256,1]).astype('float32')
    for i in range(21):
        label_i = np.array(Image.open('samples/%s.png' % i)).astype('float32')
        a = np.reshape(label_i[:,:,0], [256,256,1])
        b = np.zeros_like(a).astype('float32')
        b[np.where(a>250)] = 1
        label[i,:,:,:] = b
        figure()
        subplot(121)
        imshow(a[:,:,0])
        subplot(122)
        imshow(b[:,:,0])
        show()
    np.save('samples/all_label.npy',label)

class DATA:
    def __init__(self, num=21):
        self.image = np.load('samples/all.npy').tolist()['data']
        self.image = np.transpose(self.image[:,:,:,0:num], [3,0,1,2])

        self.label = np.load('samples/all_label.npy')
        print(self.image.shape, self.label.shape)
        self.shape = self.image.shape

    def trainBatch(self, num, chose_sample=False):
        data = np.zeros([num, self.shape[1], self.shape[2], self.shape[3]])
        label = np.zeros([num, self.shape[1], self.shape[2], 1])
        i = 0
        while i<num:
            if chose_sample:
                index=11
            else:
                index = int(np.random.random()*self.shape[0])
            data[i,:,:,:] = self.image[index, :,:,:]
            label[i,:,:,:] = self.label[index, :,:,:]
            i=i+1
        return [data, label]


class DATA_BIG:
    # from samples_big
    def __init__(self, num=9, from_file=True, downsample=2, file = None):

        self.image = []
        self.label = []
        self.shape = [0, 0, 0, 0]
        self.downsample = downsample
        self.num = num
        self.ori_images = []
        self.ori_labels = []
        if not from_file:
            self.from_file()
        else:
            if file is None:
                all = np.load('samples_big/all.npy').tolist()
            else:
                all = np.load(file).tolist()
            self.image = all['data'].astype('float32')
            self.label = all['label'].astype('float32')
            self.shape = self.image.shape

    def from_file(self):

        ori_images, ori_labels = self.get_ori_data()
        self.form_images(ori_images)
        self.form_labels(ori_labels)
        assert len(self.ori_images)==len(self.ori_labels)

    def get_ori_data(self):
        ori_images = []
        ori_labels = []
        num = self.num
        downsample = self.downsample
        for i in range(num):
            image = Image.open('samples_big/%02d.jpg' % (i+1))
            label = Image.open('samples_big/Inked%02d_LI.jpg' % (i + 1))
            h, w = image.size
            if downsample:
                image = image.resize([int(h / downsample), int(w / downsample)])
                label = label.resize([int(h / downsample), int(w / downsample)])
            ori_images.append(image)
            ori_labels.append(label)
        return [ori_images, ori_labels]

    def form_labels(self, ori_labels):
        self.ori_labels = []
        for i in range(self.num):
            label = np.array(ori_labels[i]).astype('float32')
            r = label[:, :, 0]
            g = label[:, :, 1]
            b = label[:, :, 2]
            r = (np.sign(np.sign(r - 220.8) + np.sign(245.1 - r) - 1.1) + 1) / 2
            g = (np.sign(40.1 - g) + 1) / 2
            b = (np.sign(40.1 - b) + 1) / 2
            label = r * g * b
            print(i)
            print(label.shape)
            label.astype('float32')
            # print(label)
            self.ori_labels.append(label)

    def form_images(self, ori_images):
        self.ori_images = []
        for i in range(self.num):
            self.ori_images.append(np.array(ori_images[i]).astype('float32'))

    def dataset(self, num):
        h, w, c, num = [256, 256, 3, num]
        results = np.zeros([num, h,w,c])
        labels = np.zeros([num, h,w, 1])
        i = 0
        index = []
        while i<num:
            pic = int(np.random.random()*len(self.ori_images))
            data = self.ori_images[pic]
            label = self.ori_labels[pic]
            ph, pw, pc = data.shape
            index_h = int(np.random.random() * (ph - h - 2))
            index_w = int(np.random.random() * (pw - w - 2))
            index.append([pic, index_h, index_w, i])
            # print(ph, pw, pc, index_h, index_w, label.shape)
            sample_d = data[index_h:(index_h + h), index_w:(index_w + w), :]
            sample_l = label[index_h:(index_h + h), index_w:(index_w + w)]
            if sample_l.sum()<100:
                continue
            deg = int(np.random.random()*4)
            sample_d = np.rot90(sample_d, deg)
            sample_l = np.rot90(sample_l, deg)


            results[i, :, :, :] = sample_d
            labels[i, :, :,0] = sample_l
            i = i+1
            print('dataset num: \r %05d/%d'%(i,num), end='')

        # np.save('samples_big/all.npy', np.array({'data': results, 'label': labels}))
        self.image = results
        self.label = labels
        self.shape = self.image.shape

    def trainBatch(self, num, chose_sample=False):
        data = np.zeros([num, self.shape[1], self.shape[2], self.shape[3]])
        label = np.zeros([num, self.shape[1], self.shape[2], 1])
        i = 0
        while i < num:
            if chose_sample:
                index = 0+i
            else:
                index = int(np.random.random() * self.shape[0])
            ima = self.image[index, :, :, :]
            data[i, :, :, :] = ima
            label[i, :, :, :] = self.label[index, :, :, :]
            i = i + 1
        data = data/128-1

        return [data, label]


class DATA_gee(DATA_BIG):
    # Data with three chn, gray image, edge deteciton sobel, edge detection canny
    def get_ori_data(self):
        ori_images = []
        ori_labels = []
        num = self.num
        downsample = self.downsample
        for i in range(num):
            label = Image.open('samples_big/InkedInked%02d_LI.jpg' % (i + 1))
            h, w = label.size
            if downsample:
                assert downsample == 2
                label = label.resize([int(h / downsample), int(w / downsample)])
            ori_labels.append(label)

            image = np.load('samples_big/%02d_gee.npy' % (i + 1))
            # 这是因为image的h,w维度与label array 化之后刚好差了1
            image = np.transpose(image.reshape([image.shape[0], 3, image.shape[1]//3]), [0,2,1])
            image[:,:,1] = image[:,:,1]*128+128
            image[:, :, 2] = image[:, :, 2] * 128+128
            ori_images.append(image)
            print('image shape:', image.shape[0:2], 'label_shape:', np.array(label).shape,'image size', label.size)
            assert image.shape[0] == np.array(label).shape[0]
            assert image.shape[1] == np.array(label).shape[1]
        return [ori_images, ori_labels]

# a = DATA_BIG(8, from_file=False, downsample=2)
# a.dataset(2400)
# a.trainBatch(1)