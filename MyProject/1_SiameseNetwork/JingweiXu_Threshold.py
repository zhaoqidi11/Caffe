class JingweiXu():

    def get_vector(self):
        sys.path.insert(0, '/data/caffe/python')
        import caffe
        import cv2
        import math
        import csv

        caffe.set_mode_gpu()
        caffe.set_device(0)
        # load model(.prototxt) and weight (.caffemodel)
        os.chdir('/data/Meisa/ResNet/ResNet-50')
        ResNet_Weight = './resnet50_cvgj_iter_320000.caffemodel'  # pretrained on il 2012 and place 205

        ResNet_Def = 'deploynew_nosoftmax.prototxt'
        net = caffe.Net(ResNet_Def,
                        ResNet_Weight,
                        caffe.TEST)

        # load video
        i_Video = cv2.VideoCapture('/data/RAIDataset/Video/2.mp4')

        # get width of this video
        wid = int(i_Video.get(3))

        # get height of this video
        hei = int(i_Video.get(4))

        # get the number of frames of this video
        framenum = int(i_Video.get(7))

        if i_Video.isOpened():
            success = True
        else:
            success = False
            print('Can\' open this video!')



        # Frame stores all frames of this video
        Frame = []
        while success:
            success, frame = i_Video.read()
            Frame.append(frame)

        # # Convert .binaryproto to .npy file
        # blob = caffe.proto.caffe_pb2.BlobProto()
        # data = open('hybridCNN_mean.binaryproto', 'rb').read()
        # blob.ParseFromString(data)
        # array = np.array(caffe.io.blobproto_to_array(blob))
        # mean_npy = array[0]
        # np.save('place205.npy', mean_npy)
        # mu = np.load('place205.npy')
        #
        # mu = mu.mean(1).mean(1)

        transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

        transformer.set_transpose('data', (2, 0, 1))
        # transformer.set_mean('data', mu)
        transformer.set_raw_scale('data', 255)
        transformer.set_channel_swap('data', (2, 1, 0))

        net.blobs['data'].reshape(1,
                                  3,
                                  224, 224)

        FrameV = []

        for i in range(len(Frame)):
            if Frame[i] is None:
                print i
                continue
            transformed_image = transformer.preprocess('data', Frame[i])
            net.blobs['data'].data[...] = transformed_image
            output = net.forward()
            FrameV.append(output['score'][0].tolist())

        return FrameV




    def RGBToGray(self, RGBImage):

        import numpy as np
        return np.dot(RGBImage[..., :3], [0.299, 0.587, 0.114])






    def CutVideoIntoSegments(self):
        import math
        import cv2
        import numpy as np

        # It save the pixel intensity between 20n and 20(n+1)
        d = []

        i_Video = cv2.VideoCapture('/data/RAIDataset/Video/2.mp4')
        if i_Video.isOpened():
            success = True
        else:
            success = False
            print('Can\' open this video!')

        # It save the number of frames in this video
        FrameNumber = int(i_Video.get(7))

        # The number of segments
        Count = int(math.ceil(float(FrameNumber) / 21.0))
        for i in range(Count):

            i_Video.set(1, 20*i)
            ret1, frame_20i = i_Video.read()

            if(20*(i+1)) >= FrameNumber:
                break

            i_Video.set(1, 20*(i+1))
            ret2, frame_20i1 = i_Video.read()

            d.append(np.sum(np.abs(self.RGBToGray(frame_20i) - self.RGBToGray(frame_20i1))))


        # The number of group
        GroupNumber = int(math.ceil(float(FrameNumber) / 10.0))

        MIUG = np.mean(d)
        for i in range(GroupNumber):
            MIUL = np.mean(d[10*i:10*i+10])
            SigmaL = np.std(d[10*i:10*i+10])

            TL = MIUL + 
        print 'a'



    # Calculate the cosin distance between vector1 and vector2
    def cosin_distance(self, vector1, vector2):
        dot_product = 0.0
        normA = 0.0
        normB = 0.0
        for a, b in zip(vector1, vector2):
            dot_product += a * b
            normA += a ** 2
            normB += b ** 2
        if normA == 0.0 or normB == 0.0:
            return None
        else:
            return dot_product / ((normA * normB) ** 0.5)

    # Calculate the D1
    def getD1(self, Segment):
        return self.cosin_distance(Segment[0], Segment[-1])


if __name__ == '__main__':
    test1 = JingweiXu()
    test1.CutVideoIntoSegments()
