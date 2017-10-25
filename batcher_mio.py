"""
taken and modified from https://github.com/pranv/ARC
"""
import csv
import os
from collections import defaultdict
from math import exp, log
from random import randint

import cv2
import numpy as np
import torch
from numpy.random import choice, shuffle
from torch.autograd import Variable

use_cuda = False
pjoin = os.path.join


class MioLocalization(object):
    def __init__(self, path=os.path.join('data', 'omniglot.npy'), batch_size=128, image_size=128):
        """
        batch_size: the output is (2 * batch size, 1, image_size, image_size)
                    X[i] & X[i + batch_size] are the pair
        image_size: size of the image
        data_split: in number of alphabets, e.g. [30, 10] means out of 50 Omniglot characters,
                    30 is for training, 10 for validation and the remaining(10) for testing
        within_alphabet: for verfication task, when 2 characters are sampled to form a pair,
                        this flag specifies if should they be from the same alphabet/language
        ---------------------
        Data Augmentation Parameters:
            flip: here flipping both the images in a pair
            scale: x would scale image by + or - x%
            rotation_deg
            shear_deg
            translation_px: in both x and y directions
        """
        """
        MioTCDClassification handles the IO from the MioTCD-Classification challenge
        :param filepath: File path where to find the dataset : String
        :param size: output shape (in pixel) the output will be (size,size)  : Int
        """
        self.filepath = path
        self.nb_class = 11
        self.shape = image_size
        self.output_shape = [image_size, image_size]
        self.batch_size = batch_size
        self.classes = ["articulated_truck"
            , "bicycle"
            , "bus"
            , "car"
            , "motorcycle"
            , "motorized_vehicle"
            , "non-motorized_vehicle"
            , "pedestrian"
            , "pickup_truck"
            , "single_unit_truck"
            , "work_van"]
        self.generate_dataset()

    def __get_bounding_box(self, setting):
        path = pjoin(self.filepath, "gt_{}.csv".format(setting))
        boxes = defaultdict(list)
        with open(path, "r") as f:
            reader = csv.reader(f)
            for row in reader:
                boxes[row[0]].append([row[1]] + [float(k) for k in row[2:]])
        return boxes

    def generate_dataset(self):
        """
        Get the paths from the dataset
        :return: None
        """
        datas = {}
        datas["test"] = [pjoin(self.filepath, "test", i) for i in
                         os.listdir(pjoin(self.filepath, "test"))], self.__get_bounding_box("test")
        datas["train"] = [pjoin(self.filepath, "train", i) for i in
                          os.listdir(pjoin(self.filepath, "train"))], self.__get_bounding_box("train")
        data_test = [(i, datas["test"][1][i.split("/")[-1][:-4]]) for i in datas["test"][0]]
        data_train = [(i, datas["train"][1][i.split("/")[-1][:-4]]) for i in datas["train"][0]]
        shuffle(data_train)
        shuffle(data_test)
        self.X_train, self.Y_train = zip(*data_train)
        self.X_test, self.Y_test = zip(*data_test)
        self.X_train, self.X_valid = np.split(self.X_train, [int(0.8 * len(self.X_train))])
        self.Y_train, self.Y_valid = np.split(self.Y_train, [int(0.8 * len(self.Y_train))])

    def fetch_batch(self, part):
        parts = {'train': (self.X_train, self.Y_train),
                 'val': (self.X_valid, self.Y_valid),
                 'test': (self.X_test, self.Y_test)}
        X_set, Y_set = parts[part]
        X, y = zip(*[self.get_sample(X_set, Y_set) for _ in range(self.batch_size)])
        X, y = np.array(X), np.array(y)
        X = Variable(torch.from_numpy(X))
        y = Variable(torch.from_numpy(y))
        if use_cuda:
            X, y = X.cuda(), y.cuda()
        return X, y

    def rescale(self, pts, f):
        return [int(x // f) for x in pts]

    def get_sample(self, X_set, Y_set):
        good = False
        while not good:
            id = randint(0, len(X_set) - 1)
            X, y = X_set[id], Y_set[id]
            if y:
                good = True
                y = y[randint(0,len(y) - 1)]

        im = cv2.imread(X)
        im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)/ 255.
        H, W = im.shape[:2]
        x1, y1, x2, y2 = y[1:]

        wi = int(max(0, x1 - randint(5, 50)))
        wj = int(min(W, x2 + randint(5, 50)))
        diffx = wi

        hi = int(max(0, y1 - randint(5, 50)))
        hj = int(min(H, y2 + randint(5, 50)))

        hi,hj = sorted([hi,hj])
        wi,wj = sorted([wi,wj])

        diffy = hi
        crop = im[hi:hj, wi:wj]
        newH, newW = crop.shape[:2]
        x1, x2 = x1 - diffx, x2 - diffx
        y1, y2 = y1 - diffy, y2 - diffy

        x1, x2 = self.rescale([x1, x2], newW / self.shape)
        y1, y2 = self.rescale([y1, y2], newH / self.shape)
        crop = cv2.resize(crop, (self.shape, self.shape))

        cx = ((x2 + x1) / 2) / self.shape
        cy = ((y2 + y1) / 2) / self.shape
        try:
            if x2 == x1:
                w = 0
            else:
                w = log((x2 - x1))
            if y2 == y1:
                h = 0
            else:
                h = log((y2 - y1))
        except Exception:
            print(x1,x2,y1,y2,wi,wj,hi,hj, newH,newW)
            raise
        y = np.array([cx, cy, w, h])
        X = crop
        return X, y

    def get_back(self, param):
        acc = []
        for y in param.shape[0]:
            cx, cy, w, h = y
            cx, cy = int(cx * 128), int(cy * 128)
            w = int(exp(w) / 2)
            h = int(exp(h) / 2)
            acc.append((cx,cy,w,h))
        return acc

    def draw(self,im,cx,cy,w,h):
        X = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(X, (cx - w, cy - h), (cx + w, cy + h), (255, 0, 0))
        cv2.circle(X, (cx, cy), 1, (0, 255, 0), 3)
        return X


if __name__ == '__main__':
    batcher = MioLocalization(path='/media/braf3002/hdd2/Downloads/MIO-TCD-Localization',batch_size=1)
    while True:
        X,y = batcher.fetch_batch('train')
        X,y = batcher.fetch_batch('val')
        X,y = batcher.fetch_batch('test')
        continue
        X = X.data.numpy()[0]
        y = y.data.numpy()[0]

        X = (X * 255).astype(np.uint8)
        cx,cy,w,h = y
        cx,cy = int(cx*128), int(cy*128)
        w = int(exp(w)/2)
        h = int(exp(h)/2)
        X = cv2.cvtColor(X,cv2.COLOR_GRAY2BGR)
        cv2.rectangle(X,(cx-w,cy-h),(cx+w,cy+h),(255,0,0))
        cv2.circle(X, (cx, cy), 1, (0, 255, 0), 3)

        cv2.imshow('x',X)
        cv2.waitKey(1)

