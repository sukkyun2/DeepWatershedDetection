from __future__ import print_function
import numpy as np
import tensorflow as tf
from models.dwd_net import build_dwd_net
from main.dws_transform import perform_dws
from PIL import Image
#from main.config import cfg
from datasets import fcn_groundtruth
import scipy.special as sc


np.random.seed(314)
tf.compat.v1.set_random_seed(314)


class DWSDetector:
    def __init__(self, parsed, path, imdb):
        self.model_path = path
        self.config = parsed
        self.imdb = imdb

        self.saved_net = "backbone"

        self.tf_session = None
        self.sess = tf.compat.v1.Session()
        print('Loading model')

        if "DeepScores" not in self.model_path:
            self.input = tf.compat.v1.placeholder(tf.float32, shape=[None, None, None, 3])
        else:
            self.input = tf.compat.v1.placeholder(tf.float32, shape=[None, None, None, 1])

        print("Initializing Model:" + self.config.model)
        used_heads = set()
        self.used_heads_loss =[]
        for _, assign in enumerate(self.config.training_assignements):
            used_heads.add(assign["stamp_func"])
            self.used_heads_loss.append([assign["stamp_func"],assign["stamp_args"]["loss"]])
        self.used_heads = list(used_heads)
        self.used_heads.sort()
        # for now:
        #self.used_heads = ['stamp_class', 'stamp_bbox', 'stamp_energy']
        #self.used_heads = ['stamp_class', 'stamp_energy', 'stamp_bbox']
        #self.used_heads = ['stamp_bbox', 'stamp_class', 'stamp_energy']
        #self.used_heads = ['stamp_bbox', 'stamp_energy', 'stamp_class']
        #self.used_heads = ['stamp_energy', 'stamp_class', 'stamp_bbox']
        #self.used_heads = ['stamp_energy', 'stamp_bbox', 'stamp_class']

        self.network_heads, init_fn = build_dwd_net(
        self.input, model=parsed.model, num_classes=len(self.imdb._classes), pretrained_dir="", max_energy=parsed.max_energy,
             substract_mean=False, individual_upsamp = self.config.individual_upsamp, paired_mode=self.config.paired_data,
            used_heads=self.used_heads, sparse_heads=parsed.sparse_heads)

        self.saver = tf.compat.v1.train.Saver(max_to_keep=1000)
        # self.sess.run(tf.global_variables_initializer())
        # self.sess.run(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='deep_watershed/energy_logits_pair0_3'))
        print("Loading weights")
        self.saver.restore(self.sess, self.model_path + "/" + self.saved_net)
        self.tf_session = self.sess
        self.counter = 0

    def classify_img(self, img, cutoff=0, min_ccoponent_size=0):
        """
        This function classifies an image based on the results of the net, has been tested with different values of cutoff and min_component_size and 
        we have observed that it is very robust to perturbations of those values.
        inputs:
            img - the image, an ndarray
            cutoff - the cutoff we do for the energy
            min_component_size - the minimum size of the connected component
        returns:
            dws_list - the list of bounding boxes the dwdnet infers
        """
        if img.shape[0] > 1:
            img = np.expand_dims(img, 0)

        if img.shape[-1] > 3:
            img = np.expand_dims(img, -1)

        # y_mulity = int(np.ceil(img.shape[1] / 160.0))
        # x_mulity = int(np.ceil(img.shape[2] / 160.0))
        # if "realistic" not in self.model_path:
        #     canv = np.ones([y_mulity * 160, x_mulity * 160], dtype=np.uint8) * 255
        #     canv = np.expand_dims(np.expand_dims(canv, -1), 0)
        # else:
        #     canv = np.ones([y_mulity * 160, x_mulity * 160, 3], dtype=np.uint8) * 255
        #     canv = np.expand_dims(canv, 0)
        #
        # canv[0, 0:img.shape[1], 0:img.shape[2]] = img[0]

        #Image.fromarray(canv[0]).save(cfg.ROOT_DIR + "/output_images/" + "debug"+ 'input' + '.png')
        #import pickle
        #feed_train = pickle.load(open("feed_dict_train.pickle","rb"))

        # create fetch list
        print("create fetch list")
        fetch_list = []

        for pair in range(self.config.paired_data):
            for head in self.used_heads_loss:
                fetch_list.append(self.network_heads[pair][head[0]][head[1]][-1])

        #fetch_list = self.network_heads[0]["stamp_energy"]["softmax"]
        #img_test = np.stack((img[:,:,:,0], img[:,:,:,1], img[:,:,:,2]), -1)

        preds = self.tf_session.run([fetch_list], feed_dict={self.input: img})
        preds = preds[0]

        print(np.unique(np.argmax(preds[0], axis=-1)))

        #save_debug_panes(pred_energy, pred_class, pred_bbox,self.counter)
        #Image.fromarray(canv[0]).save(cfg.ROOT_DIR + "/output_images/" + "debug"+ 'input' + '.png')

        # Apply softmax where necessary
        i = 0
        for pair in range(self.config.paired_data):
            for head in self.used_heads_loss:
                if head[1] == "softmax":
                    preds[i] = np.argmax(preds[i], axis=-1)
                i += 1


        i = 0
        paired_detect = []
        for pair in range(self.config.paired_data):
            pred_dict = {}
            for head in self.used_heads_loss:
                if head[0] == "stamp_energy":
                    pred_dict["stamp_energy"] = preds[i]
                elif head[0] == "stamp_class" and False: # check config.class_estimation
                    pred_dict["stamp_class"] = preds[i]
                elif head[0] == "stamp_bbox" and self.config.bbox_estimation == "bbox_head":
                    pred_dict["stamp_bbox"] = preds[i]
                i += 1
            dws_list = perform_dws(pred_dict, cutoff, min_ccoponent_size,self.config)
            paired_detect.append(dws_list)

        self.counter += 1

        return paired_detect


    def get_images(self, data, predict_boxes=None, gt_boxes= None, text=False):
        """
        Utility function which draws the bounding boxes from both the inference and ground truth, useful to do manual inspection of results
        arguments:
            data - the image in ndarray format.
            boxes - boxes which we want to draw in the image.
            gt - set it to true if you want to also save the ground truth in addition to the results of the detector.
            text - set it to trye if you want to see also the classes of the classification/ground_truth in addition to bounding boxes.
        returns:
            im_input - the image with the results of the detection.
            im_gt - None if gt is set to False, the image with the drawn ground truth if set to True.
        """
        if data.shape[-1] == 1:
            data = data.squeeze(-1)

        if data.shape[-1] > 3:
            data = np.stack([data, data, data],-1)

        from PIL import ImageDraw

        im_pred = Image.fromarray(data.astype("uint8"))

        draw = ImageDraw.Draw(im_pred)
        # overlay GT boxes
        for row in predict_boxes:
            # cv2.rectangle(im_input, (row[0], row[1], row[2], row[3]), (0, 255, 0), 1)
            draw.rectangle(((row[0], row[1]), (row[2], row[3])), outline="red")

        if gt_boxes is not None:
            # overlay GT boxes
            for row in gt_boxes:
                # cv2.rectangle(im_input, (row[0], row[1], row[2], row[3]), (0, 255, 0), 1)
                draw.rectangle(((row[0], row[1]), (row[2], row[3])), outline="green")

        if text:
            draw = ImageDraw.Draw(im_pred)
            # overlay GT boxes
            for row in gt_boxes:
                draw.text((row[2], row[3]), str(row[4]), fill="red")

        return im_pred


    def show_images(self, data, gt_boxes=None, gt=False, text=False):
        """
        Utility functions which shows the results of get_images in the display window.
        arguments:
            data - the image in ndarray format.
            boxes - boxes which we want to draw in the image.
            gt - set it to true if you want to also save the ground truth in addition to the results of the detector.
            text - set it to trye if you want to see also the classes of the classification/ground_truth in addition to bounding boxes.
        returns:
            None
        """
        im_input, im_gt = self.get_images(data, gt_boxes, gt, text)
        im_input.show()
        im_gt.show()


    def save_images(self, config, data, predict_boxes, gt_boxes=None, text=False):
        """
        Utility function which saves the results of get_images.
        arguments:
            data - the image in ndarray format.
            boxes - boxes which we want to draw in the image.
            gt - set it to true if you want to also save the ground truth in addition to the results of the detector.
            text - set it to trye if you want to see also the classes of the classification/ground_truth in addition to bounding boxes.
            counter - each image is given a name starting from 0.png, 1.png, ..., num_images.png
        returns:
            None
        """

        if config.paired_data > 1:
            for i in range(config.paired_data):
                im_pred = self.get_images(data[:, :, i], predict_boxes[i], gt_boxes, text)

                im_pred.save(config.root_dir + "/output_images/inference/" + 'prediction' + str(self.counter) + "_" + str(i) + '.png')

        else:
            im_pred = self.get_images(data, predict_boxes, gt_boxes, text)

            im_pred.save(config.root_dir + "/output_images/inference/" + 'prediction' + str(self.counter) + '.png')

        return


    def save_debug_panes(self, pred_energy, pred_class, pred_bbox,preds, counter=0):
        """
        Utility function which saves the output panes from the Network directly as images
        arguments:
            pred_energy - energy map prediction
            pred_class - class map prediction
            pred_bbox - bounding box map prediction
            counter - each image is given a name starting from 0.png, 1.png, ..., num_images.png
        returns:
            None
        """
        panes_list = [[pred_energy, "stamp_energy", "softmax"],
        [pred_class, "stamp_class", "softmax"],
        [pred_bbox, "stamp_bbox", "softmax"]]

        for pane in panes_list:
            pan_vis = fcn_groundtruth.color_map(pane[0], {"stamp_func": [pane[1]], "stamp_args": {"loss": pane[2]}})
            im = Image.fromarray(pan_vis[0])
            im.save(preds.root_dir + "/output_images/inference/" + pane[1] + str(counter) + '.png')


        return

