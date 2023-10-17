import os
import tensorflow as tf
import numpy as np
import sys

sys.path.insert(0, os.path.dirname(__file__)[:-4])
#from main.config import cfg

from models.dwd_net import build_dwd_net

from datasets.factory import get_imdb
from utils.safe_softmax_wrapper import safe_softmax_cross_entropy_with_logits
import roi_data_layer.roidb as rdl_roidb
from roi_data_layer.layer import RoIDataLayer
import tf_slim as slim
#from utils.prefetch_wrapper import PrefetchWrapper
from utils.prefetch_wrapper_cache import PrefetchWrapperCache as PrefetchWrapper


from tensorflow.python.ops import array_ops
import pickle
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import json
import hashlib
import copy
import datetime
import shutil



from datasets.fcn_groundtruth import stamp_class, stamp_directions, stamp_energy, stamp_bbox, stamp_semseg, \
    try_all_assign, get_gt_visuals, get_map_visuals, overlayed_image

nr_classes = None
store_dict = True


def main(parsed):
    args = parsed[0]
    print(args)

    iteration = 1
    np.random.seed(args.random_seed)

    # load database
    imdb_train, roidb_train, imdb_val, roidb_val, data_layer_train, data_layer_val = load_database(args)

    imdb = [imdb_train, imdb_val]
    data_layer = [data_layer_train, data_layer_val]


    #global nr_classes
    nr_classes = len(imdb_train._classes)
    args.nr_classes.append(nr_classes)
    args.semseg_ind = imdb_train.semseg_index()

    fingerprint = build_config_fingerprint(args)

    args_txt = copy.deepcopy(args)
    # replaces keywords with function handles in training assignements
    save_objectness_function_handles(args)

    # tensorflow session
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)

    # input and output tensors
    if "DeepScores_300dpi" in args.dataset:
        input = tf.compat.v1.placeholder(tf.float32, shape=[None, None, None, 1])
        resnet_dir = args.pretrained_dir + "/DeepScores/"
        refinenet_dir = args.pretrained_dir + "/DeepScores_semseg/"

    elif "DeepScores" in args.dataset:
        input = tf.compat.v1.placeholder(tf.float32, shape=[None, None, None, 1])
        resnet_dir = args.pretrained_dir + "/DeepScores/"
        refinenet_dir = args.pretrained_dir + "/DeepScores_semseg/"

    elif "MUSICMA" in args.dataset:
        input = tf.compat.v1.placeholder(tf.float32, shape=[None, None, None, 1])
        resnet_dir = args.pretrained_dir + "/DeepScores/"
        refinenet_dir = args.pretrained_dir + "/DeepScores_semseg/"

    elif "macrophages" in args.dataset:
        input = tf.compat.v1.placeholder(tf.float32, shape=[None, None, None, 3])
        resnet_dir = args.pretrained_dir + "/ImageNet/"
        refinenet_dir = args.pretrained_dir + "/VOC2012/"

    elif "Dota_2018" in args.dataset:
        input = tf.compat.v1.placeholder(tf.float32, shape=[None, None, None, 3])
        resnet_dir = args.pretrained_dir + "/ImageNet/"
        refinenet_dir = args.pretrained_dir + "/VOC2012/"

    else:
        input = tf.compat.v1.placeholder(tf.float32, shape=[None, None, None, 3])
        resnet_dir = args.pretrained_dir + "/ImageNet/"
        refinenet_dir = args.pretrained_dir + "/VOC2012/"

    if not (len(args.training_help) == 1 and args.training_help[0] is None):
        # initialize helper_input
        helper_input = tf.compat.v1.placeholder(tf.float32, shape=[None, None, None, input.shape[-1] + 1])
        feed_head = slim.conv2d(helper_input, input.shape[-1], [3, 3], scope='gt_feed_head')
        input = feed_head

    print("Initializing Model:" + args.model)
    # model has all possible output heads (even if unused) to ensure saving and loading goes smoothly
    network_heads, init_fn = build_dwd_net(
        input, model=args.model, num_classes=nr_classes, pretrained_dir=resnet_dir, max_energy=args.max_energy,
        individual_upsamp = args.individual_upsamp, assigns=args.training_assignements, substract_mean=False, n_filters=args.n_filters)

    # use just one image summary OP for all tasks
    # train
    train_pred_placeholder = tf.compat.v1.placeholder(tf.uint8, shape=[1, None, None, 3])
    images_sums = []
    images_placeholders = []

    images_placeholders.append(train_pred_placeholder)
    images_sums.append(tf.compat.v1.summary.image('DWD_debug_train_img', train_pred_placeholder))
    train_images_summary_op = tf.compat.v1.summary.merge(images_sums)

    # valid
    valid_pred_placeholder = tf.compat.v1.placeholder(tf.uint8, shape=[1, None, None, 3])
    images_sums = []

    images_placeholders.append(valid_pred_placeholder)
    images_sums.append(tf.compat.v1.summary.image('DWD_debug_valid_img', valid_pred_placeholder))
    valid_images_summary_op = tf.compat.v1.summary.merge(images_sums)

    images_summary_op = [train_images_summary_op, valid_images_summary_op]


    # initialize tasks
    preped_assign = []
    for aid, assign in enumerate(args.training_assignements):
        [loss, optim, gt_placeholders, scalar_summary_op,
         mask_placholders] = initialize_assignement(aid, assign, imdb, network_heads, sess, data_layer, input, args)
        preped_assign.append(
            [loss, optim, gt_placeholders, scalar_summary_op, images_summary_op, images_placeholders, mask_placholders])

    # init tensorflow session
    saver = tf.compat.v1.train.Saver(max_to_keep=1000)
    sess.run(tf.compat.v1.global_variables_initializer())

    # load model weights
    checkpoint_dir = get_checkpoint_dir(args)
    checkpoint_name = "backbone"
    if args.continue_training == "Last":
        shutil.rmtree(checkpoint_dir)
        old_path, nr = checkpoint_dir.rsplit("_",1)
        old_path = old_path + "_" + str(int(nr) - 1)
        shutil.copytree(old_path, checkpoint_dir)
        args.continue_training = "True"

    if args.continue_training == "True":
        print("Loading checkpoint")
        saver.restore(sess, checkpoint_dir + "/" + checkpoint_name)
    elif args.pretrain_lvl == "deepscores_to_musicma":
        pretrained_vars = []
        for var in slim.get_model_variables():
            if not ("class_pred" in var.name):
                pretrained_vars.append(var)
        print("Loading network pretrained on Deepscores for Muscima")
        loading_checkpoint_name = args.pretrained_dir + "/DeepScores_to_Muscima/" + "backbone"
        init_fn = slim.assign_from_checkpoint_fn(loading_checkpoint_name, pretrained_vars)
        init_fn(sess)
    elif args.pretrain_lvl == "DeepScores_to_300dpi":
        pretrained_vars = []
        for var in slim.get_model_variables():
            if not ("class_pred" in var.name):
                pretrained_vars.append(var)
        print("Loading network pretrained on Deepscores for Muscima")
        loading_checkpoint_name = args.pretrained_dir+ "/DeepScores_to_300dpi/" + "backbone"
        init_fn = slim.assign_from_checkpoint_fn(loading_checkpoint_name, pretrained_vars)
        init_fn(sess)
    else:
        if args.pretrain_lvl == "semseg" and init_fn is not None:
            # load all variables except the ones in scope "deep_watershed"
            pretrained_vars = []
            for var in slim.get_model_variables():
                if not ("deep_watershed" in var.name or "gt_feed_head" in var.name):
                    pretrained_vars.append(var)

            print("Loading network pretrained on semantic segmentation")
            loading_checkpoint_name = refinenet_dir + args.model + ".ckpt"
            init_fn = slim.assign_from_checkpoint_fn(loading_checkpoint_name, pretrained_vars)
            init_fn(sess)
        elif args.pretrain_lvl == "class" and init_fn is not None:
            print("Loading pretrained weights for level: " + args.pretrain_lvl)
            init_fn(sess)
        else:
            print("Not loading a pretrained network")

    #sess.run(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='deep_watershed/energy_logits_pair0'))

    # set up tensorboard
    writer = tf.compat.v1.summary.FileWriter(checkpoint_dir, sess.graph)
    # store config
    with open(checkpoint_dir+"/"+datetime.datetime.now().isoformat()+
            "__"+fingerprint+".txt", "w") as text_config:
        text_config.write(str(args_txt))
    # pickle config
    with open(checkpoint_dir + "/" + datetime.datetime.now().isoformat() +
              "__" + fingerprint + ".p", "wb") as pickle_config:
        pickle.dump(args_txt,pickle_config)


    if args.train_only_combined != "True":
        # execute tasks
        for do_a in args.do_assign:
            assign_nr = do_a["assign"]
            do_itr = do_a["Itrs"]
            training_help = args.training_help[do_a["help"]]
            iteration = execute_assign(args, input, saver, sess, checkpoint_dir, checkpoint_name, data_layer, writer,
                                       network_heads,
                                       do_itr, args.training_assignements[assign_nr], preped_assign[assign_nr], iteration,
                                       training_help,fingerprint)

    # execute combined tasks
    for do_comb_a in args.combined_assignements:
        do_comb_itr = do_comb_a["Itrs"]
        rm_length = do_comb_a["Running_Mean_Length"]
        loss_factors = do_comb_a["loss_factors"]
        orig_assign = [args.training_assignements[i] for i in do_comb_a["assigns"]]
        preped_assigns = [preped_assign[i] for i in do_comb_a["assigns"]]
        training_help = None  # unused atm
        execute_combined_assign(args, data_layer, training_help, orig_assign, preped_assigns, loss_factors, do_comb_itr,
                                iteration, input, rm_length,
                                network_heads, sess, checkpoint_dir, checkpoint_name, saver, writer,fingerprint)

    print("done :)")

def run_batch_combined_assign():
    return "fetched_loss"


def execute_combined_assign(args, data_layer, training_help, orig_assign, preped_assigns, loss_factors, do_comb_itr,
                            iteration, input_ph, rm_length,
                            network_heads, sess, checkpoint_dir, checkpoint_name, saver, writer,fingerprint):
    # init data layer
    if args.prefetch == "True":
        data_layer[0] = PrefetchWrapper(data_layer[0].forward, args.prefetch_len, args.prefetch_size, args.prefetch_cache_dir, args.prefetch_proc,fingerprint, args, orig_assign, training_help)

    # combine losses
    if rm_length is not None:
        past_losses = np.ones((len(loss_factors), rm_length), np.float32)

    loss_scalings_placeholder = tf.compat.v1.placeholder(tf.float32, [len(loss_factors)])
    loss_tot = None
    for i in range(len(preped_assigns)):
        if loss_tot is None:
            loss_tot = preped_assigns[i][0] * loss_scalings_placeholder[i]
        else:
            loss_tot += preped_assigns[i][0] * loss_scalings_placeholder[i]

    # init optimizer -- make sure optimizer is fresh
    with tf.compat.v1.variable_scope("combined_opt" + str(0), reuse=False):
        var_list = [var for var in tf.compat.v1.trainable_variables() if "BatchNorm" not in var.name and "bias" not in var.name]

        # maybe exclude shortcuts from regularization
        var_list_res = [var for var in var_list if "resnet_" in var.name]
        var_list_not_res = [var for var in var_list if "resnet_" not in var.name]
        loss_L2_downsam = tf.add_n([tf.nn.l2_loss(v) for v in var_list_res]) * args.regularization_coefficient_downsamp
        loss_L2_upsam = tf.add_n([tf.nn.l2_loss(v) for v in var_list_not_res]) * args.regularization_coefficient_upsamp

        loss_tot_reg = loss_tot + loss_L2_downsam + loss_L2_upsam
        optimizer_type = args.optim
        if args.optim == 'rmsprop':
            optim = tf.compat.v1.train.RMSPropOptimizer(learning_rate=args.learning_rate, decay=0.995).minimize(loss_tot_reg,
                                                                                                      var_list=var_list)
        elif args.optim == 'adam':
            optim = tf.compat.v1.train.AdamOptimizer(learning_rate=args.learning_rate).minimize(loss_tot_reg, var_list=var_list)
        else:
            optim = tf.compat.v1.train.MomentumOptimizer(learning_rate=args.learning_rate, momentum=0.9).minimize(loss_tot_reg,
                                                                                                        va_list=var_list)
    opt_inizializers = [var.initializer for var in tf.compat.v1.global_variables() if "combined_opt" + str(0) in var.name]
    sess.run(opt_inizializers)
    # compute step
    print("training on combined assignments")
    print("for " + str(do_comb_itr) + " iterations")

    # waste elements off queue because qu0.eue clear does not work
    for i in range(14):
        data_layer[0].forward(args, orig_assign, training_help)

    for itr in range(iteration, (iteration + do_comb_itr)):
        # load batch - only use batches with content
        # batch_not_loaded = True
        # while batch_not_loaded:
        #     blob = data_layer[0].forward(args, orig_assign, training_help)
        #     batch_not_loaded = len(blob[0][0]["gt_boxes"].shape) != 3 or sum(["assign" in key for key in blob[0][0].keys()]) != len(preped_assigns)
        #
        # feed_dict = {}
        # data_list = []
        # for ind_batch, batch_ele in enumerate(blob):
        #     sub_data_list = []
        #     for ind_sub_batch, sub_batch_ele in enumerate(batch_ele):
        #         sub_data_list.append(sub_batch_ele["data"])
        #     # stack input data
        #     if len(sub_data_list) > 1:
        #         data_list.append(np.concatenate(sub_data_list, -1))
        #     else:
        #         data_list.append(sub_data_list[0])
        #
        # # stack minibatch
        # if len(data_list) > 1:
        #     feed_dict[input_ph] = np.concatenate(data_list, 0)
        # else:
        #     feed_dict[input_ph] = data_list[0]
        #
        # # pad with zeros if last dim is exactly 2
        # if feed_dict[input_ph].shape[-1] == 2:
        #     feed_dict[input_ph] = np.concatenate([feed_dict[input_ph], np.zeros(feed_dict[input_ph].shape[:-1] + (1,))], -1)
        #
        # # iterate over assignements
        # for i1 in range(len(preped_assigns)):
        #     gt_placeholders = preped_assigns[i1][2]
        #     mask_placeholders = preped_assigns[i1][6]
        #     # iterate over sub-batch
        #     for index_sb, (gt_sb, mask_sb) in enumerate(zip(gt_placeholders, mask_placeholders)):
        #         # iterate over downsampling
        #         for index_ds, (gt_ds, mask_ds) in enumerate(zip(gt_sb, mask_sb)):
        #             # concat over batch axis
        #             feed_dict[gt_ds] = np.concatenate([batch_ele[index_sb]["assign" + str(i1)]["gt_map" + str(len(gt_sb) - 1 - index_ds)] for batch_ele in blob], 0)
        #             feed_dict[mask_ds] = np.stack([batch_ele[index_sb]["assign" + str(i1)]["mask" + str(len(gt_sb) - 1 - index_ds)] for batch_ele in blob], 0)
        feed_dict, blob = load_feed_dict(data_layer, args, orig_assign, training_help, input_ph, preped_assigns)

        # compute running mean for losses
        if rm_length is not None:
            feed_dict[loss_scalings_placeholder] = loss_factors / np.maximum(np.mean(past_losses, 1), np.repeat(1.0E-6, past_losses.shape[0]))
        else:
            feed_dict[loss_scalings_placeholder] = loss_factors

        # with open('feed_dict_train.pickle', 'wb') as handle:
        #    pickle.dump(feed_dict[input_ph], handle, protocol=pickle.HIGHEST_PROTOCOL)

        #train step
        fetch_list = list()
        fetch_list.append(optim)
        fetch_list.append(loss_tot)
        fetch_list.append(loss_tot_reg)
        #fetch_list.append(loss_L2)
        for preped_a in preped_assigns:
            fetch_list.append(preped_a[0])
        fetches = sess.run(fetch_list, feed_dict=feed_dict)

        if rm_length is not None:
            past_losses[:, :-1] = past_losses[:, 1:]  # move by one timestep
            past_losses[:, -1] = fetches[-past_losses.shape[0]:]  # add latest loss

        if itr % args.print_interval == 0 or itr == 1:
            print("loss unregularized at  itr: " + str(itr) + " at: "+ str(datetime.datetime.now())+":"+str(fetches[1]))
            print("loss regularized at  itr: " + str(itr) + " at: " + str(datetime.datetime.now()) + ":" + str(fetches[2]))

            if rm_length is not None:
                print(past_losses)

        if itr % args.tensorboard_interval == 0 or itr == 1:

            post_assign_to_tensorboard(orig_assign, preped_assigns, network_heads, feed_dict, itr, sess, writer, blob)

        if itr % args.validation_loss_task == 0 or itr == 1:
            # approximate validation loss
            val_loss = 0
            for i in range(args.validation_loss_task_nr_batch):
                feed_dict, blob = load_feed_dict(data_layer, args, orig_assign, training_help, input_ph, preped_assigns,  valid=1)
                if rm_length is not None:
                    feed_dict[loss_scalings_placeholder] = loss_factors / np.maximum(np.mean(past_losses, 1), np.repeat(1.0E-6, past_losses.shape[0]))
                else:
                    feed_dict[loss_scalings_placeholder] = loss_factors
                loss_fetch = sess.run([loss_tot], feed_dict=feed_dict)
                val_loss += loss_fetch[0]

            val_loss = val_loss/args.validation_loss_task_nr_batch
            print("Validation loss estimate at itr " + str(itr) + ": " + str(val_loss))

            post_assign_to_tensorboard(orig_assign, preped_assigns, network_heads, feed_dict, itr, sess, writer, blob, valid=1)



        if itr % args.save_interval == 0:
            print("saving weights")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver.save(sess, checkpoint_dir + "/" + checkpoint_name)

    iteration = (iteration + do_comb_itr)
    if args.prefetch == "True":
        data_layer.kill()

    return iteration


def post_assign_to_tensorboard(orig_assign, preped_assigns, network_heads, feed_dict, itr, sess, writer, blob, valid = 0):

    gt_visuals = []
    map_visuals = []
    # post scalar summary per assign, store fetched maps
    for i in range(len(preped_assigns)):
        assign = orig_assign[i]
        _, _, _, scalar_summary_op, images_summary_op, images_placeholders, _ = preped_assigns[i]
        fetch_list = [scalar_summary_op[valid]]
        # fetch sub_predicitons
        # nr_feature_maps = len(network_heads[assign["stamp_func"][0]][assign["stamp_args"]["loss"]])
        #
        # [fetch_list.append(
        #     network_heads[assign["stamp_func"][0]][assign["stamp_args"]["loss"]][nr_feature_maps - (x + 1)]) for x
        #     in
        #     range(len(assign["ds_factors"]))]

        for nw_head_key in network_heads.keys():
            if i == int(nw_head_key.split("_")[0]):
                nr_feature_maps = len(network_heads[nw_head_key])

                [fetch_list.append(network_heads[nw_head_key][nr_feature_maps - (x + 1)]) for x
                    in range(len(assign["ds_factors"]))]


        summary = sess.run(fetch_list, feed_dict=feed_dict)
        writer.add_summary(summary[0], float(itr))

        gt_visual = get_gt_visuals(blob, assign, i, pred_boxes=None, show=False)
        map_visual = get_map_visuals(summary[1:], assign, show=False)
        gt_visuals.append(gt_visual)
        map_visuals.append(map_visual)

    # stitch one large image out of all assigns
    stitched_img = get_stitched_tensorboard_image(orig_assign, gt_visuals, map_visuals, blob, itr)
    stitched_img = np.expand_dims(stitched_img, 0)
    #obsolete
    #images_feed_dict = get_images_feed_dict(assign, blob, gt_visuals, map_visuals, images_placeholders)
    images_feed_dict = dict()
    images_feed_dict[images_placeholders[valid]] = stitched_img

    # save images to tensorboard
    summary = sess.run([images_summary_op[valid]], feed_dict=images_feed_dict)
    writer.add_summary(summary[0], float(itr))
    

    return None


def focal_loss(prediction_tensor, target_tensor, weights=None, alpha=0.25, gamma=2):
    r"""Compute focal loss for predictions.
        Multi-labels Focal loss formula:
            FL = -alpha * (z-p)^gamma * log(p) -(1-alpha) * p^gamma * log(1-p)
                 ,which alpha = 0.25, gamma = 2, p = sigmoid(x), z = target_tensor.
    Args:
     prediction_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing the predicted logits for each class
     target_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing one-hot encoded classification targets
     weights: A float tensor of shape [batch_size, num_anchors]
     alpha: A scalar tensor for focal loss alpha hyper-parameter
     gamma: A scalar tensor for focal loss gamma hyper-parameter
    Returns:
        loss: A (scalar) tensor representing the value of the loss function
    """
    softmax_p = tf.nn.softmax(prediction_tensor)
    zeros = array_ops.zeros_like(softmax_p, dtype=softmax_p.dtype)

    # For poitive prediction, only need consider front part loss, back part is 0;
    # target_tensor > zeros <=> z=1, so poitive coefficient = z - p.
    pos_p_sub = array_ops.where(target_tensor > zeros, target_tensor - softmax_p, zeros)
    # For negative prediction, only need consider back part loss, front part is 0;
    # target_tensor > zeros <=> z=1, so negative coefficient = 0.
    neg_p_sub = array_ops.where(target_tensor > zeros, zeros, softmax_p)
    per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.math.log(tf.clip_by_value(softmax_p, 1e-8, 1.0)) \
                          - (1 - alpha) * (neg_p_sub ** gamma) * tf.math.log(tf.clip_by_value(1.0 - softmax_p, 1e-8, 1.0))
    # print(tf.reduce_mean(per_entry_cross_ent))
    return per_entry_cross_ent
    # return tf.reduce_mean(per_entry_cross_ent)


def initialize_assignement(aid, assign, imdb, network_heads, sess, data_layer, input, args):
    gt_placeholders = get_gt_placeholders(assign, imdb, args.paired_data, args.nr_classes[0], args)

    loss_mask_placeholders = []
    for pair_nr in range(args.paired_data):
        loss_mask_placeholders.append([tf.compat.v1.placeholder(tf.float32, shape=[None, None, None, 1]) for x in assign["ds_factors"]])


    pair_contrib_loss = []
    for pair_nr in range(args.paired_data):
        debug_fetch = dict()
        s_task_id = str(aid)+"_"+str(pair_nr)
        if assign["stamp_func"][0] == "stamp_directions":
            loss_components = []
            for x in range(len(assign["ds_factors"])):
                # TODO ignored for the moment
                raise NotImplementedError

                debug_fetch[str(x)] = dict()
                # # mask, where gt is zero
                split1, split2 = tf.split(gt_placeholders[x], 2, -1)
                debug_fetch[str(x)]["split1"] = split1

                mask = tf.squeeze(split1 > 0, -1)
                debug_fetch[str(x)]["mask"] = mask

                masked_pred = tf.boolean_mask(tensor=network_heads[s_task_id][x], mask=mask)
                debug_fetch[str(x)]["masked_pred"] = masked_pred

                masked_gt = tf.boolean_mask(tensor=gt_placeholders[x], mask=mask)
                debug_fetch[str(x)]["masked_gt"] = masked_gt

                # norm prediction
                norms = tf.norm(tensor=masked_pred, ord="euclidean", axis=-1, keepdims=True)
                masked_pred = masked_pred / norms
                debug_fetch[str(x)]["masked_pred_normed"] = masked_pred

                gt_1, gt_2 = tf.split(masked_gt, 2, -1)
                pred_1, pred_2 = tf.split(masked_pred, 2, -1)
                inner_2 = gt_1 * pred_1 + gt_2 * pred_2
                debug_fetch[str(x)]["inner_2"] = inner_2
                inner_2 = tf.maximum(tf.constant(-1, dtype=tf.float32),
                                     tf.minimum(tf.constant(1, dtype=tf.float32), inner_2))

                acos_inner = tf.acos(inner_2)
                debug_fetch[str(x)]["acos_inner"] = acos_inner

                loss_components.append(acos_inner)
        else:
            nr_feature_maps = len(network_heads[s_task_id])
            nr_ds_factors = len(assign["ds_factors"])
            if assign["stamp_args"]["loss"] == "softmax":
                loss_components = [tf.nn.softmax_cross_entropy_with_logits(
                    logits=network_heads[s_task_id][
                        nr_feature_maps - nr_ds_factors + x],
                    labels=tf.stop_gradient(gt_placeholders[pair_nr][x]), axis=-1) for x in range(nr_ds_factors)]
                # loss_components = [focal_loss(prediction_tensor=network_heads[assign["stamp_func"][0]][assign["stamp_args"]["loss"]][nr_feature_maps-nr_ds_factors+x],
                #                                                target_tensor=gt_placeholders[x]) for x in range(nr_ds_factors)]


                for x in range(nr_ds_factors):
                    debug_fetch["logits_" + str(x)] = network_heads[s_task_id][
                        nr_feature_maps - nr_ds_factors + x]
                    debug_fetch["labels" + str(x)] = gt_placeholders[pair_nr][x]
                debug_fetch["loss_components_softmax"] = loss_components
            else:
                loss_components = [tf.compat.v1.losses.mean_squared_error(
                    predictions=network_heads[s_task_id][
                        nr_feature_maps - nr_ds_factors + x],
                    labels=gt_placeholders[pair_nr][x], reduction="none") for x in range(nr_ds_factors)]
                debug_fetch["loss_components_mse"] = loss_components

        # apply loss mask
        comp_multy = []
        for i in range(len(loss_components)):
            # maybe expand dims
            if len(loss_components[i].shape) == 3:
                cond_result = tf.expand_dims(loss_components[i], -1)
            else:
                cond_result = loss_components[i]
            comp_multy.append(tf.multiply(cond_result, loss_mask_placeholders[pair_nr][i]))
        # call tf.reduce mean on each loss component
        final_loss_components = [tf.reduce_mean(input_tensor=x) for x in comp_multy]

        stacked_components = tf.stack(final_loss_components)

        if assign["layer_loss_aggregate"] == "min":
            loss = tf.reduce_min(input_tensor=stacked_components)
        elif assign["layer_loss_aggregate"] == "avg":
            loss = tf.reduce_mean(input_tensor=stacked_components)
        else:
            raise NotImplementedError("unknown layer aggregate")

        pair_contrib_loss.append(loss)

        # ---------------------------------------------------------------------
        # Debug code -- THIS HAS TO BE COMMENTED OUT UNLESS FOR DEBUGGING
        #
        # sess.run(tf.global_variables_initializer())
        # blob = data_layer[0].forward(args, [assign], None)
        #
        # feed_dict = {}
        #
        # data_list = []
        # for ind_batch, batch_ele in enumerate(blob):
        #     sub_data_list = []
        #     for ind_sub_batch, sub_batch_ele in enumerate(batch_ele):
        #         sub_data_list.append(sub_batch_ele["data"])
        #
        #     # stack input data
        #     if len(sub_data_list) > 1:
        #         data_list.append(np.concatenate(sub_data_list, -1))
        #     else:
        #         data_list.append(sub_data_list[0])
        #
        # # stack minibatch
        # if len(data_list) > 1:
        #     feed_dict[input] = np.concatenate(data_list, 0)
        # else:
        #     feed_dict[input] = data_list[0]
        #
        #
        # # pad with zeros if last dim is exactly 2
        # if feed_dict[input].shape[-1] == 2:
        #     feed_dict[input] = np.concatenate([feed_dict[input], np.zeros(feed_dict[input].shape[:-1] + (1,))], -1)
        #
        # # iterate over sub-batch
        # for index_sb, (gt_sb, mask_sb) in enumerate(zip(gt_placeholders, loss_mask_placeholders)):
        #     # iterate ds-factor
        #     for index_ds, (gt_ds, mask_ds) in enumerate(zip(gt_sb, mask_sb)):
        #         # concat over batch axis
        #         feed_dict[gt_ds] = np.concatenate([batch_ele[index_sb]["assign0"]["gt_map" + str(len(gt_sb) - 1 - index_ds)] for batch_ele in blob], 0)
        #         feed_dict[mask_ds] = np.stack([batch_ele[index_sb]["assign0"]["mask" + str(len(gt_sb) - 1 - index_ds)] for batch_ele in blob], 0)
        #
        # # train step
        # loss_fetch = sess.run(debug_fetch, feed_dict=feed_dict)
        # loss_fetch_1 = sess.run(loss, feed_dict=feed_dict)
        # end debug code
        # ---------------------------------------------------------------------

    stacked_components = tf.stack(pair_contrib_loss)
    loss = tf.reduce_mean(input_tensor=stacked_components)
    if args.train_only_combined != "True":
        # init optimizer
        var_list = [var for var in tf.compat.v1.trainable_variables()]
        optimizer_type = args.optim
        loss_L2 = tf.add_n([tf.nn.l2_loss(v) for v in var_list
                            if 'bias' not in v.name]) * args.regularization_coefficient
        loss += loss_L2
        if optimizer_type == 'rmsprop':
            optim = tf.compat.v1.train.RMSPropOptimizer(learning_rate=args.learning_rate, decay=0.995).minimize(loss,
                                                                                                      var_list=var_list)
        elif optimizer_type == 'adam':
            optim = tf.compat.v1.train.AdamOptimizer(learning_rate=args.learning_rate).minimize(loss, var_list=var_list)
        else:
            optim = tf.compat.v1.train.MomentumOptimizer(learning_rate=args.learning_rate, momentum=0.9).minimize(loss,
                                                                                                        var_list=var_list)
    else:
        optim = None

    # init summary operations
    # define summary ops
    scalar_sums = []
    scalar_sums.append(tf.compat.v1.summary.scalar("train_loss_" + get_config_id(assign) + "_", loss))

    for comp_nr in range(len(loss_components)):
        scalar_sums.append(tf.compat.v1.summary.scalar("train_loss_component_" + get_config_id(assign) + "Nr" + str(comp_nr) + "_",
                                             final_loss_components[comp_nr]))

    train_scalar_summary_op = tf.compat.v1.summary.merge(scalar_sums)

    scalar_sums = []
    scalar_sums.append(tf.compat.v1.summary.scalar("valid_loss_" + get_config_id(assign) + "_", loss))

    for comp_nr in range(len(loss_components)):
        scalar_sums.append(tf.compat.v1.summary.scalar("valid_loss_component_" + get_config_id(assign) + "Nr" + str(comp_nr) + "_",
                                             final_loss_components[comp_nr]))

    valid_scalar_summary_op = tf.compat.v1.summary.merge(scalar_sums)

    scalar_summary_op = [train_scalar_summary_op, valid_scalar_summary_op]


    return loss, optim, gt_placeholders, scalar_summary_op, loss_mask_placeholders


def load_feed_dict(data_layer, args, assign, training_help, input_ph, preped_assigns, valid=0):

    # load batch - only use batches with content
    batch_not_loaded = True
    while batch_not_loaded:
        blob = data_layer[valid].forward(args, assign, training_help)
        batch_not_loaded = len(blob[0][0]["gt_boxes"].shape) != 3 or sum(["assign" in key for key in blob[0][0].keys()]) != len(preped_assigns)

    feed_dict = {}
    data_list = []
    for ind_batch, batch_ele in enumerate(blob):
        sub_data_list = []
        for ind_sub_batch, sub_batch_ele in enumerate(batch_ele):
            sub_data_list.append(sub_batch_ele["data"])
        # stack input data
        if len(sub_data_list) > 1:
            data_list.append(np.concatenate(sub_data_list, -1))
        else:
            data_list.append(sub_data_list[0])

    # stack minibatch
    if len(data_list) > 1:
        feed_dict[input_ph] = np.concatenate(data_list, 0)
    else:
        feed_dict[input_ph] = data_list[0]

    # pad with zeros if last dim is exactly 2
    if feed_dict[input_ph].shape[-1] == 2:
        feed_dict[input_ph] = np.concatenate([feed_dict[input_ph], np.zeros(feed_dict[input_ph].shape[:-1] + (1,))], -1)

    # iterate over assignements
    for i1 in range(len(preped_assigns)):
        gt_placeholders = preped_assigns[i1][2]
        mask_placeholders = preped_assigns[i1][6]
        # iterate over sub-batch
        for index_sb, (gt_sb, mask_sb) in enumerate(zip(gt_placeholders, mask_placeholders)):
            # iterate over downsampling
            for index_ds, (gt_ds, mask_ds) in enumerate(zip(gt_sb, mask_sb)):
                try:
                    # concat over batch axis
                    feed_dict[gt_ds] = np.concatenate([batch_ele[index_sb]["assign" + str(i1)]["gt_map" + str(len(gt_sb) - 1 - index_ds)] for batch_ele in blob], 0)
                    feed_dict[mask_ds] = np.stack([batch_ele[index_sb]["assign" + str(i1)]["mask" + str(len(gt_sb) - 1 - index_ds)] for batch_ele in blob], 0)
                except:
                    print("hoyeasdf")

    return feed_dict, blob



def execute_assign(args, input_placeholder, saver, sess, checkpoint_dir, checkpoint_name, data_layer, writer, network_heads,
                   do_itr, assign, prepped_assign, iteration, training_help, fingerprint):
    loss, optim, gt_placeholders, scalar_summary_op, images_summary_op, images_placeholders, mask_placeholders = prepped_assign

    if args.prefetch == "True":
        data_layer[0] = PrefetchWrapper(data_layer[0].forward, args.prefetch_len, args.prefetch_size, args.prefetch_cache_dir, args.prefetch_proc, fingerprint, args, args, [assign], training_help)

    print("training on:" + str(assign))
    print("for " + str(do_itr) + " iterations")
    for itr in range(iteration, (iteration + do_itr)):

        # run a training batch
        feed_dict, blob = load_feed_dict(data_layer, args, [assign], training_help, input_placeholder, [prepped_assign])

        _, loss_fetch = sess.run([optim, loss], feed_dict=feed_dict)

        if itr % args.print_interval == 0 or itr == 1:
            print("loss at itr: " + str(itr))
            print(loss_fetch)

        if itr % args.tensorboard_interval == 0 or itr == 1:

            # fetch_list = [scalar_summary_op[0]]
            #
            # # fetch sub_predicitons for each sub_batch
            # for nw_heads_sub_b in network_heads:
            #     nr_feature_maps = len(nw_heads_sub_b[assign["stamp_func"][0]][assign["stamp_args"]["loss"]])
            #
            #     [fetch_list.append(
            #         nw_heads_sub_b[assign["stamp_func"][0]][assign["stamp_args"]["loss"]][nr_feature_maps - (x + 1)]) for x
            #         in
            #         range(len(assign["ds_factors"]))]
            #
            # summary = sess.run(fetch_list, feed_dict=feed_dict)
            # writer.add_summary(summary[0], float(itr))
            #
            # # feed one stitched image to summary op
            # gt_visuals = get_gt_visuals(blob, assign, 0, pred_boxes=None, show=False)
            # map_visuals = get_map_visuals(summary[1:], assign, show=False)
            #
            # stitched_img = get_stitched_tensorboard_image([assign], [gt_visuals], [map_visuals], blob, itr)
            # stitched_img = np.expand_dims(stitched_img, 0)
            # # obsolete
            # #images_feed_dict = get_images_feed_dict(assign, blob, None, None, images_placeholders)
            # images_feed_dict = dict()
            # images_feed_dict[images_placeholders[0]] = stitched_img
            #
            # # save images to tensorboard
            # summary = sess.run([images_summary_op[0]], feed_dict=images_feed_dict)
            # writer.add_summary(summary[0], float(itr))

            post_assign_to_tensorboard([assign], [prepped_assign], network_heads, feed_dict, itr, sess, writer, blob, valid=0)

        if itr % args.validation_loss_task == 0:
            # approximate validation loss
            val_loss = 0
            for i in range(args.validation_loss_task_nr_batch):
                feed_dict, blob = load_feed_dict(data_layer, args, [assign], training_help, input_placeholder, [prepped_assign],  valid=1)
                loss_fetch = sess.run([loss], feed_dict=feed_dict)
                val_loss += loss_fetch[0]

            val_loss = val_loss/args.validation_loss_task_nr_batch
            print("Validation loss estimate at itr " + str(itr) + ": " + str(val_loss))


            # # post to tensorboard
            # fetch_list = [scalar_summary_op[1]]
            # # fetch sub_predicitons
            # for nw_heads_sub_b in network_heads:
            #     nr_feature_maps = len(nw_heads_sub_b[assign["stamp_func"][0]][assign["stamp_args"]["loss"]])
            #
            #     [fetch_list.append(
            #         nw_heads_sub_b[assign["stamp_func"][0]][assign["stamp_args"]["loss"]][nr_feature_maps - (x + 1)]) for x
            #         in
            #         range(len(assign["ds_factors"]))]
            #
            # summary = sess.run(fetch_list, feed_dict=feed_dict)
            # writer.add_summary(summary[0], float(itr))
            #
            # # feed one stitched image to summary op
            # gt_visuals = get_gt_visuals(blob, assign, 0, pred_boxes=None, show=False)
            # map_visuals = get_map_visuals(summary[1:], assign, show=False)
            #
            # stitched_img = get_stitched_tensorboard_image([assign], [gt_visuals], [map_visuals], blob, itr)
            # stitched_img = np.expand_dims(stitched_img, 0)
            # # obsolete
            # # images_feed_dict = get_images_feed_dict(assign, blob, None, None, images_placeholders)
            # images_feed_dict = dict()
            # images_feed_dict[images_placeholders[1]] = stitched_img
            #
            # # save images to tensorboard
            # summary = sess.run([images_summary_op[1]], feed_dict=images_feed_dict)
            # writer.add_summary(summary[0], float(itr))
            post_assign_to_tensorboard([assign], [prepped_assign], network_heads, feed_dict, itr, sess, writer, blob, valid=1)


        if itr % args.validation_loss_final == 0:
            print("do full validation")

        if itr % args.save_interval == 0:
            print("saving weights")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver.save(sess, checkpoint_dir + "/" + checkpoint_name)
            global store_dict
            if store_dict:
                print("Saving dictionary")
                dictionary = args.dict_info
                with open(os.path.join(checkpoint_dir, 'dict' + '.pickle'), 'wb') as handle:
                    pickle.dump(dictionary, handle, protocol=pickle.HIGHEST_PROTOCOL)
                store_dict = False  # we need to save the dict only once

    iteration = (iteration + do_itr)
    if args.prefetch == "True":
        data_layer.kill()

    return iteration


# def get_images_feed_dict(assign, blob, gt_visuals, map_visuals, images_placeholders):
#
#     # obsolete, should not be used!
#
#     feed_dict = dict()
#     # reverse map vis order
#     for i in range(len(assign["ds_factors"])):
#         feed_dict[images_placeholders[i]] = np.concatenate([gt_visuals[i], map_visuals[i]])
#
#     for key in feed_dict.keys():
#         feed_dict[key] = np.expand_dims(feed_dict[key], 0)
#
#     if blob["helper"] is not None:
#         feed_dict[images_placeholders[len(images_placeholders) - 2]] = (
#                 blob["helper"] / np.max(blob["helper"]) * 255).astype(np.uint8)
#     else:
#         data_shape = blob["data"].shape[: -1]+ (3,)
#         feed_dict[images_placeholders[len(images_placeholders) - 2]] = np.zeros(data_shape, dtype=np.uint8)
#
#     if blob["data"].shape[3] == 1:
#         img_data = np.concatenate([blob["data"], blob["data"], blob["data"]], -1).astype(np.uint8)
#     else:
#         img_data = blob["data"].astype(np.uint8)
#     feed_dict[images_placeholders[len(images_placeholders) - 1]] = img_data
#     return feed_dict

def get_spacer_pattern(shape):
    shape_a, shape_b, depth = shape

    x = np.arange(0, shape_b, 1)
    y = np.arange(0, shape_a, 1)
    xx, yy = np.meshgrid(x, y, sparse=True)
    z = (xx + yy)%100
    spacer = z/np.max(z)*255
    spacer[spacer==0]=150
    spacer = np.stack([spacer*0,spacer,spacer*0], axis=-1)

    return spacer



def get_stitched_tensorboard_image(assign, gt_visuals, map_visuals, blob, itr):
    pix_spacer = 3

    # use first image of batch
    blob = blob[0]

    sub_b_img = []
    for sub_ind, sub_blob in enumerate(blob):
        #print("doit!")
        # input image + gt
        input_gt = overlayed_image(sub_blob["data"][0], gt_boxes=sub_blob["gt_boxes"][0], pred_boxes=None)

        # input image + prediction
        #TODO get actual predictions
        input_pred = overlayed_image(sub_blob["data"][0], gt_boxes=None, pred_boxes=sub_blob["gt_boxes"][0])

        # concat inputs
        conc = np.concatenate((input_gt, get_spacer_pattern((input_gt.shape[0], pix_spacer, 3)).astype("uint8"), input_pred), axis = 1)
        # im = Image.fromarray(conc)
        # im.save(sys.argv[0][:-17] + "asdfsadfa.png")

        # iterate over tasks
        for i in range(len(assign)):
            # concat task outputs
            for ii in range(len(assign[i]["ds_factors"])):
                sub_map = np.concatenate([gt_visuals[i][sub_ind][ii], get_spacer_pattern((gt_visuals[i][sub_ind][ii].shape[0], pix_spacer,3)).astype("uint8"), map_visuals[i][ii+sub_ind*len(assign[i]["ds_factors"])]], axis = 1)
                if sub_map.shape[1] != conc.shape[1]:
                    expand = get_spacer_pattern((sub_map.shape[0], conc.shape[1], sub_map.shape[2]))
                    expand[:, 0:sub_map.shape[1]] = sub_map
                    sub_map = expand.astype("uint8")
                conc = np.concatenate((conc, get_spacer_pattern((pix_spacer, conc.shape[1],3)).astype("uint8"), sub_map), axis = 0)


        # show loss masks if necessary
        show_masks = True
        if show_masks:
            for i in range(len(assign)):
                # concat task outputs
                for ii in range(len(assign[i]["ds_factors"])):
                    mask = sub_blob["assign"+str(i)]["mask"+str(ii)]
                    mask = mask/np.max(mask)*255
                    mask = np.concatenate([mask,mask,mask], -1)

                    sub_map = np.concatenate(
                        [gt_visuals[i][sub_ind][ii], get_spacer_pattern((gt_visuals[i][sub_ind][ii].shape[0], pix_spacer, 3)).astype("uint8"),
                         mask.astype("uint8")], axis=1)
                    if sub_map.shape[1] != conc.shape[1]:
                        expand = get_spacer_pattern((sub_map.shape[0], conc.shape[1], sub_map.shape[2]))
                        expand[:, 0:sub_map.shape[1]] = sub_map
                        sub_map = expand.astype("uint8")
                    conc = np.concatenate((conc, get_spacer_pattern((pix_spacer, conc.shape[1], 3)).astype("uint8"), sub_map), axis=0)
        sub_b_img.append(conc)
        sub_b_img.append(get_spacer_pattern((conc.shape[0], pix_spacer, 3)).astype("uint8"))
    conc = np.concatenate(sub_b_img, axis=1)

    # prepend additional info
    add_info = Image.fromarray(np.ones((50, conc.shape[1],3), dtype="uint8")*255)

    draw = ImageDraw.Draw(add_info)
    font = ImageFont.truetype("/usr/share/fonts/truetype/ubuntu-font-family/UbuntuMono-R.ttf", 18)
    draw.text((2, 2), "Iteration Nr: " + str(itr), (0, 0, 0), font=font)
    # add_info.save(sys.argv[0][:-17] + "add_info.png")
    add_info = np.asarray(add_info).astype("uint8")
    conc = np.concatenate((add_info, conc), axis=0)
    return conc

def get_gt_placeholders(assign, imdb, paired_data, nr_classes, args):
    gt_placehoders = []
    for pair_nr in range(paired_data):
        gt_dim = assign["stamp_func"][1](None, assign["stamp_args"], nr_classes, args)
        gt_placehoders.append([tf.compat.v1.placeholder(tf.float32, shape=[None, None, None, gt_dim]) for x in assign["ds_factors"]])
    return gt_placehoders


def get_config_id(assign):
    return assign["stamp_func"][0] + "_" + assign["stamp_args"]["loss"]


def get_checkpoint_dir(args):
    # assemble path
    if "300dpi" in args.dataset:
        image_mode = "300dpi"
    if "DeepScores" in args.dataset:
        image_mode = "music"
    elif "MUSICMA" in args.dataset:
        image_mode = "music_handwritten"
    elif "macrophages" in args.dataset:
        image_mode = "macrophages"
    elif "Dota" in args.dataset:
        image_mode = "Dota"
    else:
        image_mode = "realistic"
    tbdir = args.exp_dir + "/" + image_mode + "/" + "pretrain_lvl_" + args.pretrain_lvl + "/" + args.model
    if not os.path.exists(tbdir):
        os.makedirs(tbdir)
    runs_dir = os.listdir(tbdir)
    if args.continue_training == "True" or args.continue_training == "Last":
        tbdir = tbdir + "/" + "run_" + str(len(runs_dir) - 1)
    else:
        tbdir = tbdir + "/" + "run_" + str(len(runs_dir))
        os.makedirs(tbdir)
    return tbdir


def get_training_roidb(imdb, use_flipped):
    """Returns a roidb (Region of Interest database) for use in training."""
    if use_flipped:
        print('Appending horizontally-flipped training examples...')
        imdb.append_flipped_images()
        print('done')

    print('Preparing training data...')
    rdl_roidb.prepare_roidb(imdb)
    print('done')

    return imdb.roidb


def save_objectness_function_handles(args):
    FUNCTION_MAP = {'stamp_directions': stamp_directions,
                    'stamp_energy': stamp_energy,
                    'stamp_class': stamp_class,
                    'stamp_bbox': stamp_bbox,
                    'stamp_semseg': stamp_semseg
                    }

    for obj_setting in args.training_assignements:
        obj_setting["stamp_func"] = [obj_setting["stamp_func"], FUNCTION_MAP[obj_setting["stamp_func"]]]

    return args


def build_config_fingerprint(config):

    m = hashlib.sha224()
    relevant_args = [config.crop, config.crop_top_left_bias, config.augmentation_type, config.max_edge,
                     config.use_flipped,config.substract_mean,config.pad_to, config.pad_with, config.batch_size,
                     config.dataset, config.prefetch_size, config.max_energy, config.bbox_angle, config.class_estimation,
                     config.sparse_heads]
    for x in config.scale_list:
        relevant_args.append(x)
    relevant_args.append(json.dumps(config.training_assignements, sort_keys=True))

    for i in relevant_args:
        m.update(str(i).encode('utf-8'))

    return m.hexdigest()


def load_database(args):
    print("Setting up image database: " + args.dataset)
    imdb = get_imdb(args,args.dataset)
    print('Loaded dataset `{:s}` for training'.format(imdb.name))
    roidb = get_training_roidb(imdb, args.use_flipped == "True")
    print('{:d} roidb entries'.format(len(roidb)))

    if args.dataset_validation != "no":
        print("Setting up validation image database: " + args.dataset_validation)
        imdb_val = get_imdb(args,args.dataset_validation)
        print('Loaded dataset `{:s}` for validation'.format(imdb_val.name))
        roidb_val = get_training_roidb(imdb_val, False)
        print('{:d} roidb entries'.format(len(roidb_val)))
    else:
        imdb_val = None
        roidb_val = None

    data_layer = RoIDataLayer(roidb, imdb.num_classes, augmentation=args.augmentation_type)

    if roidb_val is not None:
        data_layer_val = RoIDataLayer(roidb_val, imdb_val.num_classes, random=True)

    return imdb, roidb, imdb_val, roidb_val, data_layer, data_layer_val


def get_nr_classes():
    return nr_classes
