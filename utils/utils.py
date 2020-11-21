import numpy as np
import cv2
import operator
import matplotlib.pyplot as plt


def load_weights(model, weights_file_path):
    conv_layer_size = 110
    conv_output_idxs = [93, 101, 109]
    with open(weights_file_path, 'rb') as f:
        major, minor, revision, seen, _ = np.fromfile(f, dtype=np.int32, count=5)
        bn_idx = 0
        for conv_idx in range(conv_layer_size):
            conv_layer_name = f'conv2d_{conv_idx}' if conv_idx > 0 else 'conv2d'
            bn_layer_name = f'batch_normalization_{bn_idx}' if bn_idx > 0 else 'batch_normalization'

            conv_layer = model.get_layer(conv_layer_name)
            filters = conv_layer.filters
            kernel_size = conv_layer.kernel_size[0]
            input_dims = conv_layer.input_shape[-1]

            if conv_idx not in conv_output_idxs:
                bn_weights = np.fromfile(f, dtype=np.float32, count=4*filters)
                bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]
                bn_layer = model.get_layer(bn_layer_name)
                bn_idx += 1
            else:
                conv_bias = np.fromfile(f, dtype=np.float32, count=filters)
            
            # darknet shape: (out_dim, input_dims, height, width)
            # tf shape: (height, width, input_dims, out_dim)
            conv_shape = (filters, input_dims, kernel_size, kernel_size)
            conv_weights = np.fromfile(f, dtype=np.float32, count=np.product(conv_shape))
            conv_weights = conv_weights.reshape(conv_shape).transpose([2, 3, 1, 0])

            if conv_idx not in conv_output_idxs:
                conv_layer.set_weights([conv_weights])
                bn_layer.set_weights(bn_weights)
            else:
                conv_layer.set_weights([conv_weights, conv_bias])
        
        if len(f.read()) == 0:
            print("all weight read")
        else:
            print(f'failed to read  all weights, # of unread weights: {len(f.read())}')


def get_detection_data(img, model_outputs, class_names):

    detections = []
    num_boxes = model_outputs[-1][0]
    
    boxes, scores, classes = [output[0][:num_boxes] for output in model_outputs[:-1]]
    h, w = img.shape[:2]
    
    for i, box in enumerate(boxes):
        xmin = int(box[0] * w)
        ymin = int(box[1] * h)
        xmax = int(box[2] * w)
        ymax = int(box[3] * h)
        print("{}, {}, {}, {}".format(xmin, ymin, xmax, ymax))
        box_w = xmax - xmin
        box_h = ymax - ymin
        score = scores[i]
        cls = class_names[int(classes[i])]
        detections.append([xmin, ymin, xmax, ymax, cls, score, box_w, box_h])
    
    return detections


def draw_bbox(img, detections, cmap, random_color=True, figsize=(10, 10), show_text=True):

    img = np.array(img)
    scale = max(img.shape[0:2]) / 416
    line_width = int(2 * scale)

    for _, detection in enumerate(detections):
        x1, y1, x2, y2, cls, score, w, h = detection 
        color = list(np.random.random(size=3)*255) if random_color else cmap[cls]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, line_width)
        if show_text:
            text = f'{cls} {score:.2f}'
            font = cv2.FONT_HERSHEY_DUPLEX
            font_scale = max(0.3 * scale, 0.3)
            thickness = max(int(1 * scale), 1)
            (text_width, text_height) = cv2.getTextSize(text, font, font_scale, thickness)[0]
            cv2.rectangle(img, (x1 - line_width//2, y1 - text_height), (x1 + text_width, y1), color, cv2.FILLED)
            cv2.putText(img, text, (x1, y1), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    plt.figure(figsize=figsize)
    plt.imshow(img)
    plt.show()
    return img


def voc_ap(rec, prec):

    """
    Calculate the AP given the recall and precision array
    1st) We compute a version of the measured precision/recall curve with
         precision monotonically decreasing
    2nd) We compute the AP as the area under this curve by numerical integration.
    """

    rec.insert(0, 0.0) # begining of list
    rec.append(1.0) # end of list
    prec.insert(0, 0.0) # begining of list
    prec.append(1.0) # end of list
    
    mrec = rec[:]
    mprec = prec[:]

    for i in range(len(mprec)-2, -1, -1):
        mprec[i] = max(mprec[i], mprec[i+1])

    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i-1]:
            i_list.append(i)
    
    ap = 0.0
    for i in i_list:
        ap += ((mrec[i] - mrec[i-1]) * mprec[i])
    return ap, mrec, mprec


def draw_plot_func(dictionary, n_classes, window_title, plot_title, x_label, output_path, to_show, plot_color, true_p_bar):
    
    sorted_dic = sorted(dictionary.items(), key=operator.itemgetter(1))
    sorted_keys, sorted_values = zip(*sorted_dic)

    if true_p_bar != "":
        """
        TP(True Positives): green
        FP(False Positives): red
        FN(False Negatives): pink
        """
        fp_sorted = []
        tp_sorted = []
        for key in sorted_keys:
            fp_sorted.append(dictionary[key] - true_p_bar[key])
            tp_sorted.append(true_p_bar[key])
        plt.barh(range(n_classes), fp_sorted, align='center', color='crimson', label='False Positives')
        plt.barh(range(n_classes), tp_sorted, align='center', color='forestgreen', label='True Positive', left=fp_sorted)
        plt.legend(loc='lower right')
        
        fig = plt.gcf()
        axes = plt.gca()
        r = fig.canvas.get_renderer()
        for i, val in enumerate(sorted_values):
            fp_val = fp_sorted[i]
            tp_val = tp_sorted[i]
            fp_str_val = " " + str(fp_val)
            tp_str_val = fp_str_val + " " + str(tp_val)
            t = plt.text(val, i, tp_str_val, color='forestgreen', va='center', fontweight='bold')
            plt.text(val, i, fp_str_val, color='crimson', va='center', fontweight='bold')
            if i == (len(sorted_values)-1):
                adjust_axes(r, t, fig, axes)
    else:
        plt.barh(range(n_classes), sorted_values, color=plot_color)
        fig = plt.gcf()
        axes = plt.gca()
        r = fig.canvas.get_renderer()
        for i, val in enumerate(sorted_values):
            str_val = " " + str(val)
            if val < 1.0:
                str_val = " {0:.2f}".format(val)
            t = plt.text(val, i, str_val, color=plot_color, va='center', fontweight='bold')
            if i == (len(sorted_values)-1):
                adjust_axes(r, t, fig, axes)
    
    fig.canvas.set_window_title(window_title)
    tick_font_size = 12
    plt.yticks(range(n_classes), sorted_keys, fontsize=tick_font_size)
    
    init_height = fig.get_figheight()
    dpi = fig.dpi
    height_pt = n_classes * (tick_font_size * 1.4)
    height_in = height_pt / dpi
    top_margin = 0.15
    bottom_margin = 0.05
    figure_height = height_in / (1 - top_margin - bottom_margin)
    if figure_height > init_height:
        fig.set_figheight(figure_height)
    
    plt.title(plot_title, fontsize=14)
    plt.xlabel(x_label, fontsize="large")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.show()


def adjust_axes(r, t, fig, axes):
    bb = t.get_window_extent(renderer=r)
    text_width_inches = bb.width / fig.dpi
    current_fig_width = fig.get_figwidth()
    new_fig_width = current_fig_width + text_width_inches
    propotion = new_fig_width / current_fig_width
    x_lim = axes.get_xlim()
    axes.set_xlim([x_lim[0], x_lim[1]*propotion])