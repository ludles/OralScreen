# -*- coding: utf-8 -*-

from glob import glob
import cv2, random, os, csv
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Dropout#, AveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
import itertools
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score


L2_WEIGHT_DECAY = 1e-4

# %%
def load_set(subset_dir, size_in, size_out, is_per_slide=False):
    
    def random_crop(img, size):
        width, height = size
        assert img.shape[0] >= height
        assert img.shape[1] >= width
        x = random.randint(0, img.shape[1] - width)
        y = random.randint(0, img.shape[0] - height)
        img = img[y:y+height, x:x+width]
        return img
    
    def read_sample(img_path):
        if size_out[-1] == 1:
            img = cv2.imread(img_path, 0)
        else:
            img = cv2.imread(img_path)
        if img.shape[:2] != size_in[:2]:
            img = cv2.resize(img, size_in[:2])
        if img.ndim == 2:
            img = np.expand_dims(img, axis=-1)
        img = random_crop(img, size_out[:2])
        return img
    
    cancer = np.array([read_sample(path) for path in glob(subset_dir + 'Cancer/*')])
    healthy = np.array([read_sample(path) for path in glob(subset_dir + 'Healthy/*')])
    X = np.concatenate((cancer, healthy), axis=0)
    
    Y = np.concatenate((np.zeros((len(cancer), 1)), np.ones((len(healthy), 1))), axis=0)
    Y = to_categorical(Y)
    
    if is_per_slide:
        dir_cancer = subset_dir + 'Cancer/'
        dir_healthy = subset_dir + 'Healthy/'
        slides_test0 = list(set([os.path.basename(patch_path)[:2] for patch_path in glob(dir_cancer + '*.jpg')]))
        slides_test1 = list(set([os.path.basename(patch_path)[:2] for patch_path in glob(dir_healthy + '*.jpg')]))
        slides_test = slides_test0 + slides_test1
        
        sizes0 = [len(glob(dir_cancer + i_slide + '*.jpg')) for i_slide in slides_test0]
        sizes1 = [len(glob(dir_healthy + i_slide + '*.jpg')) for i_slide in slides_test1]
        sizes = sizes0 + sizes1
        
        indices = [i for i in range(sum(sizes))]
        
        index_slide = {slides_test[i]:indices[sum(sizes[:i]):sum(sizes[:i+1])] for i in range(len(slides_test))}
        np.random.shuffle(indices)
        X = X[indices]
        Y = Y[indices]    
       
        return X, Y, indices, index_slide, slides_test0, slides_test1
        
    else:
        # shuffle set
        index = [i for i in range(len(X))]
        np.random.shuffle(index)
        X = X[index]
        Y = Y[index]
        return X, Y
    
    

# %%    Non-interpolation augmentation
def aug_non_inter(img):
    
    def ori(img):
        return img
    
    def fliph(img):
        return np.fliplr(img)
    
    def flipv(img):
        return np.flipud(img)
    
    def fliphv(img):
        return np.fliplr(np.flipud(img))
    
    def ori90(img):
        return np.rot90(img)
    
    def fliph90(img):
        return np.rot90(np.fliplr(img))
    
    def flipv90(img):
        return np.rot90(np.flipud(img))
    
    def fliphv90(img):
        return np.rot90(np.fliplr(np.flipud(img)))
    
    aug_functions = [ori, fliph, fliphv, fliphv, ori90, fliph90, fliphv90, fliphv90]   
    
    return random.choice(aug_functions)(img)

# %%    Create model
def build_resnet(input_shape, classes, pretrain):
    from tensorflow.keras.applications import resnet50
    # Get base model
    if pretrain == 1:
        base_model = resnet50.ResNet50(
                include_top=False, 
                weights='imagenet', 
                input_tensor=None, 
                input_shape=input_shape, 
                pooling='avg', 
                classes=classes)
    else:
        base_model = resnet50.ResNet50(
                include_top=False, 
                weights=None, 
                input_tensor=None, 
                input_shape=input_shape, 
                pooling='avg', 
                classes=classes)

    # Add final layers
    X = base_model.output
#    X = AveragePooling2D(pool_size=(2, 2), name = "avg_pool")(X)
#    X = Flatten()(X)
    X = Dropout(0.5)(X)
    predictions = Dense(
            classes, 
            kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
            bias_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
            activation='softmax', 
            name='fc')(X)
    
    # This is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

def build_densenet(input_shape, classes, pretrain):
    from tensorflow.keras.applications import densenet
    # Get base model
    if pretrain == 1:
        base_model = densenet.DenseNet201(
                include_top=False, 
                weights='imagenet', 
                input_tensor=None, 
                input_shape=input_shape, 
                pooling='avg', 
                classes=classes)
    else:
        base_model = densenet.DenseNet201(
                include_top=False, 
                weights=None, 
                input_tensor=None, 
                input_shape=input_shape, 
                pooling='avg', 
                classes=classes)

    # Add final layers
    X = base_model.output
#    X = AveragePooling2D(pool_size=(2, 2), name = "avg_pool")(X)
#    X = Flatten()(X)
    X = Dropout(0.5)(X)
    predictions = Dense(
            classes, 
            kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
            bias_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
            activation='softmax', 
            name='fc')(X)
    
    # This is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)
    return model

#%% Confusion matrix plot

def plot_confusion_matrix(cm, classes,
                          normalize=True,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def accuracy_curve(h, log_dir):
    acc, loss, val_acc, val_loss = h.history['acc'], h.history['loss'], h.history['val_acc'], h.history['val_loss']
    epoch = len(acc)
    plt.figure(figsize=(17, 5))
    plt.subplot(121)
    plt.plot(range(epoch), acc, label='Train')
    plt.plot(range(epoch), val_acc, label='Validation')
    plt.title('Accuracy over ' + str(epoch) + ' Epochs', size=15)
    plt.legend()
    plt.grid(True)
    plt.subplot(122)
    plt.plot(range(epoch), loss, label='Train')
    plt.plot(range(epoch), val_loss, label='Validation')
    plt.title('Loss over ' + str(epoch) + ' Epochs', size=15)
    plt.legend()
    plt.grid(True)
#    plt.show()
    plt.savefig(log_dir + 'learning_curve_DenseNet.png', bbox_inches='tight', transparent=False)

def evaluate(y_test, y_pred, target_names):
    y_true = np.argmax(y_test, axis=1)
    accu = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, pos_label=0, average='binary')
    recall = recall_score(y_true, y_pred, pos_label=0, average='binary')
    f1 = f1_score(y_true, y_pred, pos_label=0, average='binary')
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=target_names, digits=5)
    return {'accuracy':accu, 'precision':precision, 'recall':recall, 'f1':f1, 'cm':cm, 'report':report}

def write_results(metrics, args):
    '''
    args.architecture
    args.pretrain
    args.dataset
    args.fold
    args.index
    '''
    # log dir
    dir_res = f"./results/dataset_{args.dataset}/"
    if not os.path.exists(dir_res):
        os.makedirs(dir_res)
    csv_name = f"{args.architecture}_pre{args.pretrain}.csv"
    
    # Write results into .csv file for table
    header = ['fold', 'i_model', 'accuracy', 'precision', 'recall', 'f1', 'TP', 'FP', 'FN', 'TN', 'report']
    if not os.path.exists(dir_res + csv_name):
        with open(dir_res + csv_name, 'a+', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
    with open(dir_res + csv_name, 'a+', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([args.fold, args.index, 
                         metrics['accuracy'],
                         metrics['precision'],
                         metrics['recall'],
                         metrics['f1'],
                         metrics['cm'][0][0],
                         metrics['cm'][1][0],
                         metrics['cm'][0][1],
                         metrics['cm'][1][1],
                         metrics['report']
                         ])

def write_per_slide_results(y_test, y_pred, metrics, args, indices, index_slide, slides_test0, slides_test1):
    
    # log dir
    dir_res = f"./results/per_slide/{args.architecture}_pre{args.pretrain}/"
    if not os.path.exists(dir_res):
        os.makedirs(dir_res)
    csv_name = f"dataset_{args.dataset}.csv"

    # Write results into .csv file for further calculating
    header = ['fold', 'i_model', 'i_slide', 'slide_class', 'slide_accuracy', 'perc_cancer']
    if not os.path.exists(dir_res + csv_name):
        with open(dir_res + csv_name, 'a+', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
    
    y_true = np.argmax(y_test, axis=1)
    slides_test = slides_test0 + slides_test1
    for i_slide in slides_test:
        indices_slide = [indices.index(i) for i in index_slide[i_slide]]
        Y_true_slide = y_true[indices_slide]
        Y_pred_slide = y_pred[indices_slide]
        acc_slide = accuracy_score(Y_true_slide, Y_pred_slide)
        if i_slide in slides_test0:
            slide_class = 'Cancer'
            perc_cancer = acc_slide
        elif i_slide in slides_test1:
            slide_class = 'Healthy'
            perc_cancer = 1 - acc_slide
            
        with open(dir_res + csv_name, 'a+', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([args.fold, args.index, i_slide, slide_class, acc_slide, perc_cancer])
    
    
    