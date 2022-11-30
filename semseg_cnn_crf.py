'''
Ganga Nakarmi
West Virginia University
Coding assistance from Llew Williams

This code was used in collecting data for this research paper:
   "A crowdsourced approach to documenting users’ preferences for landscape attributes in the proposed Appalachian Geopark Region in West Virginia"

This code started with file obtained from github by Dr D. Buscombe. Original header info is below.  
'''
## written by Dr Daniel Buscombe
## Northern Arizona University
## daniel.buscombe@nau.edu

## from: https://github.com/dbuscombe-usgs/dl_landscapes_paper
## If you find these codes/data useful, please cite:
## Buscombe and Ritchie (2018) "Landscape classification with deep neural networks", submitted to Geosciences June 2018
## https://eartharxiv.org/5mx3c

#general
from __future__ import division
from importlib.resources import Resource
from joblib import Parallel, delayed, cpu_count
import os, time
from glob import glob
from imageio import imread

#numerical
import tensorflow as tf
#tf.disable_eager_execution()
import tensorflow as tf2
import numpy as np
from scipy.io import savemat, loadmat
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import create_pairwise_bilateral, unary_from_labels, unary_from_softmax

#csv
import csv

#plots
import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as colors
from scipy.misc import imresize

from tile_utils import *


#supress tf warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import warnings
warnings.filterwarnings("ignore")

# suppress divide and invalid warnings
np.seterr(divide='ignore')
np.seterr(invalid='ignore')

##==============================================================
# New code here

location_of_output_file = 'OUTPUT_FILE_LOCATION' #file location can be added here or input from prompt later

#resr output file name - CRF
esr_output_file_name = 'OUTPUT_FILE_NAME_CRF.csv'  

#Lcorig output file name - CNN
Lcorig_output_file_name = 'OUTPUT_FILE_NAME_CNN.csv'  

#output_folder_for_png
output_folder_for_png = 'OUTPUT_PNG_FOLDER'

#output_folder_for_mat
output_folder_for_mat = 'OUTPUT_MAT_FOLDER'


#
def write_csv_file(name,classes,first_time,output_file_name):
    output_file_location_and_name = os.path.join(location_of_output_file,output_file_name)
    if first_time==False:
        out_file = open(output_file_location_and_name, 'a',encoding="utf-8",newline='')
    else:
        out_file = open(output_file_location_and_name, 'w',encoding="utf-8",newline='')
    if classes.any():
        writer = csv.writer(out_file)
        counter = 0 
        list_to_write = []
        zero = 0
        one = 0
        two = 0
        three = 0
        four = 0
        five = 0 
        six = 0 
        if first_time==True:
            writer.writerow(['name','zero','one','two','three','four','five','six','total'])
            first_time = False
        for i in range(len(classes)):            
            for j in range(len(classes[i])):
                counter+=1
                class_value = classes[i][j]
                if class_value == 0:
                    zero += 1
                elif class_value == 1:
                    one += 1
                elif class_value == 2:
                    two += 1
                elif class_value == 3:
                    three += 1
                elif class_value == 4:
                    four += 1
                elif class_value == 5:
                    five += 1
                elif class_value == 6:
                    six += 1
        list_to_write.append(name)
        list_to_write.append(str(zero))
        list_to_write.append(str(one))
        list_to_write.append(str(two))
        list_to_write.append(str(three))
        list_to_write.append(str(four))
        list_to_write.append(str(five))
        list_to_write.append(str(six))
        list_to_write.append(str(counter))
        writer.writerow(list_to_write)
            #if counter > 100:
            #    break
    out_file.close()


###-----------------------

# =========================================================

def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph
# =========================================================
def getCP(tmp, graph):
  
   input_name = "import/Placeholder" #input" 
   output_name = "import/final_result" 

   input_operation = graph.get_operation_by_name(input_name);
   output_operation = graph.get_operation_by_name(output_name);

   with tf.Session(graph=graph) as sess:
      results = sess.run(output_operation.outputs[0],
                      {input_operation.outputs[0]: np.expand_dims(tmp, axis=0)})
   results = np.squeeze(results)

   # Sort to show labels of first prediction in order of confidence
   top_k = results.argsort()[-len(results):][::-1]

   return top_k[0], results[top_k[0]], results[top_k] #, np.std(tmp[:,:,0])

# =========================================================
def getCP_shift(result, Zx, Zy, k, shift, graph):

   input_name = "import/Placeholder" #input" 
   output_name = "import/final_result" 

   input_operation = graph.get_operation_by_name(input_name);
   output_operation = graph.get_operation_by_name(output_name);

   results = []
   with tf.Session(graph=graph) as sess:
      results.append(sess.run(output_operation.outputs[0],
                      {input_operation.outputs[0]: np.expand_dims(result[Zx[k], Zy[k], :], axis=0)}))
      try:
         results.append(sess.run(output_operation.outputs[0],
                      {input_operation.outputs[0]: np.expand_dims(result[Zx[k]+shift, Zy[k], :], axis=0)}))
      except:
         pass
      try:
         results.append(sess.run(output_operation.outputs[0],
                      {input_operation.outputs[0]: np.expand_dims(result[Zx[k]-shift, Zy[k], :], axis=0)}))
      except:
         pass
      try:
         results.append(sess.run(output_operation.outputs[0],
                      {input_operation.outputs[0]: np.expand_dims(result[Zx[k], Zy[k]+shift, :], axis=0)}))
      except:
         pass
      try:
         results.append(sess.run(output_operation.outputs[0],
                      {input_operation.outputs[0]: np.expand_dims(result[Zx[k], Zy[k]-shift, :], axis=0)}))
      except:
         pass
      try:
         results.append(sess.run(output_operation.outputs[0],
                      {input_operation.outputs[0]: np.expand_dims(result[Zx[k]-shift, Zy[k]-shift, :], axis=0)}))
      except:
         pass
      try:
         results.append(sess.run(output_operation.outputs[0],
                      {input_operation.outputs[0]: np.expand_dims(result[Zx[k]+shift, Zy[k]-shift, :], axis=0)}))
      except:
         pass
      try:
         results.append(sess.run(output_operation.outputs[0],
                      {input_operation.outputs[0]: np.expand_dims(result[Zx[k]-shift, Zy[k]+shift, :], axis=0)}))
      except:
         pass
      try:
         results.append(sess.run(output_operation.outputs[0],
                      {input_operation.outputs[0]: np.expand_dims(result[Zx[k]+shift, Zy[k]+shift, :], axis=0)}))
      except:
         pass

   if len(results)>1:
      results = np.squeeze(results)
      #results = np.mean(results, axis=0)
      w=np.ones(len(results))
      w[0]=2
      results = np.average(results, axis=0, weights=w) 
   else:
      results = np.squeeze(results)

   # Sort to show labels of first prediction in order of confidence
   top_k = results.argsort()[-len(results):][::-1]

   return top_k[0], results[top_k[0]], results[top_k] ##, results[top_k[0]] - results[top_k[1]]
  
   
# =========================================================
def norm_im(image_path):
   input_mean = 0 #128
   input_std = 255 #128

   input_name = "file_reader"
   output_name = "normalized"
   img = imread(image_path)
   nx, ny, nz = np.shape(img)

   theta = np.std(img).astype('int')

   file_reader = tf.compat.v1.read_file(image_path, input_name)
   image_reader = tf.image.decode_jpeg(file_reader, channels = 3,
                                        name='jpeg_reader')
   float_caster = tf.cast(image_reader, tf.float32)

   dims_expander = tf.expand_dims(float_caster, 0);
   normalized = tf.divide(tf.subtract(dims_expander, [input_mean]), [input_std])
   sess = tf.compat.v1.Session()
   return np.squeeze(sess.run(normalized))
      

# =========================================================
def getCRF(image, Lc, theta, n_iter, label_lines, compat_spat=12, compat_col=40, scale=5, prob=0.5):

#        n_iters: number of iterations of MAP inference.
#        sxy_gaussian: standard deviations for the location component
#            of the colour-independent term.
#        compat_gaussian: label compatibilities for the colour-independent
#            term (can be a number, a 1D array, or a 2D array).
#        kernel_gaussian: kernel precision matrix for the colour-independent
#            term (can take values CONST_KERNEL, DIAG_KERNEL, or FULL_KERNEL).
#        normalisation_gaussian: normalisation for the colour-independent term
#            (possible values are NO_NORMALIZATION, NORMALIZE_BEFORE, NORMALIZE_AFTER, NORMALIZE_SYMMETRIC).
#        sxy_bilateral: standard deviations for the location component of the colour-dependent term.
#        compat_bilateral: label compatibilities for the colour-dependent
#            term (can be a number, a 1D array, or a 2D array).
#        srgb_bilateral: standard deviations for the colour component
#            of the colour-dependent term.
#        kernel_bilateral: kernel precision matrix for the colour-dependent term
#            (can take values CONST_KERNEL, DIAG_KERNEL, or FULL_KERNEL).
#        normalisation_bilateral: normalisation for the colour-dependent term
#            (possible values are NO_NORMALIZATION, NORMALIZE_BEFORE, NORMALIZE_AFTER, NORMALIZE_SYMMETRIC).

      H = image.shape[0]
      W = image.shape[1]

      d = dcrf.DenseCRF2D(H, W, len(label_lines)+1)
      U = unary_from_labels(Lc.astype('int'), len(label_lines)+1, gt_prob= prob)

      d.setUnaryEnergy(U)

      del U

      # This potential penalizes small pieces of segmentation that are
      # spatially isolated -- enforces more spatially consistent segmentations
      # This adds the color-independent term, features are the locations only.
      # sxy = The scaling factors per dimension.
      d.addPairwiseGaussian(sxy=(theta,theta), compat=compat_spat, kernel=dcrf.DIAG_KERNEL, #compat=6
                      normalization=dcrf.NORMALIZE_SYMMETRIC)

      # sdims = The scaling factors per dimension.
      # schan = The scaling factors per channel in the image.
      # This creates the color-dependent features and then add them to the CRF
      feats = create_pairwise_bilateral(sdims=(theta, theta), schan=(scale, scale, scale), #11,11,11
                                  img=image, chdim=2)

      del image

      d.addPairwiseEnergy(feats, compat=compat_col, #20
                    kernel=dcrf.DIAG_KERNEL,
                    normalization=dcrf.NORMALIZE_SYMMETRIC)
      del feats

      Q = d.inference(n_iter)

      #preds = np.array(Q, dtype=np.float32).reshape(
      #  (len(label_lines)+1, nx, ny)).transpose(1, 2, 0)
      #preds = np.expand_dims(preds, 0)
      #preds = np.squeeze(preds)

      return np.argmax(Q, axis=0).reshape((H, W)) #, preds#, p, R, d.klDivergence(Q),


#==============================================================
def run_inference_on_images(image_path, classifier_file, decim, tile, fct, n_iter, labels, compat_spat, compat_col, scale, winprop, prob, theta, prob_thres, cmap1):

   #=============================================
   # Image pre-processing
   #=============================================

   img = imread(image_path)
   nx, ny, nz = np.shape(img)
   print("image_path", image_path)

   result = norm_im(image_path)

   nxo, nyo, nzo = np.shape(result)

   ## pad image so it is divisible by N windows with no remainder 
   result = np.vstack((np.hstack((result,np.fliplr(result))), np.flipud(np.hstack((result,np.fliplr(result))))))
   result = result[:nxo+np.mod(nxo,tile),:nyo+np.mod(nyo,tile), :] 

   nx, ny, nz = np.shape(result)

   gridy, gridx = np.meshgrid(np.arange(ny), np.arange(nx))

   Zx,_ = sliding_window(gridx, (tile,tile), (tile,tile))
   Zy,_ = sliding_window(gridy, (tile,tile), (tile,tile))


   if decim>1:
      Zx = Zx[::decim]
      Zy = Zy[::decim]


   print('CNN ... ')
   
   graph = load_graph(classifier_file)

   overlap = 50

   w1 = []
   if overlap>0:
      shift = int(tile*overlap/100)
      for i in range(len(Zx)):
         w1.append(getCP_shift(result, Zx, Zy, i, shift, graph))
   else:
      Z,ind = sliding_window(result, (tile,tile,3), (tile, tile,3))
      for i in range(len(Z)):
         w1.append(getCP(Z[i], graph))



   ##C=most likely, P=prob, PP=all probs
   C, P, PP = zip(*w1)

   del w1

   C = np.asarray(C)
   P = np.asarray(P)
   PP = np.asarray(PP)

   C = C+1 #add 1 so all labels are >=1
   PP = np.squeeze(PP)

   ## create images with classes and probabilities
   Lc = np.zeros((nx, ny))
   Lp = np.zeros((nx, ny))


   mn = np.int(tile-(tile*winprop)) #tile/2 - tile/4)
   mx = np.int(tile+(tile*winprop)) #tile/2 + tile/4)

   for k in range(len(Zx)): 
      Lc[Zx[k][mn:mx,mn:mx], Zy[k][mn:mx,mn:mx]] = Lc[Zx[k][mn:mx,mn:mx], Zy[k][mn:mx,mn:mx]]+C[k] 
      Lp[Zx[k][mn:mx,mn:mx], Zy[k][mn:mx,mn:mx]] = Lp[Zx[k][mn:mx,mn:mx], Zy[k][mn:mx,mn:mx]]+P[k] 


   Lpp = np.zeros((nx, ny, np.shape(PP)[0]))

   for k in range(len(Zx)): 
      for l in range(np.shape(PP)[0]):
         Lpp[Zx[k], Zy[k], 0] = Lpp[Zx[k], Zy[k], l]+PP[k][0]

   Lpp = Lpp[:nxo, :nyo, :]      
   Lp = Lp[:nxo, :nyo]      
   Lc = Lc[:nxo, :nyo]

   nxo, nyo, nz = np.shape(img)

   Lcorig = Lc.copy()
   Lcorig[Lp < prob_thres] = np.nan

   Lc[np.isnan(Lcorig)] = 0

   name, ext = os.path.splitext(image_path)
   name = name.split(os.sep)[-1]

   imgr = imresize(img, fct)
   Lcr = np.round(imresize(Lc, fct, interp='nearest')/255 * np.max(Lc))

   #=============================================
   # Conditional Random Field post-processing
   #=============================================
   print('CRF ... ')
   res = getCRF(imgr, Lcr, theta, n_iter, labels, compat_spat, compat_col, scale, prob)
   print("res shape", res.shape)
   print("res type", type(res))
   del imgr
   resr = np.round(imresize(res, 1/fct, interp='nearest')/255 * np.max(res))
   del res

   print('Plotting and saving ... ')
   #print(name)
   print("Lcorig shape", Lcorig.shape)
   print("Lcorig type", type(Lcorig))
  
   print("resr shape", resr.shape)
   print("resr type", type(resr))
   #=============================================
   fig = plt.figure()
   fig.subplots_adjust(wspace=0.4)
   ax1 = fig.add_subplot(131)
   ax1.get_xaxis().set_visible(False)
   ax1.get_yaxis().set_visible(False)

   im = ax1.imshow(img)
   plt.title('a) Input', loc='left', fontsize=6)

   ax1 = fig.add_subplot(132)
   ax1.get_xaxis().set_visible(False)
   ax1.get_yaxis().set_visible(False)

   im = ax1.imshow(img)
   plt.title('b) CNN prediction', loc='left', fontsize=6)
   im2 = ax1.imshow(Lcorig-1, cmap=cmap1, alpha=0.5, vmin=0, vmax=len(labels))
   divider = make_axes_locatable(ax1)
   cax = divider.append_axes("right", size="5%")
   cb=plt.colorbar(im2, cax=cax)
   cb.set_ticks(np.arange(len(labels)+1.5)) 
   cb.ax.set_yticklabels(labels)
   cb.ax.tick_params(labelsize=4) 
   plt.axis('tight')


   ax1 = fig.add_subplot(133)
   ax1.get_xaxis().set_visible(False)
   ax1.get_yaxis().set_visible(False)

   
   im = ax1.imshow(img)
   plt.title('c) CRF prediction', loc='left', fontsize=6)
   im2 = ax1.imshow(resr, cmap=cmap1, alpha=0.5, vmin=0, vmax=len(labels))
   divider = make_axes_locatable(ax1)
   cax = divider.append_axes("right", size="5%")
   cb=plt.colorbar(im2, cax=cax)
   cb.set_ticks(np.arange(len(labels)+1.5)) 
   cb.ax.set_yticklabels(labels)
   cb.ax.tick_params(labelsize=4)
   

   outputfile = output_folder_for_png+name+'DESIRED_LABEL'+str(tile)+'.png'
   print(outputfile)
   plt.savefig(output_folder_for_png+name+'DESIRED_LABEL'+str(tile)+'.png', dpi=600, bbox_inches='tight')
   del fig; plt.close()

   # accumulating metrics in csv file for statistics
   try:
      classes_to_write_to_csv = resr.astype('int')
      write_csv_file(name,classes_to_write_to_csv,first_time,esr_output_file_name)
   except:
      print("csv write failed")
   #can I get cnn too
   try:
      Lcorig_minus_one = Lcorig-1
      classes_to_write_to_csv = Lcorig_minus_one.astype('int')
      write_csv_file(name,classes_to_write_to_csv,first_time,Lcorig_output_file_name)
   except:
      print("csv cnn write failed")

  
#==============================================================

#==============================================================
if __name__ == '__main__':

   #tile = 96
   tile = 224
   winprop = 1.0
   direc = 'DIRECTORY_HOLDING_IMAGES'
   prob_thres = 0.5
   n_iter = 20
   compat_col = 100
   theta = 60 #100
   scale = 1
   #decim = 2
   decim = 1
   fct =  0.25 
   compat_spat = 5 #2
   class_file = 'LABELS.txt'
   prob = 0.5
   # to create a new file
   first_time = False
   #=============================================

   ## Loads label file, strips off carriage return
   labels = [line.rstrip() for line 
	in tf.compat.v1.gfile.FastGFile(class_file)]
                #in tf.gfile.GFile(class_file)]

   code= {}
   for label in labels:
      code[label] = [i for i, x in enumerate([x.startswith(label) for x in labels]) if x].pop()


   classifier_file = 'geopark_mobilenetv2_224_new_blue_2022-01-27.pb'

   #new colors 
   cmap1 = list(labels)   
   tmp = [i for i, x in enumerate([x.startswith('anthro') for x in labels]) if x].pop()
   #cmap1[tmp] = '#A1887F'
   cmap1[tmp] = '#CC00CC'

   tmp = [i for i, x in enumerate([x.startswith('forest') for x in labels]) if x].pop()
   cmap1[tmp] = '#2E7D32'

   tmp = [i for i, x in enumerate([x.startswith('grass') for x in labels]) if x].pop()
   cmap1[tmp] = '#DCE775'

   tmp = [i for i, x in enumerate([x.startswith('road') for x in labels]) if x].pop()
   cmap1[tmp] = '#0000cc'

   tmp = [i for i, x in enumerate([x.startswith('rock') for x in labels]) if x].pop()
   cmap1[tmp] = '#FF0000'

   tmp = [i for i, x in enumerate([x.startswith('sky') for x in labels]) if x].pop()
   #cmap1[tmp] = '#D4E6F1'
   cmap1[tmp] = '#00FFFF'

   tmp = [i for i, x in enumerate([x.startswith('water') for x in labels]) if x].pop()
   cmap1[tmp] = '#1565C0'


   max_proc = 4

   cmap1 = colors.ListedColormap(cmap1)
   images = sorted(glob(direc+os.sep+'**/*.jpg',recursive=True))
   print("images ", images)
   one_for_testing = 1

   w = Parallel(n_jobs=np.min((max_proc,1)), verbose=10)(delayed(run_inference_on_images)(image_path, classifier_file, decim, tile, fct, n_iter, labels, compat_spat, compat_col, scale, winprop, prob, theta, prob_thres, cmap1) for image_path in images)
  


