import sys
import os
import numpy as np
import skimage
import matplotlib.pyplot as plt
import time
import json
import random
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras import backend as K
from ESrganModel import ESRGAN
from keras.utils.data_utils import OrderedEnqueuer


#To stop showing the Tensorflow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
####################################################
################ Configure Model ###################
####################################################
#Loading the Data
load_imgs_hr = np.load('/content/Proba-V/training_set/HR_train.npy')
load_imgs_lr = np.load('/content/Proba-V/training_set/LR_train.npy')

def load_data(batch_size=1):

    batch_images = np.random.choice(load_imgs_hr.shape[0], size=batch_size)
    imgs_hr = load_imgs_hr[batch_images]
    imgs_lr = load_imgs_lr[batch_images]
    return imgs_hr, imgs_lr


log_weight_path= '/content/Proba-V/output/'
dataname='train_genz'
# instantiate modules
Esrgan = ESRGAN()


######################################################################
################ Vizualization And cPSNR computing ##################
#####################################################################


def eval_generated_images(epoch, generator, iteration,  examples=1 , dim=(1, 3),
                         save=True , plot=True):

    imgs_hr, imgs_lr = load_data(examples)
    gen_img = generator.predict(imgs_lr)
    imgs_hr = imgs_hr.astype(np.float32)
    cPSNR = Esrgan.cPSNR(imgs_hr, gen_img)
    cPSNR = cPSNR.eval(session = tf.compat.v1.Session())
    if plot:
        print('Calculated cPSNR: ', cPSNR)
        fig = plt.figure(figsize=(15, 5))

        plt.subplot(dim[0], dim[1], 1)
        plt.imshow(np.squeeze(imgs_lr[0]), interpolation='nearest')
        plt.axis('off')

        plt.subplot(dim[0], dim[1], 2)
        plt.imshow(np.squeeze(gen_img[0]), interpolation='nearest')
        plt.axis('off')

        plt.subplot(dim[0], dim[1], 3)
        plt.imshow(np.squeeze(imgs_hr[0]), interpolation='nearest')
        plt.axis('off')

        plt.tight_layout()
        if save:
            fig.savefig("output/predict_{}_{}.png".format(epoch+1, iteration+1))
            plt.close(fig)

    return cPSNR
###################################################
################ Training Esrgan  ##################
###################################################

def train(epochs=1000, batch_size=10, sample_interval=200):

    data_len = len(load_imgs_hr) # number of scenes in the dataset 1160
    disciminator_output_shape = (batch_size,24,24,1)

    # VALID / FAKE targets for discriminator
    real = np.ones(disciminator_output_shape)
    fake = np.zeros(disciminator_output_shape)
    

    #Uncomment the next statement if you want to load from a given trained weights
    Esrgan.load_weights('/content/Proba-V/weights3/train_genz_generator_X_epoch2.h5', '/content/Proba-V/weights3/train_genz_discriminator_X_epoch2.h5')
    print_losses = {"G": [], "D": [], "cPSNR" : []}
    losses_per_epoch = {"G": [], "D": [], "cPSNR" : []}
    for epoch in range(epochs):

    # Loop through epochs / iterations
        for iteration in range(data_len//batch_size):

            print('epoch: '+ str(epoch) +' iteration: ' + str(iteration))
 
            # Train discriminator   
            imgs_hr, imgs_lr = load_data(batch_size)
            generated_hr = Esrgan.generator.predict(imgs_lr)
            real_loss = Esrgan.discriminator.train_on_batch(imgs_hr, real)
            fake_loss = Esrgan.discriminator.train_on_batch(generated_hr, fake)
            discriminator_loss = 0.5 * np.add(real_loss, fake_loss)
            cal_cPSNR = eval_generated_images(epoch, Esrgan.generator, iteration, examples=1 , dim=(1, 3),save=True , plot=False)
            # Train generator
            features_hr = Esrgan.vgg.predict(Esrgan.preprocess_vgg(imgs_hr))
            generator_loss = Esrgan.esrgan.train_on_batch(imgs_lr, [real, features_hr])            

            # Save losses            
            # with open('output/epoch_{}.csv'.format(epoch), 'a') as csvfile:
            #     filewriter = csv.writer(csvfile, delimiter=',')
            #     filewriter.writerow([generator_loss[0],generator_loss[1],generator_loss[2],discriminator_loss[0],discriminator_loss[1],cal_cPSNR])
            print_losses['G'].append(generator_loss)
            print_losses['D'].append(discriminator_loss)
            print_losses['cPSNR'].append(cal_cPSNR)

            # Save the network weights
            # Save the network losses
            print(">> Saving the network weights")
            Esrgan.save_weights(os.path.join(log_weight_path, dataname), epoch)
            # Plot gen_img every sample interval
            if iteration == 0 or iteration % sample_interval == 0:
                eval_generated_images(epoch, Esrgan.generator, iteration)
                
        # Save the network model
        print(">> Saving the network model")
        Esrgan.save_model(os.path.join(log_weight_path, dataname), epoch)
        # Show the progress
         
        g_avg_loss = np.array(print_losses['G']).mean(axis=0)
        d_avg_loss = np.array(print_losses['D']).mean(axis=0)
        cPSNR_avg = np.array(print_losses['cPSNR']).mean(axis=0)
        losses_per_epoch['G'].append(g_avg_loss.tolist())
        losses_per_epoch['D'].append(d_avg_loss.tolist())
        losses_per_epoch['cPSNR'].append(float(cPSNR_avg))

        print("\nEpoch {}/{} \n>> Generator/GAN: {}\n>> Discriminator: {}".format(
            epoch, epochs,
            ", ".join(["{}={:.4f}".format(k, v) for k, v in zip(Esrgan.esrgan.metrics_names, g_avg_loss)]),
            ", ".join(["{}={:.4f}".format(k, v) for k, v in zip(Esrgan.discriminator.metrics_names, d_avg_loss)])
        ))
        
        # Save cPSNR value and Gen & Disc losses
        with open('losses_{}.json'.format(epoch), 'w') as fp:
            json.dump(losses_per_epoch, fp)

        del print_losses
        del losses_per_epoch
def test(batch_size = 4):
    gen_model = load_model('/content/Proba-V/models/train_genz_generator_model_X_epoch2.h5', custom_objects = {'cPSNR' : Esrgan.cPSNR})
    load_imgs_lr = np.load('/content/Proba-V/LR_test.npy')
    data_len = len(load_imgs_lr)
    limit = 0
    for iteration in range(data_len//batch_size):
        
        imgs_lr = load_imgs_lr[limit:limit+batch_size]
        limit = iteration+batch_size
        generated_sr = gen_model.predict(imgs_lr)
        for i in range(len(imgs_lr)):

            fig = plt.figure(figsize=(15, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(np.squeeze(imgs_lr[i]), interpolation='nearest')
            plt.axis('off')

            plt.subplot(1, 2, 2)
            plt.imshow(np.squeeze(generated_sr[i]), interpolation='nearest')
            plt.axis('off')        
            plt.tight_layout()
            fig.savefig("predictions/predict_{}_{}.png".format(i+1, iteration+1))
            plt.close(fig)

if __name__ == '__main__':

    if len(sys.argv)==1:
        print('Please insert train or test argument')
        sys.exit(0)
    if sys.argv[1] == 'train':
        epochs = 10
        train(epochs=epochs, batch_size=4, sample_interval=90)
        print("#"*100)
        print("#"*36," {} Epochs Completed! ".format(epochs),"#"*36)
        print("#"*100)
    elif sys.argv[1] == 'test':
        test(batch_size = 1)
        print("#"*100)
        print(" "*6,"#"*30,"Predictions Completed! ","#"*30)
        print("#"*100)
    else:
        print('You can use only train or test options')


    # ------------------------------------------------------------------------
    #  These parameters are set to train for 1000 epochs, with 16GB of GPU
    #  VRAM available on Colab and 25GB RAM CPU.
    #  Due to the fact that Colab crashs often,I have divided the training
    #  into mini-training where epochs = 10 
    # ------------------------------------------------------------------------


