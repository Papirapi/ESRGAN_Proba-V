import os
import tensorflow
import skimage
import numpy as np
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Input, Activation, Add, Concatenate, Multiply, Flatten, add
from tensorflow.keras.layers import BatchNormalization, LeakyReLU, PReLU, Conv2D, Dense
from tensorflow.keras.layers import UpSampling2D, Lambda, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras import backend as K
from keras.callbacks import TensorBoard, ModelCheckpoint, LambdaCallback

class ESRGAN():

    """ Model Architecture is inspired from the Paper:
        https://arxiv.org/pdf/1809.00219.pdf   """

    def __init__(self,gen_lr=1e-4, dis_lr=1e-4,training_mode=True,loss_weights=[1e-3, 0.006]):

        # Input / Output shape
        self.channels = 3
        self.lr_dim = 128                 # Low resolution dimension
        self.lr_shape = (self.lr_dim, self.lr_dim , self.channels)
        self.hr_dim = 384                 # High resolution dimesion
        self.hr_shape = (self.hr_dim, self.hr_dim, self.channels)
        
        # Gan setup settings
        self.gan_loss = 'mse'
        self.dis_loss = 'binary_crossentropy'
        self.gen_lr = gen_lr
        self.dis_lr = dis_lr
        # Build & compile the generator network
        self.generator = self.build_generator()
        self.compile_generator(self.generator)

        self.loss_weights = loss_weights

        self.vgg = self.build_vgg()
        self.compile_vgg(self.vgg)
        self.discriminator = self.build_discriminator()
        self.compile_discriminator(self.discriminator)
        self.esrgan = self.build_esrgan()
        self.compile_esrgan(self.esrgan)

    def build_vgg(self):
        """
        Load pre-trained VGG weights from keras applications
        Extract features to be used in loss function from last conv layer, see architecture at:
        https://github.com/keras-team/keras/blob/master/keras/applications/vgg19.py
        """

        # Input image to extract features from
        img = Input(shape=(self.hr_shape[0],self.hr_shape[1],3))

        # Get the vgg network. Extract features from last conv layer
        vgg = VGG19(weights="imagenet",include_top=False,input_shape=(self.hr_shape[0],self.hr_shape[1],3))
        vgg.outputs = [vgg.layers[20].output]

        # Create model and compile
        model = Model(inputs=img, outputs=vgg(img))
        model.trainable = False
        return model    

    def preprocess_vgg(self, x):
        """Take a HR image [-1, 1], convert to [0, 255], then to input for VGG network"""
        if isinstance(x, np.ndarray):
            return preprocess_input((x+1)*127.5)
        else:            
            return Lambda(lambda x: preprocess_input(tensorflow.add(x, 1) * 127.5))(x)
            

    def build_generator(self, ):
        """
        Build the generator network according to description in the paper.
        :return: the compiled model
        """

        def dense_block(input):
            x1 = Conv2D(64, kernel_size=3, strides=1, padding='same')(input)
            x1 = LeakyReLU(0.2)(x1)
            x1 = Concatenate()([input, x1])

            x2 = Conv2D(64, kernel_size=3, strides=1, padding='same')(x1)
            x2 = LeakyReLU(0.2)(x2)
            x2 = Concatenate()([input, x1, x2])

            x3 = Conv2D(64, kernel_size=3, strides=1, padding='same')(x2)
            x3 = LeakyReLU(0.2)(x3)
            x3 = Concatenate()([input, x1, x2, x3])

            x4 = Conv2D(64, kernel_size=3, strides=1, padding='same')(x3)
            x4 = LeakyReLU(0.2)(x4)
            x4 = Concatenate()([input, x1, x2, x3, x4])

            x5 = Conv2D(64, kernel_size=3, strides=1, padding='same')(x4)
            x5 = Lambda(lambda x: x * 0.2)(x5)
            x = Add()([x5, input])
            return x

        def RRDB(input):
            x = dense_block(input)
            x = dense_block(x)
            x = dense_block(x)
            x = Lambda(lambda x: x * 0.2)(x)
            out = Add()([x, input])
            return out


        def up_sample_block(model, kernel_size=3, filters=256, strides=1, size=3, skip=None):


            if skip is not None:
                model = Concatenate(axis=-1)([model, skip])

            model = Conv2D(filters, kernel_size, strides, padding = "same", kernel_initializer='he_normal',name='upSampleConv2d_' + str(size))(model)
            model = PReLU(shared_axes=[1, 2], name='upSamplePReLU_' + str(size))(model)
            model = UpSampling2D(size = size)(model)
  
            return model

        # Input low resolution image
        lr_input = Input(shape= self.lr_shape)

        # Pre-residual
        x_start = Conv2D(64, kernel_size=3, strides=1, padding='same')(lr_input)
        x_start = LeakyReLU(0.2)(x_start)

        # Residual-in-Residual Dense Block
        x = RRDB(x_start)

        # Post-residual block
        x = Conv2D(64, kernel_size=3, strides=1, padding='same')(x)
        x = Lambda(lambda x: x * 0.2)(x)
        x = Add()([x, x_start])

        # Upsampling depending on factor
        x = up_sample_block(x, size=3)
        x = Conv2D(64, kernel_size=3, strides=1, padding='same')(x)
        x = LeakyReLU(0.2)(x)
        hr_output = Conv2D(3, kernel_size=3, strides=1, padding='same', activation='tanh')(x)

        # Create model and compile
        model = Model(inputs=lr_input, outputs=hr_output)
        model.summary()

        return model


    def build_discriminator(self, filters=64):
        """
        Build the discriminator network according to description in the paper.
        """

        def conv2d_block(input, filters, strides=1, bn=True):
            d = Conv2D(filters, kernel_size=3, strides=strides, padding='same')(input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        # Input high resolution image
        img = Input(shape=self.hr_shape)
        x = conv2d_block(img, filters, bn=False)
        x = conv2d_block(x, filters, strides=2)
        x = conv2d_block(x, filters * 2)
        x = conv2d_block(x, filters * 2, strides=2)
        x = conv2d_block(x, filters * 4)
        x = conv2d_block(x, filters * 4, strides=2)
        x = conv2d_block(x, filters * 8)
        x = conv2d_block(x, filters * 8, strides=2)
        x = Dense(filters * 16)(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Dropout(0.4)(x)
        x = Dense(1)(x)
 
        # Create model and compile
        model = Model(inputs=img, outputs=x)

        return model


    def build_esrgan(self):
        """Create the combined ESRGAN network"""

        # Input LR images
        img_lr = Input(self.lr_shape)

        # Create a high resolution image from the low resolution one
        generated_hr = self.generator(img_lr)
        generated_features = self.vgg(
            self.preprocess_vgg(generated_hr)
        )

        # In the combined model we only train the generator
        self.discriminator.trainable = False

        # Determine whether the generator HR images are OK
        generated_check = self.discriminator(generated_hr)
        
        # Create sensible names for outputs in logs
        generated_features = Lambda(lambda x: x, name='Content')(generated_features)
        generated_check = Lambda(lambda x: x, name='Adversarial')(generated_check)

        # Create model and compile
        # Using binary_crossentropy with reversed label, to get proper loss, see:
        # https://danieltakeshi.github.io/2017/03/05/understanding-generative-adversarial-networks/
        model = Model(inputs=img_lr, outputs=[generated_check, generated_features])        
        return model


    def compile_vgg(self, model):
        """Compile the generator with appropriate optimizer"""
        model.compile(
            loss='mse',
            optimizer=Adam(self.gen_lr, 0.9),
            metrics=['accuracy']
        )


    def compile_generator(self, model):
        """Compile the generator with appropriate optimizer"""
        model.compile(
            loss=self.gan_loss,
            optimizer=Adam(self.gen_lr, 0.9),
            metrics=['mse', self.cPSNR]
        )


    def compile_discriminator(self, model):
        """Compile the generator with appropriate optimizer"""
        model.compile(
            loss=self.dis_loss,
            optimizer=Adam(self.dis_lr, 0.9),
            metrics=['accuracy']
        )


    def compile_esrgan(self, model):
        """Compile the GAN with appropriate optimizer"""
        model.compile(
            loss=[self.dis_loss, self.gan_loss],
            loss_weights=self.loss_weights,
            optimizer=Adam(self.gen_lr, 0.9)
        )


    def cMSE(self, y_true, y_pred):
        # apply the quality mask
        obs = tensorflow.equal(y_true, 0.05)
        clr = tensorflow.math.logical_not(obs)
        _hr = tensorflow.boolean_mask(y_true, clr )
        _sr = tensorflow.boolean_mask(y_pred, clr )

        # calculate the bias in brightness b
        pixel_diff = _hr - _sr
        b = K.mean(pixel_diff)

        # calculate the corrected clear mean-square error
        pixel_diff -= b
        cMse = K.mean(pixel_diff * pixel_diff)

        return cMse 


    def cPSNR(self, y_true, y_pred):
        """
        PSNR is Peek Signal to Noise Ratio, see https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio
        The equation is:
        PSNR = 20 * log10(MAX_I) - 10 * log10(MSE)
        
        Since input is scaled from -1 to 1, MAX_I = 1, and thus 20 * log10(1) = 0. Only the last part of the equation is therefore neccesary.
        """
        return -10.0 * K.log(self.cMSE(y_true,y_pred)) / K.log(10.0)


    def save_weights(self, filepath, e=None):
        """Save the generator and discriminator networks"""
        self.generator.save_weights("{}_generator_X_epoch{}.h5".format(filepath,  e))
        self.discriminator.save_weights("{}_discriminator_X_epoch{}.h5".format(filepath, e))    


    def load_weights(self, generator_weights=None, discriminator_weights=None, **kwargs):
        """Load the generator and discriminator networks"""

        if generator_weights:
            self.generator.load_weights(generator_weights, **kwargs)
        if discriminator_weights:
            self.discriminator.load_weights(discriminator_weights, **kwargs)


    def save_model(self, filepath, e=None):
        """Save the generator and discriminator model"""
       
        self.generator.save("{}_generator_model_X_epoch{}.h5".format(filepath,  e))
        #self.discriminator.save("{}_discriminator_model_X_epoch{}.h5".format(filepath, e))  
