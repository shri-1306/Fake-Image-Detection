import os, datetime
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import plot_model
from PIL import Image
import matplotlib.pyplot as plt
import pydotplus
import pydot
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
base_dir = "C:/Users/Chaitra Kulkarni/OneDrive/Desktop/AIML/AIML 6 SEM/AI/Fake_Image/real_vs_fake/real-vs-fake"
train_dir = os.path.join(base_dir, 'train')
test_dir = os.path.join(base_dir, 'test')
valid_dir = os.path.join(base_dir, 'valid')
train_datagen = ImageDataGenerator( rescale = 1.0/255. )
test_datagen = ImageDataGenerator( rescale = 1.0/255. )
valid_datagen = ImageDataGenerator( rescale = 1.0/255. )
train_generator = train_datagen.flow_from_directory(train_dir, batch_size=100, class_mode='binary', target_size=(150, 150))
validation_generator = valid_datagen.flow_from_directory(valid_dir, batch_size=100, class_mode='binary', target_size=(150, 150))
test_generator = test_datagen.flow_from_directory(test_dir, batch_size=100, class_mode='binary', target_size=(150, 150))
plt.figure(figsize=(10, 10))
for i in range(9):
    img, label = next(train_generator)  # Correct way to get the next batch
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(img[0])
    if label[0] == 0.0:
        plt.title("Fake")
    else:
        plt.title("Real")
    plt.axis("off")
plt.show()
model = tf.keras.models.Sequential(
    [
     tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 3)),
     tf.keras.layers.MaxPooling2D(2,2),
     tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
     tf.keras.layers.MaxPooling2D(2,2),
     tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
     tf.keras.layers.MaxPooling2D(2,2),
     tf.keras.layers.Flatten(),
     tf.keras.layers.Dense(1064, activation='relu'),
     tf.keras.layers.Dense(2, activation='softmax')
    ]
)
tf.keras.utils.pydot = pydot
plot_model(model, to_file='model.png', show_shapes=True)
model.summary()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics = ['accuracy'])
%%time
history = model.fit(train_generator, validation_data = validation_generator, epochs = 3, validation_steps = 10, verbose=1)
test_loss, test_acc = model.evaluate(test_generator)
class_names = ['fake', 'real']
import numpy as np
from keras.preprocessing import image
# test_image = image.load_img('E:/Machine Learning Series/Datasets/archive/real_vs_fake/real-vs-fake/test/real/00461.jpg', target_size=(150, 150, 3))
test_image = image.load_img('C:/Users/Chaitra Kulkarni/OneDrive/Desktop/AIML/AIML 6 SEM/AI/Fake_Image/real_vs_fake/real-vs-fake/test/fake/0COHPC8N1I.jpg', target_size=(150, 150, 3))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = model.predict(test_image)
test_generator.class_indices
print(result)
print(
    "This image is {} with a {:.2f} percentage."
    .format(class_names[np.argmax(result)], 100 * np.max(result))
)
import numpy as np
from keras.preprocessing import image
# test_image = image.load_img('E:/Machine Learning Series/Datasets/archive/real_vs_fake/real-vs-fake/test/real/00461.jpg', target_size=(150, 150, 3))
test_image = image.load_img('C:/Users/Chaitra Kulkarni/OneDrive/Desktop/AIML/AIML 6 SEM/AI/Fake_Image/WhatsApp Image 2024-08-20 at 23.59.56_027c59c9.jpg', target_size=(150, 150, 3))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
result = model.predict(test_image)
test_generator.class_indices
print(result)





















