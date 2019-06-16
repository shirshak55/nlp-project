from vgg16 import VGG16
import numpy as np
from keras.preprocessing import image
from imagenet_utils import preprocess_input
import six.moves.cPickle as pickle
import progressbar


def preprocessing():
    image_captions = open(
        "captions/Flickr8k.token.txt").read().split('\n')
    caption = {}
    for i in range(len(image_captions)-1):
        id_capt = image_captions[i].split("\t")
        # to rip off the #0,#1,#2,#3,#4 from the tokens file
        id_capt[0] = id_capt[0][:len(id_capt[0])-2]
        try:
            caption[id_capt[0]].append(id_capt[1])
        except:
            caption[id_capt[0]] = [id_capt[1]]
    train_imgs_id = open(
        "captions/Flickr_8k.trainImages.txt").read().split('\n')[:-1]

    train_imgs_captions = open("captions/trainimgs.txt", 'w')
    for img_id in train_imgs_id:
        for captions in caption[img_id]:
            desc = "<start> "+captions+" <end>"
            train_imgs_captions.write(img_id+"\t"+desc+"\n")
            train_imgs_captions.flush()
    train_imgs_captions.close()

    test_imgs_id = open(
        "captions/Flickr_8k.testImages.txt").read().split('\n')[:-1]

    test_imgs_captions = open("captions/testimgs.txt", 'w')
    for img_id in test_imgs_id:
        for captions in caption[img_id]:
            desc = "<start> "+captions+" <end>"
            test_imgs_captions.write(img_id+"\t"+desc+"\n")
            test_imgs_captions.flush()
    test_imgs_captions.close()


def model_gen():
    model = VGG16(weights='imagenet', include_top=True,
                  input_shape=(224, 224, 3))
    return model


def encodings(model, path):
    processed_img = image.load_img(path, target_size=(224, 224))
    x = image.img_to_array(processed_img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    image_final = np.asarray(x)
    prediction = model.predict(image_final)
    prediction = np.reshape(prediction, prediction.shape[1])

    return prediction


def encode_image():
    model = VGG16(weights='imagenet', include_top=True,
                  input_shape=(224, 224, 3))
    image_encodings = {}

    train_imgs_id = open(
        "captions/Flickr_8k.trainImages.txt").read().split('\n')[:-1]
    print(len(train_imgs_id))
    test_imgs_id = open(
        "captions/Flickr_8k.testImages.txt").read().split('\n')[:-1]
    images = []
    images.extend(train_imgs_id)
    images.extend(test_imgs_id)
    print(len(images))
    bar = progressbar.ProgressBar(maxval=len(images),
                                  widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
    bar.start()
    counter = 1
    print("Encoding images")

    for img in images:
        path = "images/"+str(img)
        image_encodings[img] = encodings(model, path)
        bar.update(counter)
        counter += 1

    bar.finish()
    with open("image_encodings.p", "wb") as pickle_f:
        pickle.dump(image_encodings, pickle_f)
    print("Encodings dumped into image_encodings.p")


if __name__ == "__main__":
    encode_image()
