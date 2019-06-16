import encode_image as ei
import SceneDesc
import time as time
import pyttsx3
import sys
from keras.models import save_model, load_model


def process_caption(sd, caption):
    caption_split = caption.split()
    processed_caption = caption_split[1:]
    try:
        end_index = processed_caption.index('<end>')
        processed_caption = processed_caption[:end_index]
    except:
        pass
    return " ".join([word for word in processed_caption])


def generate_captions(sd, model, encoded_images, beam_size):
    first_word = [sd.word_index['<start>']]
    prob_level = 0.0
    capt_seq = [[first_word, prob_level]]
    max_cap_length = sd.max_length
    while len(capt_seq[0][0]) < max_cap_length:
        temp_capt_seq = []
        for caption_id in capt_seq:
            iter_capt = sequence.pad_sequences(
                [caption_id[0]], max_cap_length, padding='post')
            next_word_prob = model.predict(
                [np.asarray([encoded_images]), np.asarray(iter_capt)])[0]
            next_word_ids = np.argsort(next_word_prob)[-beam_size:]
            for word_id in next_word_ids:
                new_iter_capt, new_iter_prob = caption_id[0][:], caption_id[1]
                new_iter_capt.append(word_id)
                new_iter_prob += next_word_prob[word_id]
                temp_capt_seq.append([new_iter_capt, new_iter_prob])
        capt_seq = temp_capt_seq
        capt_seq.sort(key=lambda l: l[1])
        capt_seq = capt_seq[-beam_size:]
    best_caption = capt_seq[len(capt_seq)-1][0]
    best_caption = " ".join([sd.index_word[index] for index in best_caption])
    image_desc = process_caption(sd, best_caption)
    return image_desc


def text(img):
    t1 = time.time()
    encode = ei.model_gen()
    weight = 'models/weights.h5'
    sd = SceneDesc.scenedesc()
    model = sd.create_model(ret_model=True)
    model.load_weights(weight)
    image_path = img
    encoded_images = ei.encodings(encode, image_path)

    image_captions = generate_captions(
        sd, model, encoded_images, beam_size=3)

    print(image_captions)


if __name__ == '__main__':
    image = str(sys.argv[1])
    image = "test/"+image
    text(image)
