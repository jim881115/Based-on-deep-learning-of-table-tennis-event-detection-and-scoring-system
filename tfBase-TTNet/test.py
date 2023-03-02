from model import EventNet
from LoadBatches import generator

import tensorflow as tf

images_path = '/home/tt/TableTennis/TrackNet/Dataset/'
testing_data_csv_path = "./dataset_path/test_data_file_path.csv"

def test(weights_path):
    model = EventNet()
    # model = keras.models.load_model('eventmodel')
    # model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.001), metrics=['accuracy'])

    model.load_weights(weights_path)
    # model.set_weights('eventmodel.h5')
    test = generator(testing_data_csv_path, 1)
    
    correct_bounce = 0
    correct_net = 0
    correct_empty = 0
    wrong_cont = 0
    set_cont = 0
    while True:
        if set_cont == 380: break
        test_infor, test_label = next(test)
        test_label = test_label[0]
        set_cont += 1
        
        out = model.predict(test_infor)[0]

        if out[0] > 0.9 and test_label[0]:
            correct_net += 1
        elif out[1] > 0.9 and test_label[1]:
            correct_bounce += 1
        elif (out[0] < 0.9 and out[1] < 0.9) and (test_label[0] + test_label[1]) == 0:
            correct_empty += 1
        else:
            print('-----event detected failed-----')
            print('test set number:', set_cont)
            print('test prob:', out)
            print('target prob:', test_label)
            print('=========================================\n')
            wrong_cont += 1
    print('---test end---')
    print('correct counting(bounce, net, empty):', correct_bounce, correct_net, correct_empty)
    print('wrong counting:', wrong_cont)
    print('accuracy:', 1.0 - wrong_cont / (correct_bounce + correct_net + correct_empty + wrong_cont))


if __name__ == '__main__':
    # create_csv(poss, testing_data_csv_path)
    # shuffle_csv(testing_data_csv_path)
    with tf.device('/gpu:1'):
        model_name = 'train_model_weight2.h5'
        test('./weights/' + model_name)
