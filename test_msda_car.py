import os
net = "vgg16"
part = "test_t"
output_dir = "./output_car/car_2"
dataset = "mskda_car"
path = "./save_model/train_adap_car_30epoch"

start_epoch = 8
max_epochs = 15
os.environ["CUDA_VISIBLE_DEVICES"]="1"

for i in range(start_epoch, max_epochs + 1):
    model_dir = path + "/mskda_car_{}.pth".format(i)
    command = "python eval/test_msda.py --cuda --gc --lc --vis --part {} --net {} --dataset {} --model_dir {} --output_dir {} --num_epoch {}".format(
        part, net, dataset, model_dir, output_dir, i)
    os.system(command)