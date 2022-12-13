import os
net = "vgg16"
part = "test_t"
output_dir = "/mnt/work2/phat/CODING/Base1Non_conLoss_MSDAOD/result/sw_base1none_confidence"
dataset = "mskda_bdd"
path = "/mnt/work2/phat/CODING/Base1Non_conLoss_MSDAOD/save_model/train_adap_sw_base1none_confidence"

start_epoch = 23
max_epochs = 27
os.environ["CUDA_VISIBLE_DEVICES"]="1"

for i in range(start_epoch, max_epochs + 1):
    model_dir = path + "/mskda_bdd_{}.pth".format(i)
    command = "python eval/test_msda_base1non_confidence.py --cuda --gc --lc --vis --part {} --net {} --dataset {} --model_dir {} --output_dir {} --num_epoch {}".format(
        part, net, dataset, model_dir, output_dir, i)
    os.system(command)