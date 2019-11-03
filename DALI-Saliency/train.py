import os.path as osp
import shutil
from datetime import datetime
from pprint import pprint

import numpy as np
import torch
import torch.backends.cudnn as torchcudnn
from PIL import Image
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from torch.nn import BCELoss
from torch.optim import Adam, SGD, lr_scheduler
from torchvision import transforms
from tqdm import tqdm

from Loss.ConsistencyEnhanecdLoss import CELV1
from Loss.DiceLoss import DiceLoss
from Utils.datasets import DataLoaderX, ImageFolder
from Utils.datasets_dali import ImagePipeline
from Utils.epoch_config import arg_config, path_config, proj_root
from Utils.metric import cal_maxf, cal_pr_mae_meanf
from Utils.misc import AvgMeter, check_mkdir, make_log

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torchcudnn.benchmark = True
# 使用确定性卷积 用torchcudnn.deterministic = True
# 这样调用的CuDNN的卷积操作就是每次一样的了
torchcudnn.deterministic = True
torchcudnn.enabled = True


class Trainer():
    def __init__(self, args, path):
        super(Trainer, self).__init__()
        self.args = args
        self.path = path
        self.dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to_pil = transforms.ToPILImage()
        self.best_results = {"maxf": 0, "meanf": 0, "mae": 0}

        # 提前创建好记录文件，避免自动创建的时候触发文件创建事件
        check_mkdir(self.path["pth_log"])
        make_log(self.path["val_log"], f"=== val_log {datetime.now()} ===")
        make_log(self.path["tr_log"], f"=== tr_log {datetime.now()} ===")

        # 提前创建好存储预测结果和存放模型以及tensorboard的文件夹
        check_mkdir(self.path["save"])
        check_mkdir(self.path["pth"])
        check_mkdir(self.path["tb"])
        self.save_path = self.path["save"]

        # 依赖与前面属性的属性
        self.pth_path = self.path["final_net"]
        self.tr_loader, self.te_loader, self.val_loader = self.make_loader()
        if not self.args['use_backbone']:
            self.net = self.args[self.args["NET"]]["net"]().to(self.dev)
        else:
            self.net = self.args[self.args["NET"]]["net"](
                self.args[self.args["backbone"]]).to(self.dev)
        # 输出并记录模型运算量和参数量
        # model_msg = get_FLOPs_Params(self.net, self.args["input_size"], mode="print&return")
        # make_log(self.path["val_log"], f"=== model info ==={self.net}"
        #                                f"\n{model_msg}\n=== val record ===\n")
        pprint(self.args)

        # 损失相关
        self.sod_crit = BCELoss(reduction=self.args['reduction']).to(self.dev)
        self.dice_loss = DiceLoss().to(self.dev)
        # self.sm_loss = SmeasureLoss().to(self.dev)

        # 训练相关
        self.start_epoch = self.args["start_epoch"]
        self.end_epoch = self.args["end_epoch"]

        # 计算正常衰减的部分的迭代次数, 一个周期batch数量为len(self.tr_loader)
        self.epoch_num = self.end_epoch - self.start_epoch
        if self.tr_loader._size % self.args['batch_size'] == 0:
            self.niter_per_epoch = self.tr_loader._size // self.args['batch_size']

        else:
            self.niter_per_epoch = self.tr_loader._size // self.args['batch_size'] + 1
        self.iter_num = self.epoch_num * self.niter_per_epoch

        print(f" ==>> 总共迭代次数: {self.iter_num} <<== ")
        self.opti = self.make_optim()
        self.sche = self.make_scheduler()

    def train(self):
        for curr_epoch in range(self.start_epoch, self.end_epoch):
            torch.cuda.empty_cache()  # 定期清空模型
            train_loss_record = AvgMeter()
            for train_batch_id, train_data in enumerate(self.tr_loader):
                curr_iter = curr_epoch * self.niter_per_epoch + train_batch_id

                # 这里要和dataset.py中的设置对应好
                train_inputs = train_data[0]['images']
                train_labels = train_data[0]['masks']

                self.opti.zero_grad()
                *_, otr = self.net(train_inputs)

                loss_list = []
                loss_item_list = []
                sod_out = self.sod_crit(otr, train_labels)
                loss_list.append(sod_out)
                loss_item_list.append(f"{sod_out.item():.5f}")
                dice_out = self.dice_loss(otr, train_labels)
                loss_list.append(dice_out)
                loss_item_list.append(f"{dice_out.item():.5f}")
                train_loss = sum(loss_list)

                train_loss.backward()
                self.opti.step()

                if self.args["sche_usebatch"]:
                    if self.args["lr_type"] == "poly":
                        self.sche.step(curr_iter + 1)
                    elif self.args["lr_type"] == "cos":
                        self.sche.step()
                    else:
                        raise NotImplementedError

                # 仅在累计的时候使用item()获取数据
                train_iter_loss = train_loss.item()
                train_batch_size = train_inputs.size(0)
                train_loss_record.update(train_iter_loss, train_batch_size)

                # 记录每一次迭代的数据
                if (self.args["print_freq"] > 0 and (curr_iter + 1) % self.args["print_freq"] == 0):
                    log = (f"[I:{curr_iter}/{self.iter_num}][E:{curr_epoch}:{self.end_epoch}]>"
                           f"[{self.args[self.args['NET']]['exp_name']}]"
                           f"[Lr:{self.opti.param_groups[0]['lr']:.7f}]"
                           f"[Avg:{train_loss_record.avg:.5f}|Cur:{train_iter_loss:.5f}|"
                           f"{loss_item_list}]")
                    print(log)
                    make_log(self.path["tr_log"], log)

            # 根据周期修改学习率
            if not self.args["sche_usebatch"]:
                if self.args["lr_type"] == "poly":
                    self.sche.step(curr_epoch + 1)
                elif self.args["lr_type"] == "cos":
                    self.sche.step()
                elif self.args["lr_type"] == "step":
                    self.sche.step()
                else:
                    raise NotImplementedError

            # 每个周期都进行保存测试
            torch.save(self.net.state_dict(), self.path["final_net"])
            if self.args["val_freq"] > 0 and ((curr_epoch) % self.args["val_freq"] == 0):
                self.pth_path = self.path["final_net"]
                if self.args["test_as_val"]:
                    # 使用测试集来验证
                    results = self.test(mode="test", save_pre=False,
                                        data_path=self.args['te_data_path'])
                else:
                    results = self.test(mode="val", save_pre=False,
                                        data_path=self.args['val_data_path'])
                if results["maxf"] > self.best_results["maxf"]:
                    self.best_results = results
                    torch.save(self.net.state_dict(), self.path["best_net"])
                    msg = f"epoch:{curr_epoch}=>{results} is best, so far..."
                else:
                    msg = f"epoch:{curr_epoch}=>{results}"
                print(f" ==>> 验证结果：{msg} <<== ")
                make_log(self.path["val_log"], msg)
                self.net.train()

        # 进行最终的测试，首先输出验证结果
        print(f" ==>> 训练结束，最好验证结果：{self.best_results} <<== ")

        if self.args["val_freq"] > 0 and (not self.args["test_as_val"]):
            self.pth_path = self.path["best_net"]
        elif self.args["val_freq"] > 0 and self.args["test_as_val"]:
            self.pth_path = self.path["final_net"]
        elif not self.args["val_freq"] > 0:
            self.pth_path = self.path["final_net"]
        for data_name, data_path in self.args['te_data_list'].items():
            if data_name == 'hkuis':
                prefix = ('.png', '.png')
            else:
                prefix = self.args['prefix']
            print(f" ==>> 使用测试集{data_name}测试 <<== ")
            self.te_loader = DataLoaderX(ImageFolder(data_path,
                                                     mode="test",
                                                     in_size=self.args["input_size"],
                                                     prefix=prefix),
                                         batch_size=self.args["batch_size"],
                                         num_workers=self.args["num_workers"],
                                         shuffle=False, drop_last=False, pin_memory=True)
            results = self.test(mode="test", save_pre=False, data_path=data_path)
            msg = (f" ==>> 在{data_name}:'{data_path}'测试集上结果：{results} <<== ")
            print(msg)
            make_log(self.path["val_log"], msg)

    def test(self, mode, save_pre, data_path):
        print(f" ==>> 导入模型...{self.pth_path} <<== ")
        try:
            self.net.load_state_dict(torch.load(self.pth_path))
        except FileNotFoundError:
            print("请指定模型")
            exit()
        self.net.eval()

        if mode == "test":
            loader = self.te_loader
        elif mode == "val":
            loader = self.val_loader
        else:
            raise NotImplementedError

        gt_path = osp.join(data_path, "Mask")

        pres = [AvgMeter() for _ in range(256)]
        recs = [AvgMeter() for _ in range(256)]
        meanfs = AvgMeter()
        maes = AvgMeter()
        tqdm_iter = tqdm(enumerate(loader), total=len(loader), leave=False)
        for test_batch_id, test_data in tqdm_iter:
            tqdm_iter.set_description(f"{self.args[self.args['NET']]['exp_name']}:"
                                      f"te=>{test_batch_id + 1}")
            in_imgs, in_names = test_data
            in_imgs = in_imgs.to(self.dev)
            with torch.no_grad():
                *_, outputs = self.net(in_imgs)
            outputs_np = outputs.cpu().detach()

            for item_id, out_item in enumerate(outputs_np):
                gimg_path = osp.join(gt_path, in_names[item_id] + ".png")
                gt_img = Image.open(gimg_path).convert("L")
                out_img = self.to_pil(out_item).resize(gt_img.size)
                gt_img = np.array(gt_img)

                if save_pre:
                    oimg_path = osp.join(self.save_path, in_names[item_id] + ".png")
                    out_img.save(oimg_path)

                out_img = np.array(out_img)
                ps, rs, mae, meanf = cal_pr_mae_meanf(out_img, gt_img)
                for pidx, pdata in enumerate(zip(ps, rs)):
                    p, r = pdata
                    pres[pidx].update(p)
                    recs[pidx].update(r)
                maes.update(mae)
                meanfs.update(meanf)
        maxf = cal_maxf([pre.avg for pre in pres], [rec.avg for rec in recs])
        results = {"maxf": maxf, "meanf": meanfs.avg, "mae": maes.avg}
        return results

    def make_scheduler(self):
        total_num = self.iter_num if self.args['sche_usebatch'] else self.epoch_num
        if self.args["lr_type"] == "poly":
            lamb = lambda curr: pow((1 - float(curr) / total_num), self.args["lr_decay"])
            scheduler = lr_scheduler.LambdaLR(self.opti, lr_lambda=lamb)
        elif self.args["lr_type"] == "cos":
            scheduler = lr_scheduler.CosineAnnealingLR(self.opti, T_max=total_num - 1,
                                                       eta_min=4e-08)
        elif self.args["lr_type"] == "step":
            scheduler = lr_scheduler.StepLR(self.opti, step_size=self.args['steplr_epoch'],
                                            gamma=self.args['steplr_gamma'])
        else:
            raise NotImplementedError
        return scheduler

    def make_loader(self):
        print(f" ==>> 使用训练集{self.args['tr_data_path']}训练 <<== ")
        train_pipe = ImagePipeline(imageset_dir=self.args['tr_data_path'],
                                   image_size=self.args["input_size"],
                                   random_shuffle=True,
                                   batch_size=self.args["batch_size"])
        train_loader = DALIGenericIterator(pipelines=train_pipe,
                                           output_map=["images", "masks"],
                                           size=train_pipe.epoch_size(),
                                           auto_reset=True,
                                           fill_last_batch=False,
                                           last_batch_padded=False)

        if self.args['val_data_path'] != None:
            print(f" ==>> 使用验证集{self.args['val_data_path']}验证 <<== ")
            val_set = ImageFolder(self.args['val_data_path'],
                                  mode="test",
                                  in_size=self.args["input_size"],
                                  prefix=self.args['prefix'])
            val_loader = DataLoaderX(val_set,
                                     batch_size=self.args["batch_size"],
                                     num_workers=self.args["num_workers"],
                                     shuffle=False, drop_last=False, pin_memory=True)
        else:
            print(" ==>> 不使用验证集验证 <<== ")
            val_loader = None

        if self.args['te_data_path'] != None:
            print(f" ==>> 使用测试集{self.args['te_data_path']}测试 <<== ")
            test_set = ImageFolder(self.args['te_data_path'],
                                   mode="test",
                                   in_size=self.args["input_size"],
                                   prefix=self.args['prefix'])
            test_loader = DataLoaderX(test_set,
                                      batch_size=self.args["batch_size"],
                                      num_workers=self.args["num_workers"],
                                      shuffle=False, drop_last=False, pin_memory=True)
        else:
            print(f" ==>> 不使用测试集测试 <<== ")
            test_loader = None
        return train_loader, test_loader, val_loader

    def make_optim(self):
        if self.args["optim"] == "sgd_trick":
            # https://github.com/implus/PytorchInsight/blob/master/classification/imagenet_tricks.py
            params = [
                {
                    "params": [p for name, p in self.net.named_parameters()
                               if ("bias" in name or "bn" in name)],
                    "weight_decay": 0,
                },
                {
                    "params": [p for name, p in self.net.named_parameters()
                               if ("bias" not in name and "bn" not in name)]
                },
            ]
            optimizer = SGD(params,
                            lr=self.args["lr"],
                            momentum=self.args["momentum"],
                            weight_decay=self.args["weight_decay"])
        elif self.args["optim"] == "sgd_r3":
            params = [
                # 不对bias参数执行weight decay操作，weight decay主要的作用就是通过对网络
                # 层的参数（包括weight和bias）做约束（L2正则化会使得网络层的参数更加平滑）达
                # 到减少模型过拟合的效果。
                {
                    "params": [param for name, param in self.net.named_parameters()
                               if name[-4:] == "bias"],
                    "lr": 2 * self.args["lr"],
                },
                {
                    "params": [param for name, param in self.net.named_parameters()
                               if name[-4:] != "bias"],
                    "lr": self.args["lr"],
                    "weight_decay": self.args["weight_decay"],
                },
            ]
            optimizer = SGD(params,
                            momentum=self.args["momentum"])
        elif self.args["optim"] == "sgd_all":
            optimizer = SGD(self.net.parameters(),
                            lr=self.args["lr"],
                            weight_decay=self.args["weight_decay"],
                            momentum=self.args["momentum"])
        elif self.args["optim"] == "adam":
            optimizer = Adam(self.net.parameters(),
                             lr=self.args["lr"],
                             betas=(0.9, 0.999),
                             eps=1e-8,
                             weight_decay=self.args["weight_decay"])
        else:
            raise NotImplementedError
        print("optimizer = ", optimizer)
        return optimizer


if __name__ == "__main__":
    # 保存备份数据 ###########################################################
    print(f" ===========>> {datetime.now()}: 初始化开始 <<=========== ")
    init_start = datetime.now()
    trainer = Trainer(arg_config, path_config)
    print(f" ==>> 初始化完毕，用时：{datetime.now() - init_start} <<== ")

    shutil.copy(f"{proj_root}/Utils/epoch_config.py", path_config["cfg_log"])
    shutil.copy(__file__, path_config["trainer_log"])

    # 训练模型 ###############################################################
    print(f" ===========>> {datetime.now()}: 开始训练 <<=========== ")
    trainer.train()
    print(f" ===========>> {datetime.now()}: 结束训练 <<=========== ")
