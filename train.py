import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import PIL.Image as Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
import numpy as np
import segmentation_models_pytorch as smp


class Dataset(BaseDataset):
    def __init__(
        self,
        images_dir,
        masks_dir,
        augmentation=None,
        preprocessing=None,
    ):
        self.ids = os.listdir(images_dir)
        self.images_fps = [os.path.join(images_dir, image_id) for image_id in self.ids]
        self.masks_fps = [os.path.join(masks_dir, image_id) for image_id in self.ids]

        # convert str names to class values on masks
        # self.class_values = [self.CLASSES.index(cls.lower()) for cls in classes]
        self.class_values = [255]
        # print(self.class_values)

        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        if self.augmentation != None:
            i = i % len(self.ids)
        # read data
        image = cv2.imread(self.images_fps[i])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 扩充边界 把565*584的图片变为576*584的图片
        image = cv2.copyMakeBorder(
            image, 0, 0, 5, 6, cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )
        # 上下裁掉4个像素
        image = image[4:580, :, :]
        # #转为灰度图
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # #第三维为1
        # image = np.expand_dims(image, axis=2)

        mask = cv2.imread(self.masks_fps[i], 0)
        # 扩充边界 把565*584的图片变为576*584的图片
        mask = cv2.copyMakeBorder(mask, 0, 0, 5, 6, cv2.BORDER_CONSTANT, value=[0])
        # 上下裁掉4个像素
        mask = mask[
            4:580,
            :,
        ]
        # extract certain classes from mask (e.g. cars)
        masks = [(mask == v) for v in self.class_values]
        # print(masks)
        mask = np.stack(masks, axis=-1).astype("float")

        # apply augmentations
        if self.augmentation:
            sample = self.augmentation(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]

        # apply preprocessing
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample["image"], sample["mask"]
            # 把(3, 608, 576)大小的图片取出第二通道，变为(1，608, 576)大小
            # image = np.expand_dims(image[1], axis=0)

        return image, mask

    def __len__(self):
        if self.augmentation != None:
            return 10 * len(self.ids)
        return len(self.ids)


import albumentations as albu


def get_training_augmentation():
    train_transform = [
        albu.HorizontalFlip(p=0.5),
        # albu.PadIfNeeded(min_height=576, min_width=576, always_apply=True, border_mode=0,value=0),
        # albu.RandomCrop(height=576, width=576, always_apply=True),
        albu.ShiftScaleRotate(
            scale_limit=0.3,
            rotate_limit=(-45, 45),
            shift_limit=0.3,
            p=1,
            border_mode=0,
        ),
        albu.GaussNoise(p=0.2),
        albu.Perspective(p=0.5),
        albu.OneOf(
            [
                albu.CLAHE(p=0.5),
                albu.RandomBrightnessContrast(p=0.5),
                albu.RandomGamma(p=0.5),
            ],
            p=0.5,
        ),
        albu.OneOf(
            [
                albu.Sharpen(p=0.5),
                albu.Blur(blur_limit=3, p=0.5),
                albu.MotionBlur(blur_limit=3, p=0.5),
            ],
            p=0.5,
        ),
        albu.OneOf(
            [
                albu.RandomBrightnessContrast(p=0.5),
                albu.HueSaturationValue(p=0.5),
            ],
            p=0.5,
        ),
    ]
    return albu.Compose(train_transform)


def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        # albu.PadIfNeeded(608, 576)
        # albu.Resize(608, 576),
    ]
    return albu.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype("float32")


def get_preprocessing(mean, std):
    _transform = [
        # 根据mean和std对数据进行标准化
        albu.Normalize(mean=mean, std=std),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


# 计算std和mean
import torch


def getStat(train_data):
    print("Compute mean and variance for training data.")
    print(len(train_data))

    all_mean = np.zeros(3)
    all_std = np.zeros(3)
    for image, mask in train_data:
        # 利用cv2计算每个通道的均值和方差
        mean, std = cv2.meanStdDev(image)
        # 加入总的均值和方差中
        all_mean += mean.squeeze()
        all_std += std.squeeze()
    all_mean /= len(train_data) * 255
    all_std /= len(train_data) * 255
    return all_mean, all_std


# 主函数
def main(args):
    DATA_DIR = "./data/eyes/"
    x_train_dir = os.path.join(DATA_DIR, "train")
    y_train_dir = os.path.join(DATA_DIR, "trainannot")

    x_valid_dir = os.path.join(DATA_DIR, "val")
    y_valid_dir = os.path.join(DATA_DIR, "valannot")

    x_test_dir = os.path.join(DATA_DIR, "test")
    y_test_dir = os.path.join(DATA_DIR, "testannot")
    dataset = Dataset(x_train_dir, y_train_dir)
    # 计算mean和std
    mean, std = getStat(dataset)
    # preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS,)
    train_dataset = Dataset(
        x_train_dir,
        y_train_dir,
        augmentation=get_training_augmentation(),
        preprocessing=get_preprocessing(mean, std),
    )

    valid_dataset = Dataset(
        x_valid_dir,
        y_valid_dir,
        # augmentation=get_validation_augmentation(),
        preprocessing=get_preprocessing(mean, std),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0
    )

    valid_loader = DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0
    )

    ENCODER = args.encoder
    # ENCODER = 'timm-regnetx_002'
    ENCODER_WEIGHTS = "imagenet"
    CLASSES = ["eye"]
    ACTIVATION = (
        "sigmoid"  # could be None for logits or 'softmax2d' for multiclass segmentation
    )
    DEVICE = "cuda"
    IN_CHANNELS = args.in_channels
    # create segmentation model with pretrained encoder
    model = smp.create_model(
        args.arch,
        encoder_name=args.encoder,
        in_channels=args.in_channels,
        encoder_weights="imagenet",
        classes=1,
        activation=ACTIVATION,
    )
    # model = smp.UnetPlusPlus(
    #     encoder_name=ENCODER,
    #     encoder_weights="imagenet",
    #     in_channels=IN_CHANNELS,
    #     classes=1,
    #     activation=ACTIVATION,
    # )
    try:
        model = torch.load(f"models/best_{model.name}_{IN_CHANNELS}.pth")
        print(f"导入本地模型models/best_{model.name}_{IN_CHANNELS}成功")
    except:
        print("导入本地模型失败")

    from segmentation_models_pytorch import utils

    loss = smp.utils.losses.DiceLoss()
    metrics = [
        smp.utils.metrics.IoU(threshold=0.5),
    ]

    optimizer = torch.optim.AdamW(
        [
            dict(params=model.parameters(), lr=0.001),
        ]
    )
    train_epoch = smp.utils.train.TrainEpoch(
        model,
        loss=loss,
        metrics=metrics,
        optimizer=optimizer,
        device=DEVICE,
        verbose=True,
    )

    valid_epoch = smp.utils.train.ValidEpoch(
        model,
        loss=loss,
        metrics=metrics,
        device=DEVICE,
        verbose=True,
    )
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=3, verbose=True
    )
    # 训练
    min_loss = 100
    best_epoch = 0
    for epoch in range(0, 5000):
        print("\nEpoch: {}".format(epoch))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)

        # do something (save model, change lr, etc.)
        if min_loss > valid_logs["dice_loss"]:
            min_loss = valid_logs["dice_loss"]
            best_epoch = epoch
            torch.save(model, f"models/best_{model.name}_{IN_CHANNELS}.pth")
            print("Model saved!")

        if epoch - best_epoch == 30:
            print("Early stopping!")
            break

        lr_scheduler.step(valid_logs["dice_loss"])

    # load best saved checkpoint
    best_model = torch.load(f"models/best_{model.name}_{IN_CHANNELS}.pth")
    test_dataset = Dataset(
        x_test_dir,
        y_test_dir,
        augmentation=None,
        preprocessing=get_preprocessing(mean, std),
    )
    # 获取预测mask
    import torchvision

    try:
        os.makedirs(f"results/{model.name}/")
    except:
        pass
    num = len(test_dataset)
    image_list = []
    name_list = []
    for n in range(num):
        image, gt_mask = test_dataset[n]
        x_tensor = torch.from_numpy(image).to(DEVICE).unsqueeze(0)
        gt_mask = (gt_mask * 255).round()

        pr_mask = best_model.predict(x_tensor).squeeze(0).cpu().numpy()

        # 逆to_tensor
        image = image.transpose(1, 2, 0)
        gt_mask = gt_mask.transpose(1, 2, 0)
        pr_mask = pr_mask.transpose(1, 2, 0)

        # pr_mask上下扩充4个像素
        pr_mask = cv2.copyMakeBorder(
            pr_mask, 4, 4, 0, 0, cv2.BORDER_CONSTANT, value=[0]
        )
        # pr_mask左边裁掉5个像素，右边裁掉6个像素
        pr_mask = pr_mask[:, 5:570]
        pr_mask = pr_mask * 255

        img_save_path = os.path.join(f"results/{model.name}/", f"{n+1}.png")
        cv2.imwrite(img_save_path, pr_mask)
        image_list.append(pr_mask)
        name_list.append(f"{n+1}.png")
    # 生成csv结果
    import pandas as pd
    from PIL import Image

    def turn_to_str(image_list):
        outputs = []
        for image in image_list:
            transform = torchvision.transforms.ToTensor()
            image = Image.fromarray(image).convert("L")
            image = image.resize((512, 512), Image.Resampling.BILINEAR)
            image = transform(image)
            image[image > 0] = 1
            dots = np.where(image.flatten() == 1)[0]
            run_lengths = []
            prev = -2
            for b in dots:
                if b > prev + 1:
                    run_lengths.extend((b + 1, 0))
                run_lengths[-1] += 1
                prev = b
            output = " ".join([str(r) for r in run_lengths])
            outputs.append(output)
        return outputs

    def save_to_csv(name_list, str_list):
        df = pd.DataFrame(columns=["Id", "Predicted"])
        df["Id"] = [i.split(".")[0] for i in name_list]
        df["Predicted"] = str_list
        df.to_csv(f"results/best_{model.name}_{IN_CHANNELS}.csv", index=None)

    str_list = turn_to_str(image_list)
    save_to_csv(name_list, str_list)


if __name__ == "__main__":
    # 添加参数
    import argparse

    parser = argparse.ArgumentParser()
    # 模型名称
    parser.add_argument("-arch", type=str, default="unetplusplus", help="模型名称")
    # encoder
    parser.add_argument(
        "-encoder", type=str, default="tu-efficientnetv2_rw_t", help="backbone"
    )
    # batch_size
    parser.add_argument("-batch_size", type=int, default=1, help="batch_size")
    parser.add_argument("-in_channels", type=int, default=3, help="输入通道数")
    args = parser.parse_args()
    main(args)
