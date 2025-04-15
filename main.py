import argparse
import training_evaluating
import datasets

# 기존 유넷에 BCELogitLoss쓰니까 loss 값은 그대로 잘 떨어지긴 하네
# efficientnet + diceloss쓰니까 Loss가 0.998이러는데 이유가 있나? --> dice값 계산해보니까 0.002이렇다. 앞선 loss값도 정확한 값이었고 학습이 그냥 안되는 것이네

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="appendix_segmentation") # 프로그램 설명
    parser.add_argument('--batch_size', default=16, type=int, help="batch size") # help는 인수설명
    parser.add_argument('--epoch', default=12, type=int, help='training epoch')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--l2', default=0.0005, type=float, help='weight decay')
    parser.add_argument('--model_name', default='effi_unet', type=str, help='model name')
    parser.add_argument('--train', default='train', type=str, help='train and eval')
    args = parser.parse_args()

    # 데이터 불러오기

    train_dataloader, validation_dataloader = datasets.dataloader(args.batch_size)

    print('Completed loading your datasets.')

    # 모델 불러오기 및 학습하기
    learning = training_evaluating.SupervisedLearning(train_dataloader, validation_dataloader, args.model_name)

    if args.train == 'train':
        learning.train(args.epoch, args.lr, args.l2)
    else:
        print("test")