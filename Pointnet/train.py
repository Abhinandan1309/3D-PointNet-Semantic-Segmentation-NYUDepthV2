import torch
import torch.optim as optim
import numpy as np
import time
from custom_loss import PointNetSegLoss
from data_loader_sampling_remap import get_dataloader
from pointnet_arch import PointNetSegHead
from sklearn.metrics import matthews_corrcoef


def train():

    # Setup training parameters
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 4
    EPOCHS = 50
    NUM_TRAIN_POINTS = 10000
    NUM_CLASSES = 50  # change based on the number of classes in your dataset
    LEARNING_RATE = 0.001

    # Paths to training and validation data directories
    train_data_dir = "C:\\Users\\abhin\\Desktop\\3D_Semantic_Segmentation\\output_point_clouds\\train"
    valid_data_dir = "C:\\Users\\abhin\\Desktop\\3D_Semantic_Segmentation\\output_point_clouds\\val"

    # # Create DataLoader objects
    # train_dataloader = get_dataloader(train_data_dir, batch_size=BATCH_SIZE, shuffle=True)
    # valid_dataloader = get_dataloader(valid_data_dir, batch_size=BATCH_SIZE, shuffle=False)

    train_dataloader = get_dataloader(train_data_dir, batch_size=32, shuffle=True, downsample='random', num_samples=10000)
    valid_dataloader = get_dataloader(valid_data_dir, batch_size=32, shuffle=False, downsample='random', num_samples=10000)

    # for batch in train_dataloader:
    #     points = batch['points']  # [BATCH_SIZE, NUM_POINTS, 3]
    #     labels = batch['labels']  # [BATCH_SIZE, NUM_POINTS]
        
    #     print(f"Points shape: {points.shape}")  # Should be [BATCH_SIZE, NUM_POINTS, 3]
    #     print(f"Labels shape: {labels.shape}")  # Should be [BATCH_SIZE, NUM_POINTS]

    # Model and optimizer
    seg_model = PointNetSegHead(m=NUM_CLASSES).to(DEVICE)
    optimizer = optim.Adam(seg_model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # Loss function
    criterion = PointNetSegLoss(alpha=None, gamma=2, dice=True)

    # Metrics tracking
    train_loss, train_accuracy, train_mcc, train_iou = [], [], [], []
    valid_loss, valid_accuracy, valid_mcc, valid_iou = [], [], [], []

    def compute_iou(targets, preds):
        intersection = np.logical_and(targets, preds)
        union = np.logical_or(targets, preds)
        iou_score = np.sum(intersection) / np.sum(union)
        return iou_score

    # Training loop
    best_iou = 0
    for epoch in range(1, EPOCHS + 1):
        seg_model.train()
        _train_loss, _train_accuracy, _train_mcc, _train_iou = [], [], [], []

        for i, batch  in enumerate(train_dataloader):
            points = batch['points'].transpose(2, 1).float().to(DEVICE)
            targets = batch['labels'].squeeze().to(DEVICE)

            optimizer.zero_grad()
            preds, _, A = seg_model(points)

            pred_choice = torch.softmax(preds, dim=2).argmax(dim=2)
            loss = criterion(preds, targets, pred_choice)

            loss.backward()
            optimizer.step()

            # correct = pred_choice.eq(targets.data).cpu().sum()
            # accuracy = correct / float(BATCH_SIZE * NUM_TRAIN_POINTS)

            correct = pred_choice.eq(targets.data).cpu().sum()
            total_points = targets.numel()  # Total points in the current batch
            accuracy = correct / total_points  # Compute accuracy based on actual number of points

            # MCC and IoU calculations
            preds_np = pred_choice.cpu().numpy().flatten()
            targets_np = targets.cpu().numpy().flatten()
            mcc_score = matthews_corrcoef(targets_np, preds_np)
            iou_score = compute_iou(targets_np, preds_np)

            _train_loss.append(loss.item())
            _train_accuracy.append(accuracy.item())
            _train_mcc.append(mcc_score)
            _train_iou.append(iou_score)

            if i % 100 == 0:
                print(f'[{epoch}: {i}/{len(train_dataloader)}] '
                    f'train loss: {loss.item():.4f}, accuracy: {accuracy:.4f}, mcc: {mcc_score:.4f}, iou: {iou_score:.4f}')

        train_loss.append(np.mean(_train_loss))
        train_accuracy.append(np.mean(_train_accuracy))
        train_mcc.append(np.mean(_train_mcc))
        train_iou.append(np.mean(_train_iou))

        print(f'Epoch: {epoch} - Train Loss: {train_loss[-1]:.4f}, Train Accuracy: {train_accuracy[-1]:.4f}, '
            f'Train MCC: {train_mcc[-1]:.4f}, Train IoU: {train_iou[-1]:.4f}')

        # Step scheduler per epoch
        scheduler.step()

        time.sleep(2)

        # Validation loop
        with torch.no_grad():
            seg_model.eval()
            _valid_loss, _valid_accuracy, _valid_mcc, _valid_iou = [], [], [], []

            for i, batch in enumerate(valid_dataloader):
                points = batch['points'].transpose(2, 1).float().to(DEVICE)
                targets = batch['labels'].squeeze().to(DEVICE)


                preds, _, A = seg_model(points)
                pred_choice = torch.softmax(preds, dim=2).argmax(dim=2)
                loss = criterion(preds, targets, pred_choice)

                # correct = pred_choice.eq(targets.data).cpu().sum()
                # accuracy = correct / float(BATCH_SIZE * NUM_TRAIN_POINTS)

                correct = pred_choice.eq(targets.data).cpu().sum()
                total_points = targets.numel()  # Total points in the batch
                accuracy = correct / total_points  # Compute accuracy

                preds_np = pred_choice.cpu().numpy().flatten()
                targets_np = targets.cpu().numpy().flatten()
                mcc_score = matthews_corrcoef(targets_np, preds_np)
                iou_score = compute_iou(targets_np, preds_np)

                _valid_loss.append(loss.item())
                _valid_accuracy.append(accuracy.item())
                _valid_mcc.append(mcc_score)
                _valid_iou.append(iou_score)

                if i % 100 == 0:
                    print(f'[{epoch}: {i}/{len(valid_dataloader)}] valid loss: {loss.item():.4f}, '
                        f'accuracy: {accuracy:.4f}, mcc: {mcc_score:.4f}, iou: {iou_score:.4f}')

            valid_loss.append(np.mean(_valid_loss))
            valid_accuracy.append(np.mean(_valid_accuracy))
            valid_mcc.append(np.mean(_valid_mcc))
            valid_iou.append(np.mean(_valid_iou))

            print(f'Epoch: {epoch} - Valid Loss: {valid_loss[-1]:.4f}, Valid Accuracy: {valid_accuracy[-1]:.4f}, '
                f'Valid MCC: {valid_mcc[-1]:.4f}, Valid IoU: {valid_iou[-1]:.4f}')

            # Save best model based on IoU
            if valid_iou[-1] > best_iou:
                best_iou = valid_iou[-1]
                torch.save(seg_model.state_dict(), f'saved_model_epoch_{epoch}.pth')

        time.sleep(2)

if __name__ == "__main__":
    train()