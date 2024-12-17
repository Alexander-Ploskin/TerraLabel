import torch
from tqdm import tqdm


class SegmenterModeltrainer:
    def __init__(
        self,
        model,
        device,
        writer,
        train_dataloader,
        eval_dataloader,
        metric
    ):
        model.to(device)
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=0.00006)
        self.device = device
        self.writer = writer
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.metric = metric

    def train(self, file_path, n_epochs=200):
        train_loss_iter = []
        train_loss_epoch = []
        eval_iou = []
        eval_acc = []
        eval_loss = []
        self.model.train()
        best_val_loss = float('inf')

        for epoch in tqdm(range(n_epochs)):  # loop over the dataset multiple times
            print("Epoch:", epoch)
            curr_epoch_loss = []
            curr_epoch_eval_loss = []
            
            for idx, batch in enumerate(self.train_dataloader):
                pixel_values = batch["pixel_values"].to(self.device)
                labels = batch["labels"].to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(pixel_values=pixel_values, labels=labels)
                loss, logits = outputs.loss, outputs.logits

                loss.backward()
                self.optimizer.step()
                
                train_loss_iter.append(loss.item())
                curr_epoch_loss.append(loss.item())
                self.writer.add_scalar("Train/loss_step", train_loss_iter[-1], idx + epoch * len(self.train_dataloader))
                self.writer.add_scalar("Train/epoch", epoch + 1, idx + epoch * len(self.train_dataloader))
            
            loss_val = sum(curr_epoch_loss) / len(curr_epoch_loss)
            train_loss_epoch.append(loss_val)   
            self.writer.add_scalar("Train/loss_epoch", train_loss_epoch[-1], epoch + 1)
            if loss_val < best_val_loss:
                self.model.save_pretrained(file_path)
                best_val_loss = loss_val

            with torch.no_grad():
                for batch in self.eval_dataloader:
                    pixel_values = batch["pixel_values"].to(self.device)
                    labels = batch["labels"].to(self.device)
                    outputs = self.model(pixel_values=pixel_values, labels=labels)
                    loss, logits = outputs.loss, outputs.logits
                    curr_epoch_eval_loss.append(outputs.loss.item())
                    upsampled_logits = torch.nn.functional.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)
                    predicted = upsampled_logits.argmax(dim=1)
                    self.metric.add_batch(predictions=predicted.detach().cpu().numpy(), references=labels.detach().cpu().numpy())
                    
                metrics = self.metric.compute(
                    num_labels=6, 
                    ignore_index=255,
                    reduce_labels=False, # we've already reduced the labels before)
                )
                eval_iou.append( metrics["mean_iou"])
                eval_acc.append(metrics["mean_accuracy"])
                eval_loss.append(sum(curr_epoch_eval_loss) / len(self.eval_dataloader))
                
                self.writer.add_scalar("Eval/loss",eval_loss[-1], epoch + 1)
                self.writer.add_scalar("Eval/Accuracy", metrics["mean_accuracy"], epoch + 1)
                self.writer.add_scalar("Eval/IoU", metrics["mean_iou"], epoch + 1)
                
                print("Mean_iou:", metrics["mean_iou"])
                print("Loss:", train_loss_epoch[-1])
                print("Mean accuracy:", metrics["mean_accuracy"])
    
        return self.model
