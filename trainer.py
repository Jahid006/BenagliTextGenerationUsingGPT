from tqdm import tqdm
import torch


def train(
    model, optimizer, scheduler,
    criterion, device, train_dataset, test_dataset,
    summary_writter=None, epochs=10, logging=None,
    saving_step=100, steps_per_epoch=None,
    model_saving_dir='artifact'
    
):
    print("Saving Steps: ", saving_step)
    print("Steps Per Epoch: ", steps_per_epoch)
    steps = 0
    for epoch in tqdm(range(epochs)):
        total_train_loss, total_batch_size = 0., 0
        current_steps = 0
        for train_data in tqdm(train_dataset):
            loss = train_batch(
                model, train_data, optimizer,
                criterion, device
            )

            batch_size = train_data['input_text'].size(0)
            total_train_loss += loss
            total_batch_size += batch_size

            current_steps += 1
            steps += 1

            if steps % saving_step == 0 or current_steps == steps_per_epoch:

                total_val_loss, total_val_batch_size = 0., 0
                for it, val_data in tqdm(enumerate(test_dataset)):
                    if it == 100:
                        break
                    loss = eval_batch(model, val_data, criterion, device)

                    batch_size = val_data['input_text'].size(0)
                    total_val_loss += loss
                    total_val_batch_size += batch_size

                avg_train_loss = round(total_train_loss / total_batch_size, 6)
                avg_val_loss = round(total_val_loss / total_val_batch_size, 6)

                scheduler.step(avg_val_loss)

                summary_writter.add_scalar("train_loss", avg_train_loss, steps)
                summary_writter.add_scalar("val_loss", avg_val_loss, steps)

                logging.debug(
                    f"Steps: {steps}\tEpoch: {epoch}\tLR: {scheduler._last_lr}\t"
                    + f"Train Loss: {avg_train_loss}\tValidation Loss: {avg_val_loss}"
                )

                save_model_path = (
                    f"{model_saving_dir}/"
                    + f'{steps:09}_{epoch:06}_loss_{avg_train_loss}_vloss_{avg_val_loss}.pt'
                )

                torch.save({
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                }, save_model_path)

    return model


def train_batch(model, data, optimizer, criterion, device):

    model.train()

    input_texts = data['input_text'].to(device)
    target_text = data['target_text'].to(device)

    logits = model(input_texts)

    loss = criterion(logits.view(-1, logits.shape[-1]), target_text.view(-1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
    optimizer.step()

    loss = loss.item()

    return loss


def eval_batch(model, data, criterion, device):
    model.eval()
    with torch.no_grad():
        input_texts = data['input_text'].to(device)
        target_text = data['target_text'].to(device)

        logits = model(input_texts)
        loss = criterion(logits.view(-1, logits.shape[-1]), target_text.view(-1))
        loss = loss.item()

    return loss
