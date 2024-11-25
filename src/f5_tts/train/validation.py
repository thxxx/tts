def validate(model, valid_loader, is_make_samples=False, is_whisper=False):
    model.eval()
    valid_loss=0
    for idx, batch in enumerate(valid_loader):
        mel = batch["mel"]
        mel_lengths = batch["mel_lengths"]
        scripts = batch["script"]
        caption = batch["caption"]
        
        mel_spec = mel.permute(0, 2, 1).to(device)
        
        loss, cond, pred = model(
            mel_spec, text=text_inputs, lens=mel_lengths, noise_scheduler=self.noise_scheduler
        )
        valid_loss += loss

    return valid_loss.cpu().detach().item()/len(valid_loader)