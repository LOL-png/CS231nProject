"""
Fixed Training Loop for BFloat16 Compatibility

This is a fixed version of the training loop from cell 23 in train_full_featured.ipynb
that properly handles BFloat16 to Float32 conversion for DINOv2.
"""

# This code should replace the training loop in cell 23 of the notebook

# Main training loop with BFloat16 compatibility fix
print("Starting training with GPU-cached data...")
print("Expected: 5-20x faster training with 100GB+ GPU memory usage")
print("✓ BFloat16 compatibility fix applied for DINOv2\n")

# Import the fix
from fixes.bfloat16_compatibility import fix_batch_dtype_for_dinov2

# Get the criterion for epoch updates
criterion = manipulation_trainer.criterion

for epoch in range(config.training.num_epochs):
    # Update loss function epoch for dynamic weighting
    criterion.set_epoch(epoch)
    
    # Training epoch
    train_metrics = {'loss': 0, 'hand_mpjpe': 0, 'samples': 0}
    model.train()
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.training.num_epochs} [Train]")
    for batch_idx, batch in enumerate(pbar):
        # Move batch to device (should already be on GPU with cached dataset)
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                for k, v in batch.items()}
        
        # CRITICAL FIX: Convert BFloat16 images to Float32 for DINOv2
        if config.training.use_bf16 and 'image' in batch:
            if batch['image'].dtype == torch.bfloat16:
                batch['image'] = batch['image'].float()
        
        # Forward pass with mixed precision
        manipulation_trainer.optimizer.zero_grad()
        
        if trainer.use_amp and trainer.scaler is not None:
            # Float16 AMP
            with torch.amp.autocast('cuda'):
                outputs = model(batch)
                losses = criterion(outputs, batch)
                loss = losses['total'] if isinstance(losses, dict) else losses
            
            trainer.scaler.scale(loss).backward()
            trainer.scaler.unscale_(manipulation_trainer.optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip)
            trainer.scaler.step(manipulation_trainer.optimizer)
            trainer.scaler.update()
        else:
            # BFloat16 or no AMP
            with torch.amp.autocast('cuda', dtype=torch.bfloat16) if config.training.use_bf16 else torch.no_grad():
                outputs = model(batch)
                losses = criterion(outputs, batch)
                loss = losses['total'] if isinstance(losses, dict) else losses
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.grad_clip)
            manipulation_trainer.optimizer.step()
        
        # Extract metrics
        loss_value = loss.item()
        mpjpe_value = 0
        if 'hand_joints' in outputs and 'hand_joints_3d' in batch:
            with torch.no_grad():
                mpjpe = torch.norm(outputs['hand_joints'] - batch['hand_joints_3d'], dim=-1).mean()
                mpjpe_value = mpjpe.item()
        
        # Update metrics
        batch_size = batch['image'].shape[0]
        train_metrics['samples'] += batch_size
        train_metrics['loss'] += loss_value * batch_size
        train_metrics['hand_mpjpe'] += mpjpe_value * batch_size
        
        # Log gradient norms
        if config.debug.log_gradient_norms and batch_idx % 10 == 0:
            grad_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    grad_norm += p.grad.data.norm(2).item() ** 2
            grad_norm = grad_norm ** 0.5
            history['gradient_norms'].append(grad_norm)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss_value:.4f}',
            'mpjpe': f'{mpjpe_value:.1f}mm',
            'gpu_mem': f'{torch.cuda.memory_allocated() / 1e9:.1f}GB'
        })
        
        # Log to wandb
        if config.training.use_wandb and batch_idx % config.training.log_freq == 0:
            wandb.log({
                'train/loss': loss_value,
                'train/hand_mpjpe': mpjpe_value,
                'train/lr': manipulation_trainer.optimizer.param_groups[0]['lr'],
                'train/grad_norm': history['gradient_norms'][-1] if history['gradient_norms'] else 0,
                'system/gpu_memory_gb': torch.cuda.memory_allocated() / 1e9,
                'system/gpu_utilization': torch.cuda.utilization()
            })
    
    # Average training metrics
    train_metrics['loss'] /= train_metrics['samples']
    train_metrics['hand_mpjpe'] /= train_metrics['samples']
    
    # Validation
    val_metrics = {'loss': 0, 'hand_mpjpe': 0, 'samples': 0}
    model.eval()
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{config.training.num_epochs} [Val]"):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # CRITICAL FIX: Convert BFloat16 images to Float32 for DINOv2
            if config.training.use_bf16 and 'image' in batch:
                if batch['image'].dtype == torch.bfloat16:
                    batch['image'] = batch['image'].float()
            
            # Forward pass with mixed precision
            with torch.amp.autocast('cuda', dtype=torch.bfloat16) if config.training.use_bf16 else torch.no_grad():
                outputs = model(batch)
                losses = criterion(outputs, batch)
                loss = losses['total'] if isinstance(losses, dict) else losses
            
            batch_size = batch['image'].shape[0]
            val_metrics['samples'] += batch_size
            val_metrics['loss'] += loss.item() * batch_size
            
            if 'hand_joints' in outputs and 'hand_joints_3d' in batch:
                mpjpe = torch.norm(outputs['hand_joints'] - batch['hand_joints_3d'], dim=-1).mean()
                val_metrics['hand_mpjpe'] += mpjpe.item() * batch_size
    
    # Average validation metrics
    val_metrics['loss'] /= val_metrics['samples']
    val_metrics['hand_mpjpe'] /= val_metrics['samples']
    
    # Update history
    history['train_loss'].append(train_metrics['loss'])
    history['val_loss'].append(val_metrics['loss'])
    history['train_mpjpe'].append(train_metrics['hand_mpjpe'])
    history['val_mpjpe'].append(val_metrics['hand_mpjpe'])
    history['learning_rates'].append(manipulation_trainer.optimizer.param_groups[0]['lr'])
    
    # Save best model
    if val_metrics['loss'] < best_val_loss:
        best_val_loss = val_metrics['loss']
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': manipulation_trainer.optimizer.state_dict(),
            'val_loss': best_val_loss,
            'config': config
        }, f"{config.output_dir}/checkpoints/best_model.pth")
    
    if val_metrics['hand_mpjpe'] < best_val_mpjpe:
        best_val_mpjpe = val_metrics['hand_mpjpe']
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': manipulation_trainer.optimizer.state_dict(),
            'val_mpjpe': best_val_mpjpe,
            'config': config
        }, f"{config.output_dir}/checkpoints/best_mpjpe_model.pth")
    
    # Regular checkpoint
    if (epoch + 1) % config.training.save_freq == 0:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': manipulation_trainer.optimizer.state_dict(),
            'history': history,
            'config': config
        }, f"{config.output_dir}/checkpoints/checkpoint_epoch_{epoch+1}.pth")
    
    # Update learning rate
    manipulation_trainer.scheduler.step()
    
    # Log to wandb
    if config.training.use_wandb:
        wandb.log({
            'epoch': epoch,
            'val/loss': val_metrics['loss'],
            'val/hand_mpjpe': val_metrics['hand_mpjpe'],
            'val/best_loss': best_val_loss,
            'val/best_mpjpe': best_val_mpjpe
        })
    
    # Print epoch summary
    print(f"\nEpoch {epoch+1}/{config.training.num_epochs}:")
    print(f"  Train - Loss: {train_metrics['loss']:.4f}, MPJPE: {train_metrics['hand_mpjpe']:.2f}mm")
    print(f"  Val   - Loss: {val_metrics['loss']:.4f}, MPJPE: {val_metrics['hand_mpjpe']:.2f}mm")
    print(f"  Best  - Loss: {best_val_loss:.4f}, MPJPE: {best_val_mpjpe:.2f}mm")
    print(f"  LR: {manipulation_trainer.optimizer.param_groups[0]['lr']:.2e}")
    print(f"  GPU Memory: {torch.cuda.memory_allocated() / 1e9:.1f} GB / {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Update live plot
    clear_output(wait=True)
    fig = plot_training_progress(history, epoch + 1)
    plt.show()
    
    # Save plot
    fig.savefig(f"{config.output_dir}/training_progress.png", dpi=150, bbox_inches='tight')
    plt.close()

print("\nTraining completed!")
print(f"Final GPU memory usage: {torch.cuda.memory_allocated() / 1e9:.1f} GB")
print(f"Peak GPU memory reserved: {torch.cuda.max_memory_reserved() / 1e9:.1f} GB")
print("\n✓ BFloat16 compatibility maintained throughout training")