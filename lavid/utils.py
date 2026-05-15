import os
import glob
import torch

def get_best_checkpoint(checkpoint_dir):
    """
    Args:
        checkpoint_dir: Directory containing checkpoint files with format 'epoch={epoch}-val_loss={val_loss}.ckpt'

    Returns:
        Path to the best checkpoint file
    """
    checkpoints = glob.glob(os.path.join(checkpoint_dir, '*.ckpt'))

    if not checkpoints:
        raise ValueError(f"No checkpoint files found in {checkpoint_dir}")

    min_val_loss = float('inf')
    best_ckpt = None
    for ckpt in checkpoints:
        basename = os.path.basename(ckpt)
        if "val_loss=" not in basename:
            continue
        try:
            val_loss = float(basename.split("val_loss=")[-1].replace(".ckpt", ""))
        except ValueError:
            continue
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            best_ckpt = ckpt

    if not best_ckpt:
        raise ValueError(f"No valid checkpoints found in {checkpoint_dir}")

    return best_ckpt

def load_teacher(teacher_name, teacher_path, num_classes):
    if teacher_path is None:
        raise ValueError("teacher_path is required")

    if teacher_name == "resnet18":
        from lavid.models.resnet import resnet18
        teacher = resnet18(num_classes=num_classes)
    elif teacher_name == "resnet50" or (teacher_name and "qwen" in teacher_name):
        from lavid.models.resnet import resnet50
        teacher = resnet50(num_classes=num_classes)
    elif teacher_name == "resnet101":
        from lavid.models.resnet import resnet101
        teacher = resnet101(num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported teacher_name: {teacher_name!r}")

    checkpoint = torch.load(teacher_path)
    state_dict = checkpoint.get('state_dict', checkpoint)
    state_dict = {k.replace('model1.', ''): v for k, v in state_dict.items()}
    expected = set(teacher.state_dict().keys())
    teacher.load_state_dict({k: v for k, v in state_dict.items() if k in expected})
    return teacher
