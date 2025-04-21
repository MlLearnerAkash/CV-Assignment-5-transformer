import torchvision.transforms as transforms

def get_augmentation_set(set_name):
    """
    Returns a torchvision transform pipeline for the specified augmentation set.

    Args:
        set_name (str): One of "basic", "strong", or "mixing".

    Returns:
        torchvision.transforms.Compose: The composed transform pipeline.
    """
    # CIFAR-10 mean and std for normalization
    mean = [0.4914, 0.4822, 0.4465]
    std  = [0.2023, 0.1994, 0.2010]

    if set_name == "basic":
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    elif set_name == "strong":
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandAugment(),
            transforms.ToTensor(),
            
            transforms.RandomErasing(p=0.25),
            
            transforms.Normalize(mean=mean, std=std)
        ])
    elif set_name == "mixing":
        # Same as strong; Mixup/CutMix applied separately in training loop
        transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),  # Color jittering
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),                      # Gaussian blur
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    else:
        raise ValueError(f"Unknown augmentation set '{set_name}'. Choose from 'basic', 'strong', or 'mixing'.")

    return transform
