import torch


def show_cuda_infos():
    print("CUDA version:",torch.version.cuda)
    # Prüfen, ob CUDA verfügbar ist
    print("CUDA verfügbar:", torch.cuda.is_available())

    # Anzahl der verfügbaren GPUs
    print("Anzahl GPUs:", torch.cuda.device_count())

    # Name der aktuellen GPU
    if torch.cuda.is_available():
        print("Aktuelle GPU:", torch.cuda.get_device_name(torch.cuda.current_device()))
    else:
        print("Keine GPU verfügbar.")

def activate_CUDA_GPU_if_aviable():
    show_cuda_infos()
        # Stelle sicher, dass CUDA verfügbar ist
    if torch.cuda.is_available():
        device = 0  # Erste GPU
    else:
        device = -1  # CPU, falls CUDA nicht verfügbar ist