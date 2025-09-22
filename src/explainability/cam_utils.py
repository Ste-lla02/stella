import os
import torch

def load_checkpoint_into_model(model, ckpt_path, device):
    """
    Carica i pesi dal checkpoint dentro 'model' e lo mette in eval().
    Supporta:
      - torch.save(model.state_dict())
      - torch.save({'model_state_dict': ...})
      - torch.save({'state_dict': ...})
      - torch.save(model)  # modulo intero
    """
    if not ckpt_path or not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint non trovato: {ckpt_path}")

    state = torch.load(ckpt_path, map_location=device)

    if isinstance(state, torch.nn.Module):
        # È stato salvato l'intero modulo
        model.load_state_dict(state.state_dict())
    elif isinstance(state, dict) and 'state_dict' in state:
        model.load_state_dict(state['state_dict'])
    elif isinstance(state, dict) and 'model_state_dict' in state:
        model.load_state_dict(state['model_state_dict'])
    elif isinstance(state, dict):
        # Probabile state_dict puro
        model.load_state_dict(state)
    else:
        raise RuntimeError("Formato checkpoint non riconosciuto.")

    #model.eval()
    model.to(device).eval()
    return model

def prepare_ckpt_path(conf):
    ckpt_path = conf.get('best_predictor_model_path')
    return ckpt_path

def prepare_explain_input_dir(conf):
    input_dir = conf.get('gradcam_input')
    os.makedirs(input_dir, exist_ok=True)
    return input_dir

def prepare_explain_out_dir(conf):
    out_dir = conf.get('gradcam_output')
    os.makedirs(out_dir, exist_ok=True)
    return out_dir
