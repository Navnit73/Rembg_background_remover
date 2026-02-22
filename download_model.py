import os
from huggingface_hub import hf_hub_download

def download_modnet_model():
    """
    Downloads the MODNet ONNX model from Hugging Face if it doesn't exist locally.
    """
    model_path = "modnet.onnx"
    if not os.path.exists(model_path):
        print("Downloading MODNet ONNX model...")
        try:
            downloaded_path = hf_hub_download(repo_id="gradio/Modnet", filename="modnet.onnx")
            import shutil
            shutil.copy(downloaded_path, model_path)
            print(f"Model downloaded to {model_path} from gradio/Modnet")
        except Exception as e:
            print(f"Failed to download from gradio/Modnet: {e}")
            print("Trying alternative source...")
            try:
                downloaded_path = hf_hub_download(repo_id="onnx-community/modnet-webnn", filename="onnx/model.onnx")
                import shutil
                shutil.copy(downloaded_path, model_path)
                print(f"Model downloaded to {model_path} from onnx-community")
            except Exception as e2:
                 print(f"Failed to download from both sources. Last error: {e2}")
    else:
        print("MODNet model already exists locally.")

if __name__ == "__main__":
    download_modnet_model()
