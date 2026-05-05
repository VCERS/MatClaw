"""
Model downloading and caching utilities for MatClaw.

This module handles automatic downloading of model files from GitHub releases
and caches them locally according to XDG Base Directory specification.
"""

import os
import urllib.request
import urllib.error
from pathlib import Path
import sys


# Model URLs from GitHub releases
MODEL_URLS = {
    'elemwiseretro_precursor_predictor': 'https://github.com/VCERS/MatClaw/releases/download/v0.0.3/elemwiseretro-precursor-predictor-v2.0.pt',
    'elemwiseretro_temperature_predictor': 'https://github.com/VCERS/MatClaw/releases/download/v0.0.3/elemwiseretro-temperature-predictor-v2.0.pt',
    'elemwiseretro_temperature_normalizer': 'https://github.com/VCERS/MatClaw/releases/download/v0.0.3/elemwiseretro-temperature-normalizer-v2.0.pt',
    'convnextv2_sem_classifier': 'https://github.com/VCERS/MatClaw/releases/download/v0.0.1/convnextv2_base-finetuned-sem-classifier.pth'
}

# Legacy MatGL models removed from matgl 3.0.0 but still available on GitHub
# These models require downloading all files from GitHub raw content
LEGACY_MATGL_MODELS = {
    'MEGNet-MP-2019.4.1-BandGap-mfi': {
        'base_url': 'https://raw.githubusercontent.com/materialyzeai/matgl/main/pretrained_models/MEGNet-MP-2019.4.1-BandGap-mfi',
        'files': ['model.pt', 'model.json', 'state.pt']
    },
    'MEGNet-MP-2018.6.1-Eform': {
        'base_url': 'https://raw.githubusercontent.com/materialyzeai/matgl/main/pretrained_models/MEGNet-MP-2018.6.1-Eform',
        'files': ['model.pt', 'model.json', 'state.pt']
    }
}


def get_cache_dir() -> Path:
    """
    Get the cache directory for model files.
    
    Uses XDG Base Directory specification:
    - Linux/macOS: ~/.cache/matclaw/models/
    - Windows: %LOCALAPPDATA%\\matclaw\\cache\\models\\
    
    Returns:
        Path object pointing to the cache directory
    """
    if sys.platform == 'win32':
        # Windows: use LOCALAPPDATA
        base = os.environ.get('LOCALAPPDATA', os.path.expanduser('~\\AppData\\Local'))
        cache_dir = Path(base) / 'matclaw' / 'cache' / 'models'
    else:
        # Linux/macOS: use XDG_CACHE_HOME or default ~/.cache
        base = os.environ.get('XDG_CACHE_HOME', os.path.expanduser('~/.cache'))
        cache_dir = Path(base) / 'matclaw' / 'models'
    
    # Create directory if it doesn't exist
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def download_file(url: str, dest_path: Path, show_progress: bool = True) -> None:
    """
    Download a file from URL to destination path with progress indicator.
    
    Args:
        url: URL to download from
        dest_path: Local path to save the file
        show_progress: Whether to show download progress
    """
    try:
        # Download with progress reporting
        def _progress_hook(block_num, block_size, total_size):
            if show_progress and total_size > 0:
                downloaded = block_num * block_size
                percent = min(100, (downloaded / total_size) * 100)
                mb_downloaded = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                print(f'\rDownloading: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)', end='')
        
        # Create temporary file
        temp_path = dest_path.with_suffix(dest_path.suffix + '.tmp')
        
        # Download
        urllib.request.urlretrieve(url, temp_path, _progress_hook if show_progress else None)
        
        if show_progress:
            print()  # New line after progress
        
        # Move to final location
        temp_path.rename(dest_path)
        
    except urllib.error.URLError as e:
        if temp_path.exists():
            temp_path.unlink()
        raise RuntimeError(f"Failed to download {url}: {e}")
    except Exception as e:
        if temp_path.exists():
            temp_path.unlink()
        raise RuntimeError(f"Error downloading {url}: {e}")


def get_model_path(model_key: str, force_download: bool = False) -> Path:
    """
    Get the path to a model file, downloading it if necessary.
    
    This function follows the pattern used by popular ML libraries like
    HuggingFace Transformers and PyTorch Hub:
    1. Check if model exists in cache
    2. If not (or force_download=True), download from GitHub releases
    3. Return path to cached model
    
    Args:
        model_key: Key identifying the model
        force_download: If True, re-download even if file exists
        
    Returns:
        Path to the cached model file
        
    Raises:
        ValueError: If model_key is not recognized
        RuntimeError: If download fails
    """
    if model_key not in MODEL_URLS:
        raise ValueError(
            f"Unknown model key: {model_key}. "
            f"Valid keys are: {', '.join(MODEL_URLS.keys())}"
        )
    
    # Get cache directory and model path
    cache_dir = get_cache_dir()
    url = MODEL_URLS[model_key]
    filename = url.split('/')[-1]
    model_path = cache_dir / filename
    
    # Download if needed
    if not model_path.exists() or force_download:
        print(f"Downloading {model_key} from GitHub releases...")
        print(f"URL: {url}")
        print(f"Cache location: {model_path}")
        download_file(url, model_path)
        print(f"[OK] Downloaded {model_key}")
    
    return model_path


def clear_cache() -> None:
    """
    Clear all cached model files, including legacy MatGL models.
    """
    cache_dir = get_cache_dir()
    if cache_dir.exists():
        # Clear regular model files
        for ext in ['*.sav', '*.pt', '*.pth']:
            for file in cache_dir.glob(ext):
                file.unlink()
        
        # Clear legacy matgl model directories
        legacy_dir = cache_dir / 'matgl_legacy'
        if legacy_dir.exists():
            import shutil
            shutil.rmtree(legacy_dir)
        
        print(f"Cleared cache directory: {cache_dir}")
    else:
        print(f"Cache directory does not exist: {cache_dir}")


def get_cache_info() -> dict:
    """
    Get information about cached models.
    
    Returns:
        Dictionary with cache directory and list of cached files with sizes
    """
    cache_dir = get_cache_dir()
    cached_files = []
    
    if cache_dir.exists():
        for ext in ['*.sav', '*.pt', '*.pth']:
            for file in cache_dir.glob(ext):
                size_mb = file.stat().st_size / (1024 * 1024)
                cached_files.append({
                    'name': file.name,
                    'path': str(file),
                    'size_mb': round(size_mb, 2)
                })
    
    return {
        'cache_dir': str(cache_dir),
        'cached_files': cached_files,
        'total_size_mb': round(sum(f['size_mb'] for f in cached_files), 2)
    }


def get_legacy_matgl_model_dir(model_name: str, force_download: bool = False, verbose: bool = True) -> Path:
    """
    Get the directory for a legacy MatGL model, downloading all files if necessary.
    
    Legacy MatGL models (like MEGNet) were removed from matgl 3.0.0 but still exist
    on GitHub. This function downloads all necessary files for these models.
    
    Args:
        model_name: Name of the legacy model (e.g., 'MEGNet-MP-2019.4.1-BandGap-mfi')
        force_download: If True, re-download even if files exist
        
    Returns:
        Path to the directory containing the model files
        
    Raises:
        ValueError: If model_name is not a recognized legacy model
        RuntimeError: If download fails
    """
    if model_name not in LEGACY_MATGL_MODELS:
        raise ValueError(
            f"Unknown legacy MatGL model: {model_name}. "
            f"Valid models are: {', '.join(LEGACY_MATGL_MODELS.keys())}"
        )
    
    # Get cache directory for this specific model
    cache_dir = get_cache_dir() / 'matgl_legacy' / model_name
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    model_info = LEGACY_MATGL_MODELS[model_name]
    base_url = model_info['base_url']
    files = model_info['files']
    
    # Download each required file
    all_exist = all((cache_dir / f).exists() for f in files)
    
    if not all_exist or force_download:
        import sys
        out = sys.stderr if verbose else open(os.devnull, 'w')
        
        if verbose:
            print(f"Downloading legacy MatGL model: {model_name}", file=out)
            print(f"Cache location: {cache_dir}", file=out)
        
        for filename in files:
            file_url = f"{base_url}/{filename}"
            dest_path = cache_dir / filename
            
            if not dest_path.exists() or force_download:
                if verbose:
                    print(f"  Downloading {filename}...", file=out)
                try:
                    download_file(file_url, dest_path, show_progress=False)
                except Exception as e:
                    if verbose:
                        print(f"  [X] Failed to download {filename}: {e}", file=out)
                    raise
                if verbose:
                    print(f"  [OK] Downloaded {filename}", file=out)
        
        if verbose:
            print(f"[OK] Downloaded all files for {model_name}", file=out)
    
    return cache_dir


def load_legacy_matgl_model(model_name: str, verbose: bool = True):
    """
    Load a legacy MatGL model from local cache.
    
    This function downloads (if needed) and loads legacy MatGL models that were
    removed from matgl 3.0.0 but still exist on GitHub.
    
    Args:
        model_name: Name of the legacy model (e.g., 'MEGNet-MP-2019.4.1-BandGap-mfi')
        verbose: Whether to print download progress messages
        
    Returns:
        Loaded MatGL model ready for prediction
        
    Raises:
        ValueError: If model_name is not recognized
        RuntimeError: If model loading fails
    """
    import torch
    import json
    
    # Get model directory (downloads if needed)
    try:
        model_dir = get_legacy_matgl_model_dir(model_name, verbose=verbose)
    except Exception as e:
        raise RuntimeError(f"Failed to get legacy model directory: {e}")
    
    # Load model files
    try:
        model_pt_path = model_dir / 'model.pt'
        state_pt_path = model_dir / 'state.pt'
        model_json_path = model_dir / 'model.json'
        
        if not model_pt_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_pt_path}")
        
        # Load model.pt which contains:
        # - 'model': serialized model architecture (dict with @class, @module, init_args)
        # - 'target_transformer': Normalizer object for output scaling
        model_data = torch.load(str(model_pt_path), map_location='cpu', weights_only=False)
        
        # Load model.json which contains the full serialization info
        if not model_json_path.exists():
            raise FileNotFoundError(f"Model metadata file not found: {model_json_path}")
        
        with open(model_json_path, 'r') as f:
            model_info = json.load(f)
        
        # Use matgl's deserialization approach
        # Get the class from @module and @class fields
        modname = model_info.get('@module', '')
        classname = model_info.get('@class', '')
        
        if not modname or not classname:
            raise ValueError(f"Model JSON missing @module or @class: {model_info}")
        
        # Import the model class
        try:
            mod = __import__(modname, globals(), locals(), [classname], 0)
            cls_ = getattr(mod, classname)
        except (ImportError, AttributeError) as e:
            raise ImportError(f"Failed to import {modname}.{classname}: {e}")
        
        # Load the model using the class's load method
        # Pass the directory containing all model files
        fpaths = {
            'model.pt': model_pt_path,
            'state.pt': state_pt_path if state_pt_path.exists() else None,
            'model.json': model_json_path
        }
        
        try:
            # Call the class's load method (standard matgl pattern)
            model = cls_.load(fpaths)
        except Exception as e:
            raise RuntimeError(f"Failed to load model using {classname}.load(): {e}")
        
        return model
        
    except Exception as e:
        raise RuntimeError(f"Failed to load model from {model_dir}: {e}")


def get_cache_info() -> dict:
    """
    Get information about cached models, including legacy MatGL models.
    
    Returns:
        Dictionary with cache directory and list of cached files with sizes
    """
    cache_dir = get_cache_dir()
    cached_files = []
    legacy_models = []
    
    if cache_dir.exists():
        # Regular model files
        for ext in ['*.sav', '*.pt', '*.pth']:
            for file in cache_dir.glob(ext):
                size_mb = file.stat().st_size / (1024 * 1024)
                cached_files.append({
                    'name': file.name,
                    'path': str(file),
                    'size_mb': round(size_mb, 2)
                })
        
        # Legacy MatGL models
        legacy_dir = cache_dir / 'matgl_legacy'
        if legacy_dir.exists():
            for model_dir in legacy_dir.iterdir():
                if model_dir.is_dir():
                    total_size = sum(f.stat().st_size for f in model_dir.glob('*') if f.is_file())
                    size_mb = total_size / (1024 * 1024)
                    files = [f.name for f in model_dir.glob('*') if f.is_file()]
                    legacy_models.append({
                        'name': model_dir.name,
                        'path': str(model_dir),
                        'files': files,
                        'size_mb': round(size_mb, 2)
                    })
    
    total_size = sum(f['size_mb'] for f in cached_files) + sum(m['size_mb'] for m in legacy_models)
    
    return {
        'cache_dir': str(cache_dir),
        'cached_files': cached_files,
        'legacy_matgl_models': legacy_models,
        'total_size_mb': round(total_size, 2)
    }


if __name__ == '__main__':
    # Simple CLI for testing
    import sys
    
    if len(sys.argv) > 1:
        command = sys.argv[1]
        
        if command == 'info':
            info = get_cache_info()
            print(f"\nCache Directory: {info['cache_dir']}")
            print(f"Total Size: {info['total_size_mb']} MB\n")
            print("Cached Models:")
            for file in info['cached_files']:
                print(f"  - {file['name']} ({file['size_mb']} MB)")
            
            if info['legacy_matgl_models']:
                print("\nLegacy MatGL Models:")
                for model in info['legacy_matgl_models']:
                    print(f"  - {model['name']} ({model['size_mb']} MB)")
                    print(f"    Files: {', '.join(model['files'])}")
        
        elif command == 'clear':
            clear_cache()
        
        elif command == 'download':
            print("Downloading all models...")
            for key in MODEL_URLS.keys():
                get_model_path(key)
            print("\n[OK] All models downloaded")
        
        else:
            print(f"Unknown command: {command}")
            print("Usage: python model_downloader.py [info|clear|download]")
    else:
        info = get_cache_info()
        print(f"Cache Directory: {info['cache_dir']}")
        print(f"Cached Models: {len(info['cached_files'])}")
        print("\nUsage: python model_downloader.py [info|clear|download]")
