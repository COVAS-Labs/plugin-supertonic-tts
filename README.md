# COVAS:NEXT Plugin Supertonic TTS

Run TTS locally using Supertone Supertonic TTS models via ONNX Runtime.

## About

This plugin provides offline Text-to-Speech (TTS) capabilities for COVAS:NEXT using the **Supertonic** model. This allows for high-quality, **ultra low-latency** speech synthesis **even on slow systems**, without requiring an internet connection.

**Note:** The model is currently limited to **English** output only.

### Voice Support
The included model supports multiple voices:
- **M1**: Male Voice 1
- **M2**: Male Voice 2
- **F1**: Female Voice 1
- **F2**: Female Voice 2

## Features

- **Offline Synthesis**: No internet connection required.
- **High Quality**: Uses Supertone's Supertonic architecture.
- **Efficient**: Runs on CPU using ONNX Runtime.

## Installation

Download the latest release under the *Releases* section on the right. Follow the instructions on [COVAS:NEXT Plugins](https://ratherrude.github.io/Elite-Dangerous-AI-Integration/plugins/) to install the plugin.

Unpack the plugin into the `plugins` folder in the COVAS:NEXT AppData folder, leading to the following folder structure:
* `plugins`
    * `cn-plugin-supertonic-tts`
        * `cn-plugin-supertonic-tts.py`
        * `requirements.txt`
        * `deps`
        * `model`
        * `__init__.py`
        * etc.
    * `OtherPlugin`

# Development
During development, clone the COVAS:NEXT repository and place your plugin-project in the plugins folder.  
Install the dependencies to your local .venv virtual environment using `pip`, by running this command in the `cn-plugin-supertonic-tts` folder:
```bash
  pip install -r requirements.txt
```

Follow the [COVAS:NEXT Plugin Development Guide](https://ratherrude.github.io/Elite-Dangerous-AI-Integration/plugins/Development/) for more information on developing plugins.

## Packaging
Use the `./pack.ps1` or `./pack.sh` scripts to package the plugin and any Python dependencies in the `deps` folder.

## Releasing
This project includes a GitHub Actions workflow that automatically creates releases. To create a new release:

1. Tag your commit with a version number:
   ```bash
   git tag v1.0.0
   ```
2. Push the tag to GitHub:
   ```bash
   git push origin v1.0.0
   ```

The workflow will automatically build the plugin using the pack script and create a GitHub Release with the zip file attached.
    
## Acknowledgements

 - [COVAS:NEXT](https://github.com/RatherRude/Elite-Dangerous-AI-Integration)
 - [Supertonic](https://huggingface.co/Supertone/supertonic) - For the TTS model.
 - [ONNX Runtime](https://onnxruntime.ai/) - For the underlying inference engine.
