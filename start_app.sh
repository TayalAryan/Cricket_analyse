#!/bin/bash

# Set library paths for OpenCV and MediaPipe dependencies
export LD_LIBRARY_PATH="/nix/store/*/lib:/nix/store/*/lib64:$LD_LIBRARY_PATH"
export LD_PRELOAD=""

# Add GCC library paths
for gcc_path in /nix/store/*gcc*/lib*; do
    if [ -d "$gcc_path" ]; then
        export LD_LIBRARY_PATH="$gcc_path:$LD_LIBRARY_PATH"
    fi
done

# Start Streamlit
streamlit run app.py --server.port 5000