#!/bin/bash

cd ~/llama.cpp
source ~/entorno_llamacpp/bin/activate

./build/bin/llama-server \
  -m ~/Quimica/phi4_quimica_Q4_K_M.gguf \
  --host 127.0.0.1 \
  --port 8081 \
  -c 40000

  
  
  
