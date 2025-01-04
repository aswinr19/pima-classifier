#!/bin/bash

SESH="pima-libtorch"

tmux has-session -t $SESH 2>/dev/null

if [ $? != 0 ]; then
  tmux new-session -d -s $SESH -n "neovim"
  tmux send-keys -t $SESH:neovim "cd src" C-m
  tmux send-keys -t $SESH:neovim "nvim pima-classifier.cpp" C-m

  tmux new-window -t $SESH -n "run"
  tmux send-keys -t $SESH:run "cd build" C-m

  tmux select-window -t $SESH:neovim
fi

tmux attach-session -t $SESH
