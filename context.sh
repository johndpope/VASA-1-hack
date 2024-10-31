#!/bin/zsh

# Source zshrc to get access to the functions
source ~/.zshrc
cat VASA.py vasa_config.py train_stage_1.py model.py ./reference/vasa.txt ./reference/megaportait-samsung.txt > context.txt

